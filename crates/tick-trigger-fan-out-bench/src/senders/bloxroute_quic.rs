//! bloXroute **QUIC** sender - fastest bloXroute path, multi-region fan-out over
//! raw QUIC (quinn). Built per the official `solana-trader-quic-client-rust`.
//!
//! Connection: ALPN `solana-trader-submit-v1`, default UDP port 443, real server
//! verification via **webpki roots**, and **mTLS client auth** - the client cert
//! chain + private key are the bloXroute portal artifacts
//! (`external_gateway_cert.pem` + `external_gateway_key.pem`). There is no
//! Authorization header / byte handshake; the cert IS the identity. Keep-alive
//! 5s, idle 30s.
//!
//! Per tx (hot path): open a **unidirectional** stream, write the raw
//! `bincode(tx)` bytes (≤1232), `finish()`. Fire-and-forget - no response;
//! signature taken locally from `tx.signatures[0]`.
//!
//! **Multi-region fan-out** (bloXroute recommends "submit to your closest region
//! AND the global endpoint simultaneously"): one persistent QUIC connection per
//! host; the SAME tx is sent on all concurrently, first write to complete wins.
//! ONE in-tx tip serves all (preparer via `tip_accounts_for(BloxrouteQuic)`, min
//! 1_000_000 lamports). `outbound_ips[0]` pins the egress IP; `min_send_interval_ms`
//! paces distinct triggers. NOTE: the official `BlxEndpoint` enum lists
//! germany/amsterdam/uk/ny/la/tokyo but NOT `global` - `global.solana.dex.blxrbdn.com:443`
//! over QUIC is unconfirmed; the warmup logs per-host whether the handshake/mTLS
//! succeeded, so a non-QUIC `global` shows up clearly.

use super::{SendOutcome, TxSender};
use anyhow::{Context, Result};
use async_trait::async_trait;
use quinn::crypto::rustls::QuicClientConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use solana_sdk::transaction::Transaction;
use std::io::{BufReader, Cursor};
use std::net::{SocketAddr, ToSocketAddrs};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const QUIC_SUBMIT_ALPN: &[u8] = b"solana-trader-submit-v1";
const MAX_TX_SIZE: usize = 1232;
const DEFAULT_PORT: u16 = 443;

fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

fn substitute_region(template: &str, region: &str) -> String {
    template.replace("{region}", region)
}

/// Parse a PEM cert chain (one or more certificates).
fn certificates_from_pem(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>> {
    let mut reader = BufReader::new(Cursor::new(pem));
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<Result<Vec<_>, _>>()
        .context("parse client cert PEM")?;
    if certs.is_empty() {
        anyhow::bail!("no certificates found in client cert PEM");
    }
    Ok(certs)
}

/// Parse a PEM private key (PKCS#1 / PKCS#8 / SEC1).
fn private_key_from_pem(pem: &[u8]) -> Result<PrivateKeyDer<'static>> {
    let mut reader = BufReader::new(Cursor::new(pem));
    while let Some(item) =
        rustls_pemfile::read_one(&mut reader).context("read client key PEM")?
    {
        match item {
            rustls_pemfile::Item::Pkcs1Key(k) => return Ok(PrivateKeyDer::Pkcs1(k)),
            rustls_pemfile::Item::Pkcs8Key(k) => return Ok(PrivateKeyDer::Pkcs8(k)),
            rustls_pemfile::Item::Sec1Key(k) => return Ok(PrivateKeyDer::Sec1(k)),
            _ => continue,
        }
    }
    anyhow::bail!("no private key found in client key PEM")
}

/// Build the quinn client config: webpki server roots + mTLS client cert, ALPN
/// `solana-trader-submit-v1`, keep-alive 5s / idle 30s. Explicit ring provider.
fn build_client_config(
    cert_chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> Result<quinn::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_client_auth_cert(cert_chain, key)
        .context("set bloxroute mTLS client cert")?;
    crypto.alpn_protocols = vec![QUIC_SUBMIT_ALPN.to_vec()];

    let quic_crypto = QuicClientConfig::try_from(Arc::new(crypto))?;
    let mut client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    transport.max_idle_timeout(Some(quinn::IdleTimeout::try_from(Duration::from_secs(30))?));
    client_config.transport_config(Arc::new(transport));
    Ok(client_config)
}

/// Resolve `host[:port]` (default 443) to `(SocketAddr, server_name)`. The
/// server_name (hostname) must match the cert - verification is real.
fn resolve_host(hp: &str) -> Result<(SocketAddr, String)> {
    if let Ok(a) = hp.parse::<SocketAddr>() {
        return Ok((a, a.ip().to_string()));
    }
    let (host, port) = match hp.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().unwrap_or(DEFAULT_PORT)),
        None => (hp, DEFAULT_PORT),
    };
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve {hp}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no addresses for {hp}"))?;
    Ok((addr, host.to_string()))
}

/// Local UDP bind address: first whitelisted `outbound_ips` entry, else the
/// unspecified address matching the target family.
fn bind_addr(outbound_ips: &[String], target: SocketAddr) -> Result<SocketAddr> {
    if let Some(s) = outbound_ips.first() {
        let ip = std::net::IpAddr::from_str(s)
            .with_context(|| format!("invalid outbound_ip {s:?}"))?;
        return Ok(SocketAddr::new(ip, 0));
    }
    Ok(if target.is_ipv4() {
        "0.0.0.0:0".parse().unwrap()
    } else {
        "[::]:0".parse().unwrap()
    })
}

struct Target {
    addr: SocketAddr,
    server_name: String,
    host_display: String,
    conn: Arc<Mutex<Option<quinn::Connection>>>,
}

enum FanoutReply {
    Success { host: String, send_ack_at: Instant },
    Error { host: String, send_ack_at: Instant, error: String },
}

/// Live connection for a target - reuse if alive, else connect (mTLS handshake).
async fn get_conn(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    server_name: &str,
    slot: &Mutex<Option<quinn::Connection>>,
) -> Result<quinn::Connection> {
    let mut g = slot.lock().await;
    if let Some(c) = g.as_ref() {
        if c.close_reason().is_none() {
            return Ok(c.clone());
        }
    }
    let c = endpoint.connect(addr, server_name)?.await?;
    *g = Some(c.clone());
    Ok(c)
}

pub struct BloxrouteQuicSender {
    id: u8,
    name: String,
    endpoint_template: String,
    targets: Vec<Target>,
    endpoint: quinn::Endpoint,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl BloxrouteQuicSender {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint_template: impl Into<String>,
        regions: Vec<String>,
        outbound_ips: Vec<String>,
        tls_cert_path: &str,
        tls_key_path: &str,
        min_send_interval_ms: u64,
    ) -> Result<Self> {
        let endpoint_template = endpoint_template.into();
        if tls_cert_path.is_empty() || tls_key_path.is_empty() {
            anyhow::bail!(
                "bloxroute_quic requires tls_cert_path + tls_key_path (portal mTLS artifacts)"
            );
        }
        let cert_chain = certificates_from_pem(
            &std::fs::read(tls_cert_path).with_context(|| format!("read {tls_cert_path}"))?,
        )?;
        let key = private_key_from_pem(
            &std::fs::read(tls_key_path).with_context(|| format!("read {tls_key_path}"))?,
        )?;

        let region_hosts: Vec<String> = if regions.is_empty() {
            vec![endpoint_template.clone()]
        } else {
            regions
                .iter()
                .map(|r| substitute_region(&endpoint_template, r))
                .collect()
        };
        let mut targets = Vec::with_capacity(region_hosts.len());
        for h in &region_hosts {
            let (addr, server_name) = resolve_host(h)?;
            targets.push(Target {
                addr,
                server_name,
                host_display: h.clone(),
                conn: Arc::new(Mutex::new(None)),
            });
        }
        let first_addr = targets
            .first()
            .map(|t| t.addr)
            .unwrap_or_else(|| "0.0.0.0:0".parse().unwrap());
        let bind = bind_addr(&outbound_ips, first_addr)?;

        let client_config = build_client_config(cert_chain, key)?;
        let mut endpoint = quinn::Endpoint::client(bind)?;
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            id,
            name: name.into(),
            endpoint_template,
            targets,
            endpoint,
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        })
    }

    /// Fire-and-forget pre-warm + mTLS health check: connect each target at
    /// startup. mTLS auth happens during the handshake, so a rejected cert or a
    /// host that does not speak QUIC (e.g. `global`) is logged per host instead
    /// of silently dropping sends.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        for t in &self.targets {
            let endpoint = self.endpoint.clone();
            let addr = t.addr;
            let server_name = t.server_name.clone();
            let host = t.host_display.clone();
            let slot = t.conn.clone();
            let name = self.name.clone();
            handle.spawn(async move {
                match get_conn(&endpoint, addr, &server_name, &slot).await {
                    Ok(_) => tracing::info!(sender = %name, host = %host, "bloxroute_quic warmup: connected (mTLS ok)"),
                    Err(e) => tracing::warn!(sender = %name, host = %host, error = %e, "bloxroute_quic warmup: connect/mTLS FAILED - cert rejected or host has no QUIC"),
                }
            });
        }
    }
}

#[async_trait]
impl TxSender for BloxrouteQuicSender {
    fn id(&self) -> u8 {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn endpoint_url(&self) -> &str {
        &self.endpoint_template
    }
    fn protocol(&self) -> &'static str {
        "BLOXROUTE_QUIC"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();

        if self.min_send_interval > Duration::ZERO {
            let now = Instant::now();
            let mut last = self.last_send_at.lock();
            if throttled(*last, now, self.min_send_interval) {
                return SendOutcome {
                    send_at: now,
                    send_ack_at: Some(now),
                    signature,
                    http_status: None,
                    rpc_err_code: None,
                    rpc_err_message: None,
                    provider_request_id: None,
                    error: Some("throttled_local".into()),
                    endpoint_url_used: None,
                };
            }
            *last = Some(now);
        }

        let send_at = Instant::now();
        if self.targets.is_empty() {
            return SendOutcome {
                send_at,
                send_ack_at: None,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some("no regions configured".into()),
                endpoint_url_used: None,
            };
        }

        let bytes = bincode::serialize(tx).unwrap_or_default();
        if bytes.len() > MAX_TX_SIZE {
            return SendOutcome {
                send_at,
                send_ack_at: None,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some(format!("tx too large: {} > {MAX_TX_SIZE}", bytes.len())),
                endpoint_url_used: Some(self.endpoint_template.clone()),
            };
        }
        let bytes: Arc<Vec<u8>> = Arc::new(bytes);
        let (tx_first, mut rx_first) =
            tokio::sync::mpsc::channel::<FanoutReply>(self.targets.len());

        for t in &self.targets {
            let endpoint = self.endpoint.clone();
            let addr = t.addr;
            let server_name = t.server_name.clone();
            let host = t.host_display.clone();
            let slot = t.conn.clone();
            let bytes = bytes.clone();
            let tx_first = tx_first.clone();
            tokio::spawn(async move {
                let reply = async {
                    let conn = get_conn(&endpoint, addr, &server_name, &slot).await?;
                    let mut s = conn.open_uni().await?;
                    s.write_all(&bytes).await?;
                    s.finish()?;
                    Ok::<(), anyhow::Error>(())
                }
                .await;
                let send_ack_at = Instant::now();
                let reply = match reply {
                    Ok(()) => FanoutReply::Success { host, send_ack_at },
                    Err(e) => {
                        // Drop this target's connection so the next send reconnects.
                        *slot.lock().await = None;
                        FanoutReply::Error { host, send_ack_at, error: format!("quic: {e}") }
                    }
                };
                let _ = tx_first.send(reply).await;
            });
        }
        drop(tx_first);

        // Take the first SUCCESS; fall back to the last error if all fail.
        let mut last_err: Option<(String, Instant, String)> = None;
        while let Some(reply) = rx_first.recv().await {
            match reply {
                FanoutReply::Success { host, send_ack_at } => {
                    return SendOutcome {
                        send_at,
                        send_ack_at: Some(send_ack_at),
                        signature,
                        http_status: None,
                        rpc_err_code: None,
                        rpc_err_message: None,
                        provider_request_id: None,
                        error: None,
                        endpoint_url_used: Some(host),
                    };
                }
                FanoutReply::Error { host, send_ack_at, error } => {
                    last_err = Some((host, send_ack_at, error));
                }
            }
        }

        match last_err {
            Some((host, send_ack_at, error)) => SendOutcome {
                send_at,
                send_ack_at: Some(send_ack_at),
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some(error),
                endpoint_url_used: Some(host),
            },
            None => SendOutcome {
                send_at,
                send_ack_at: None,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some("all fan-out tasks dropped without reply".into()),
                endpoint_url_used: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_sdk::hash::Hash;
    use solana_sdk::message::Message;
    use solana_sdk::signature::{Keypair, Signer};
    use solana_system_interface::instruction as system_instruction;

    fn sample_tx() -> Transaction {
        let payer = Keypair::new();
        let ix = system_instruction::transfer(&payer.pubkey(), &payer.pubkey(), 1);
        let msg = Message::new(&[ix], Some(&payer.pubkey()));
        let mut tx = Transaction::new_unsigned(msg);
        tx.sign(&[&payer], Hash::new_unique());
        tx
    }

    /// Generate a self-signed EC P-256 cert + key as PEM (for config-build tests).
    fn gen_cert_key_pem() -> (String, String) {
        let kp = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let params = rcgen::CertificateParams::new(vec![]).unwrap();
        let cert = params.self_signed(&kp).unwrap();
        (cert.pem(), kp.serialize_pem())
    }

    fn write_temp(name: &str, contents: &str) -> String {
        let path = std::env::temp_dir().join(format!("blxq_{}_{}", std::process::id(), name));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn substitute_region_replaces_placeholder() {
        assert_eq!(
            substitute_region("{region}.solana.dex.blxrbdn.com:443", "germany"),
            "germany.solana.dex.blxrbdn.com:443"
        );
    }

    #[test]
    fn resolve_host_defaults_to_443_and_keeps_server_name() {
        let (addr, name) = resolve_host("127.0.0.1").unwrap();
        assert_eq!(addr.port(), 443);
        assert_eq!(name, "127.0.0.1");
        let (addr2, _) = resolve_host("127.0.0.1:11100").unwrap();
        assert_eq!(addr2.port(), 11100);
    }

    #[test]
    fn pem_round_trip_parses_cert_and_key() {
        let (cert_pem, key_pem) = gen_cert_key_pem();
        assert_eq!(certificates_from_pem(cert_pem.as_bytes()).unwrap().len(), 1);
        assert!(private_key_from_pem(key_pem.as_bytes()).is_ok());
    }

    #[test]
    fn build_client_config_succeeds_with_mtls_cert() {
        let (cert_pem, key_pem) = gen_cert_key_pem();
        let chain = certificates_from_pem(cert_pem.as_bytes()).unwrap();
        let key = private_key_from_pem(key_pem.as_bytes()).unwrap();
        assert!(build_client_config(chain, key).is_ok());
    }

    #[test]
    fn throttled_logic() {
        let now = Instant::now();
        let interval = Duration::from_millis(1000);
        assert!(throttled(Some(now - Duration::from_millis(100)), now, interval));
        assert!(!throttled(Some(now - Duration::from_millis(1500)), now, interval));
        assert!(!throttled(Some(now), now, Duration::ZERO));
        assert!(!throttled(None, now, interval));
    }

    #[test]
    fn new_without_cert_paths_errors() {
        let r = BloxrouteQuicSender::new(
            8,
            "blx-quic",
            "germany.solana.dex.blxrbdn.com:443",
            vec![],
            vec![],
            "",
            "",
            0,
        );
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn new_builds_endpoint_with_regions_and_labels() {
        let (cert_pem, key_pem) = gen_cert_key_pem();
        let cert_path = write_temp("cert.pem", &cert_pem);
        let key_path = write_temp("key.pem", &key_pem);
        let s = BloxrouteQuicSender::new(
            8,
            "bloxroute-quic-fra",
            "{region}.solana.dex.blxrbdn.com:443",
            vec!["germany".into(), "amsterdam".into()],
            vec![],
            &cert_path,
            &key_path,
            0,
        )
        .unwrap();
        assert_eq!(s.protocol(), "BLOXROUTE_QUIC");
        assert_eq!(s.targets.len(), 2);
        assert_eq!(s.targets[0].host_display, "germany.solana.dex.blxrbdn.com:443");
        assert_eq!(s.endpoint_url(), "{region}.solana.dex.blxrbdn.com:443");
        let _ = std::fs::remove_file(cert_path);
        let _ = std::fs::remove_file(key_path);
    }

    #[tokio::test]
    async fn send_throttles_when_last_send_recent() {
        let (cert_pem, key_pem) = gen_cert_key_pem();
        let cert_path = write_temp("cert2.pem", &cert_pem);
        let key_path = write_temp("key2.pem", &key_pem);
        let s = BloxrouteQuicSender::new(
            8,
            "blx-quic",
            "127.0.0.1:443",
            vec![],
            vec![],
            &cert_path,
            &key_path,
            10_000,
        )
        .unwrap();
        *s.last_send_at.lock() = Some(Instant::now());
        let outcome = s.send(&sample_tx()).await;
        assert_eq!(outcome.error.as_deref(), Some("throttled_local"));
        let _ = std::fs::remove_file(cert_path);
        let _ = std::fs::remove_file(key_path);
    }
}
