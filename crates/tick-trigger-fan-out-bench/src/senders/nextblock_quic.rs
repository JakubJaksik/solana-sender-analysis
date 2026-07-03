//! NextBlock **QUIC** sender - the fastest NextBlock path (raw QUIC over quinn).
//! Per the official docs (docs.nextblock.io/api/quic-transaction-submission +
//! the Rust example), NOT a guess.
//!
//! Connection (once, reused): ALPN `nb-tx/1`, standard TLS (no client cert -
//! `with_no_client_auth`), connect to `{region}.nextblock.io:11100`. Idle 60s,
//! keep-alive 15s.
//!
//! Auth (ONCE per connection, right after connect): open a **bidirectional**
//! stream, write the API key as raw UTF-8 bytes, **finish the write side**, then
//! `read_exact` ONE byte - `0x00` = authenticated. Must complete before any tx.
//! The api_key is secret: it goes only on this stream, never in logs.
//!
//! Per tx (hot path): open a **unidirectional** stream, write the raw
//! `bincode(tx)` bytes (≤1232), `finish()`. Fire-and-forget - no response;
//! signature taken locally from `tx.signatures[0]`. Invalid / rate-limited tx
//! are dropped server-side with no response, so a `min_send_interval_ms`
//! throttle paces distinct triggers.
//!
//! Tip is MANDATORY (one in-tx `SystemProgram::transfer` to a NextBlock wallet,
//! added by the preparer via `tip_accounts_for(NextblockQuic)`, min 1_000_000
//! lamports) - the QUIC docs omit it but the tip is in-tx regardless of transport.
//!
//! TLS note: the official example verifies the server cert against native roots.
//! We reuse the codebase's skip-server-verify pattern (as for AllenHark/Astralane
//! QUIC) to avoid a roots dependency - the connection is still encrypted and the
//! api-key byte-handshake is the auth boundary; txs are signed. Bind the local
//! UDP socket to a whitelisted egress IP via `outbound_ips[0]` if set.

use super::{SendOutcome, TxSender};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rustls::pki_types::CertificateDer;
use solana_sdk::transaction::Transaction;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// ALPN protocol id for NextBlock tx submission.
const ALPN_NB_TX: &[u8] = b"nb-tx/1";
/// Auth-handshake success byte.
const AUTH_OK: u8 = 0x00;
/// Max Solana tx size accepted by the QUIC endpoint.
const MAX_TX_SIZE: usize = 1232;

fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// SNI / connect server name: the host part of `host:port`. (Skip-verify makes
/// matching non-critical, but quinn requires a valid name for the handshake.)
fn server_name_of(endpoint: &str) -> String {
    endpoint
        .rsplit_once(':')
        .map(|(h, _)| h.to_string())
        .unwrap_or_else(|| endpoint.to_string())
}

/// Resolve `host:port` (hostname or IP) to a `SocketAddr`.
fn resolve_addr(endpoint: &str) -> Result<SocketAddr> {
    if let Ok(a) = SocketAddr::from_str(endpoint) {
        return Ok(a);
    }
    use std::net::ToSocketAddrs;
    endpoint
        .to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next())
        .ok_or_else(|| anyhow::anyhow!("cannot resolve nextblock endpoint: {endpoint}"))
}

/// Bind address for the local UDP socket: first whitelisted `outbound_ips` entry
/// (port 0), or the unspecified address when none configured.
fn bind_addr(outbound_ips: &[String]) -> Result<SocketAddr> {
    match outbound_ips.first() {
        Some(s) => {
            let ip = std::net::IpAddr::from_str(s)
                .with_context(|| format!("invalid outbound_ip {s:?}"))?;
            Ok(SocketAddr::new(ip, 0))
        }
        None => Ok("0.0.0.0:0".parse().unwrap()),
    }
}

/// rustls danger verifier: accept any server cert (encryption only; auth is the
/// byte handshake). Mirrors the other QUIC senders.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        use rustls::SignatureScheme as S;
        vec![
            S::ECDSA_NISTP256_SHA256,
            S::ECDSA_NISTP384_SHA384,
            S::RSA_PSS_SHA256,
            S::RSA_PSS_SHA384,
            S::RSA_PSS_SHA512,
            S::RSA_PKCS1_SHA256,
            S::RSA_PKCS1_SHA384,
            S::RSA_PKCS1_SHA512,
            S::ED25519,
        ]
    }
}

/// quinn client config: ALPN `nb-tx/1`, skip-server-verify, NO client cert,
/// idle 60s / keep-alive 15s (per NextBlock docs). Explicit ring provider so it
/// coexists with the other QUIC senders.
fn build_client_config() -> Result<quinn::ClientConfig> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN_NB_TX.to_vec()];

    let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))?;
    let mut client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(quinn::IdleTimeout::try_from(Duration::from_secs(60))?));
    transport.keep_alive_interval(Some(Duration::from_secs(15)));
    client_config.transport_config(Arc::new(transport));
    Ok(client_config)
}

/// Connect, then run the once-per-connection auth handshake (bidi: write key,
/// finish, read 1 byte == 0x00). Returns an authenticated connection.
async fn connect_and_auth(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    server_name: &str,
    api_key: &str,
) -> Result<quinn::Connection> {
    let conn = endpoint.connect(addr, server_name)?.await?;
    let (mut send, mut recv) = conn.open_bi().await.context("open auth bi-stream")?;
    send.write_all(api_key.as_bytes()).await.context("write api key")?;
    send.finish().context("finish auth write side")?;
    let mut buf = [0u8; 1];
    recv.read_exact(&mut buf).await.context("read auth response byte")?;
    if buf[0] != AUTH_OK {
        anyhow::bail!("nextblock quic auth rejected (response byte {:#04x})", buf[0]);
    }
    Ok(conn)
}

pub struct NextblockQuicSender {
    id: u8,
    name: String,
    addr: SocketAddr,
    server_name: String,
    /// `host:port` (no key) - returned by `endpoint_url()`.
    endpoint_display: String,
    /// Raw API key (secret) - sent only on the auth handshake stream.
    api_key: String,
    endpoint: quinn::Endpoint,
    conn: Arc<tokio::sync::Mutex<Option<quinn::Connection>>>,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl NextblockQuicSender {
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint_url: impl Into<String>,
        outbound_ips: Vec<String>,
        api_key: impl Into<String>,
        min_send_interval_ms: u64,
    ) -> Result<Self> {
        let endpoint_display = endpoint_url.into();
        let api_key = api_key.into();
        let addr = resolve_addr(&endpoint_display)?;
        let server_name = server_name_of(&endpoint_display);
        let bind = bind_addr(&outbound_ips)?;
        let client_config = build_client_config()?;
        let mut endpoint = quinn::Endpoint::client(bind)?;
        endpoint.set_default_client_config(client_config);
        Ok(Self {
            id,
            name: name.into(),
            addr,
            server_name,
            endpoint_display,
            api_key,
            endpoint,
            conn: Arc::new(tokio::sync::Mutex::new(None)),
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        })
    }

    /// Live, authenticated connection - reuse if alive, else connect + auth.
    async fn connection(&self) -> Result<quinn::Connection> {
        let mut guard = self.conn.lock().await;
        if let Some(c) = guard.as_ref() {
            if c.close_reason().is_none() {
                return Ok(c.clone());
            }
        }
        let c = connect_and_auth(&self.endpoint, self.addr, &self.server_name, &self.api_key).await?;
        *guard = Some(c.clone());
        Ok(c)
    }

    /// Open a uni stream, write the raw bytes, finish. Fire-and-forget.
    async fn try_send(&self, bytes: &[u8]) -> Result<()> {
        let conn = self.connection().await?;
        let mut s = conn.open_uni().await?;
        s.write_all(bytes).await?;
        s.finish()?;
        Ok(())
    }

    /// Fire-and-forget pre-warm + auth health check: connect and run the auth
    /// handshake at startup. The single-byte response gives a DEFINITIVE auth
    /// result, so a bad key/IP is logged clearly instead of silently dropping.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let endpoint = self.endpoint.clone();
        let addr = self.addr;
        let server_name = self.server_name.clone();
        let api_key = self.api_key.clone();
        let conn = self.conn.clone();
        let name = self.name.clone();
        handle.spawn(async move {
            let mut g = conn.lock().await;
            let need = g.as_ref().is_none_or(|c| c.close_reason().is_some());
            if !need {
                return;
            }
            match connect_and_auth(&endpoint, addr, &server_name, &api_key).await {
                Ok(c) => {
                    *g = Some(c);
                    tracing::info!(sender = %name, "nextblock_quic warmup: connection authenticated (0x00)");
                }
                Err(e) => {
                    tracing::warn!(sender = %name, error = %e, "nextblock_quic warmup: connect/auth FAILED - key/IP likely rejected; sends will be dropped");
                }
            }
        });
    }
}

#[async_trait]
impl TxSender for NextblockQuicSender {
    fn id(&self) -> u8 {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn endpoint_url(&self) -> &str {
        &self.endpoint_display
    }
    fn protocol(&self) -> &'static str {
        "NEXTBLOCK_QUIC"
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

        let url = self.endpoint_display.clone();
        let bytes = bincode::serialize(tx).unwrap_or_default();
        if bytes.len() > MAX_TX_SIZE {
            return SendOutcome {
                send_at: Instant::now(),
                send_ack_at: None,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some(format!("tx too large: {} > {MAX_TX_SIZE}", bytes.len())),
                endpoint_url_used: Some(url),
            };
        }

        let send_at = Instant::now();
        let result = self.try_send(&bytes).await;
        let send_ack_at = Some(Instant::now());

        match result {
            Ok(()) => SendOutcome {
                send_at,
                send_ack_at,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: None,
                endpoint_url_used: Some(url),
            },
            Err(e) => {
                *self.conn.lock().await = None;
                SendOutcome {
                    send_at,
                    send_ack_at: None,
                    signature,
                    http_status: None,
                    rpc_err_code: None,
                    rpc_err_message: None,
                    provider_request_id: None,
                    error: Some(format!("quic: {e}")),
                    endpoint_url_used: Some(url),
                }
            }
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

    #[test]
    fn server_name_of_strips_port() {
        assert_eq!(server_name_of("frankfurt.nextblock.io:11100"), "frankfurt.nextblock.io");
        assert_eq!(server_name_of("frankfurt.nextblock.io"), "frankfurt.nextblock.io");
    }

    #[test]
    fn resolve_addr_parses_ip_port() {
        assert_eq!(resolve_addr("127.0.0.1:11100").unwrap().port(), 11100);
    }

    #[test]
    fn bind_addr_uses_first_outbound_ip_or_unspecified() {
        assert_eq!(bind_addr(&[]).unwrap(), "0.0.0.0:0".parse().unwrap());
        let b = bind_addr(&["10.0.0.7".into()]).unwrap();
        assert_eq!(b.ip().to_string(), "10.0.0.7");
        assert_eq!(b.port(), 0);
    }

    #[test]
    fn build_client_config_succeeds() {
        assert!(build_client_config().is_ok());
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

    #[tokio::test]
    async fn new_builds_endpoint_and_labels() {
        let s = NextblockQuicSender::new(
            6,
            "nextblock-quic-fra",
            "127.0.0.1:11100",
            vec![],
            "SECRET-API-KEY",
            0,
        )
        .unwrap();
        assert_eq!(s.protocol(), "NEXTBLOCK_QUIC");
        assert_eq!(s.id(), 6);
        assert_eq!(s.name(), "nextblock-quic-fra");
        assert_eq!(s.endpoint_url(), "127.0.0.1:11100");
        assert_eq!(s.server_name, "127.0.0.1");
        assert!(!s.endpoint_url().contains("SECRET-API-KEY"));
    }

    #[tokio::test]
    async fn send_throttles_when_last_send_recent() {
        let s = NextblockQuicSender::new(6, "nextblock-quic", "127.0.0.1:11100", vec![], "k", 10_000)
            .unwrap();
        *s.last_send_at.lock() = Some(Instant::now());
        let tx = sample_tx();
        let outcome = s.send(&tx).await;
        assert_eq!(outcome.error.as_deref(), Some("throttled_local"));
        assert!(outcome.endpoint_url_used.is_none());
    }
}
