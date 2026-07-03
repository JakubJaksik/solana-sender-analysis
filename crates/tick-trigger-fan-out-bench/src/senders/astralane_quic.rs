//! Astralane **QUIC** sender - the fastest Astralane path (raw QUIC over quinn,
//! NOT HTTP/3). Mirrors the official `astralane-quic-client`.
//!
//! Per tx: open a **unidirectional** stream on the persistent connection, write
//! the raw `bincode(tx)` bytes (<= 1232), `finish()`. Fire-and-forget - there is
//! no reply; the signature is taken locally from `tx.signatures[0]`.
//!
//! Handshake (exactly as the official client):
//!   * ALPN = `astralane-tpu`.
//!   * Client auth = a freshly generated **self-signed EC P-256** TLS client
//!     cert whose **Common Name is the api_key** (the server authenticates the
//!     client by this CN). The api_key is secret - it lives only in the cert CN,
//!     never in logs.
//!   * Server cert verification is skipped (server uses a self-signed cert).
//!   * Connect with server name `"astralane"`; ports 7000 (standard) / 9000
//!     (MEV-protect - we use 7000 for pure latency). Idle 30s, keep-alive 25s.
//!
//! Tip is MANDATORY (one in-tx `SystemProgram::transfer` to an `astra*` account,
//! added by the preparer via `tip_accounts_for(AstralaneQuic)`) - the same as
//! the HTTP Astralane sender. NOTE QUIC silently DROPS on rate-limit (no error),
//! so a local `min_send_interval_ms` throttle paces distinct triggers.
//!
//! IP whitelist: bind the QUIC endpoint's UDP socket to a whitelisted egress IP
//! via the first `outbound_ips` entry (QUIC is one persistent connection, so we
//! pin ONE source IP rather than rotating; run multiple sender entries for more).

use super::{SendOutcome, TxSender};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rcgen::{CertificateParams, DnType, DnValue, KeyPair};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use solana_sdk::transaction::Transaction;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// ALPN protocol id for Astralane TPU.
const ALPN_ASTRALANE_TPU: &[u8] = b"astralane-tpu";
/// quinn `connect` server name (cert verification is skipped, so this is only SNI).
const ASTRALANE_SERVER_NAME: &str = "astralane";
/// Max Solana tx size accepted by the QUIC endpoint.
const MAX_TX_SIZE: usize = 1232;

/// True if a send at `now` is within `interval` of the previous send and should
/// be throttled. QUIC silently drops rate-limited tx, so pacing distinct
/// triggers locally avoids wasted (silently-dropped) sends.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// rustls danger verifier mirroring the official Astralane QUIC client: accept
/// any server certificate (the relay uses a self-signed cert; auth is the
/// client-cert CN, not the server cert).
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

/// Build the quinn client config: self-signed EC P-256 client cert with
/// CN=api_key, ALPN `astralane-tpu`, skip-server-verify, idle 30s / keep-alive
/// 25s. Uses an explicit ring provider (no process-default install) so it
/// coexists with the AllenHark QUIC sender.
fn build_client_config(api_key: &str) -> Result<quinn::ClientConfig> {
    let key_pair = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
    let mut params = CertificateParams::new(vec![])?;
    params
        .distinguished_name
        .push(DnType::CommonName, DnValue::Utf8String(api_key.to_string()));
    let cert = params.self_signed(&key_pair)?;
    let cert_der = CertificateDer::from(cert.der().to_vec());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der()));

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_client_auth_cert(vec![cert_der], key_der)
        .context("set astralane client auth cert")?;
    crypto.alpn_protocols = vec![ALPN_ASTRALANE_TPU.to_vec()];

    let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))?;
    let mut client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(quinn::IdleTimeout::try_from(Duration::from_secs(30))?));
    transport.keep_alive_interval(Some(Duration::from_secs(25)));
    client_config.transport_config(Arc::new(transport));
    Ok(client_config)
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
        .ok_or_else(|| anyhow::anyhow!("cannot resolve astralane endpoint: {endpoint}"))
}

/// Bind address for the local UDP socket: the first whitelisted `outbound_ips`
/// entry (port 0), or the unspecified address when none configured.
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

pub struct AstralaneQuicSender {
    id: u8,
    name: String,
    addr: SocketAddr,
    /// `host:port` (no key) - returned by `endpoint_url()`.
    endpoint_display: String,
    endpoint: quinn::Endpoint,
    conn: Arc<tokio::sync::Mutex<Option<quinn::Connection>>>,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl AstralaneQuicSender {
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
        let bind = bind_addr(&outbound_ips)?;
        let client_config = build_client_config(&api_key)?;
        let mut endpoint = quinn::Endpoint::client(bind)?;
        endpoint.set_default_client_config(client_config);
        Ok(Self {
            id,
            name: name.into(),
            addr,
            endpoint_display,
            endpoint,
            conn: Arc::new(tokio::sync::Mutex::new(None)),
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        })
    }

    /// Return a live QUIC connection, (re)connecting if absent or closed.
    async fn connection(&self) -> Result<quinn::Connection> {
        let mut guard = self.conn.lock().await;
        if let Some(c) = guard.as_ref() {
            if c.close_reason().is_none() {
                return Ok(c.clone());
            }
        }
        let c = self
            .endpoint
            .connect(self.addr, ASTRALANE_SERVER_NAME)?
            .await?;
        *guard = Some(c.clone());
        Ok(c)
    }

    /// Open a uni stream, write the raw bytes, finish. Fire-and-forget (no read).
    async fn try_send(&self, bytes: &[u8]) -> Result<()> {
        let conn = self.connection().await?;
        let mut send = conn.open_uni().await?;
        send.write_all(bytes).await?;
        send.finish()?;
        Ok(())
    }

    /// Fire-and-forget pre-warm: establish the persistent connection at startup
    /// so the first real send doesn't pay the QUIC handshake on the hot path.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let endpoint = self.endpoint.clone();
        let addr = self.addr;
        let conn = self.conn.clone();
        handle.spawn(async move {
            let mut g = conn.lock().await;
            let need = g.as_ref().is_none_or(|c| c.close_reason().is_some());
            if need {
                if let Ok(connecting) = endpoint.connect(addr, ASTRALANE_SERVER_NAME) {
                    if let Ok(c) = connecting.await {
                        *g = Some(c);
                    }
                }
            }
        });
    }
}

#[async_trait]
impl TxSender for AstralaneQuicSender {
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
        "ASTRALANE_QUIC"
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
                // Drop the connection so the next send reconnects.
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
    fn resolve_addr_parses_ip_port() {
        let a = resolve_addr("185.191.117.97:7000").unwrap();
        assert_eq!(a.port(), 7000);
    }

    #[test]
    fn bind_addr_uses_first_outbound_ip_or_unspecified() {
        assert_eq!(bind_addr(&[]).unwrap(), "0.0.0.0:0".parse().unwrap());
        let b = bind_addr(&["10.0.0.7".into(), "10.0.0.8".into()]).unwrap();
        assert_eq!(b.ip().to_string(), "10.0.0.7");
        assert_eq!(b.port(), 0);
    }

    #[test]
    fn build_client_config_succeeds_with_api_key_cn() {
        // Cert generation + rustls/quinn client config build (no network).
        assert!(build_client_config("api-key-uuid-123").is_ok());
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

    // `#[tokio::test]`: `quinn::Endpoint::client` binds a UDP socket and spawns
    // the endpoint driver via the ambient Tokio runtime.
    #[tokio::test]
    async fn new_builds_endpoint_and_labels() {
        let s = AstralaneQuicSender::new(
            4,
            "astralane-quic-fra",
            "127.0.0.1:7000",
            vec![],
            "SECRET-API-KEY",
            0,
        )
        .unwrap();
        assert_eq!(s.protocol(), "ASTRALANE_QUIC");
        assert_eq!(s.id(), 4);
        assert_eq!(s.name(), "astralane-quic-fra");
        assert_eq!(s.endpoint_url(), "127.0.0.1:7000");
        // api_key lives only in the cert CN, never in the display string.
        assert!(!s.endpoint_url().contains("SECRET-API-KEY"));
    }

    #[tokio::test]
    async fn send_throttles_when_last_send_recent() {
        // The throttle guard runs BEFORE any connection attempt, so a recent
        // `last_send_at` short-circuits to `throttled_local` with zero network
        // I/O. (Driving it via a real first send would hang here: QUIC connect
        // to a dead UDP port has no fast RST and would wait for the handshake
        // timeout - unlike a TCP `connection refused`.)
        let s = AstralaneQuicSender::new(
            4,
            "astralane-quic",
            "127.0.0.1:7000",
            vec![],
            "k",
            10_000,
        )
        .unwrap();
        *s.last_send_at.lock() = Some(Instant::now());
        let tx = sample_tx();
        let outcome = s.send(&tx).await;
        assert_eq!(outcome.error.as_deref(), Some("throttled_local"));
        assert!(outcome.endpoint_url_used.is_none());
    }
}
