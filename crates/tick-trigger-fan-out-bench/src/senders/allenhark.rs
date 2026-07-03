//! AllenHark Relay QUIC sender (quinn 0.11 + rustls 0.23).
//!
//! AllenHark Relay accepts a pre-signed tx over a QUIC bidirectional stream
//! (custom framing - NOT JSON-RPC). Per tx: write an optional `api-key: <key>\n`
//! preamble (or `\n` when anonymous), then `{"tx":"<base64>","simulate":false}`,
//! finish the send side, and read `{"status":"accepted","signature":...}`.
//! `simulate` MUST be false. We send ANONYMOUSLY (empty key). A mandatory tip
//! (>=0.001 SOL to a `hark…` wallet) is added by the preparer as a System
//! transfer. TLS skips server-cert verification and sets no ALPN, per AllenHark
//! docs. The connection is persistent (lazily reconnected); a fresh bi stream
//! per tx.

use super::{SendOutcome, TxSender};
use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use solana_sdk::transaction::Transaction;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

#[derive(Serialize)]
struct SubmitBody<'a> {
    tx: &'a str,
    simulate: bool,
}

/// `{"tx":"<base64(bincode(tx))>","simulate":false}`. `simulate` MUST be false
/// (the relay rejects `simulate:true` with `simulate_forbidden`).
fn build_body(tx: &Transaction) -> String {
    use base64::Engine as _;
    let serialized = bincode::serialize(tx).unwrap_or_default();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&serialized);
    serde_json::to_string(&SubmitBody { tx: &b64, simulate: false }).unwrap_or_default()
}

/// Bytes written to the QUIC stream: an optional `api-key: <key>\n` preamble
/// (or just `\n` when anonymous), followed by the JSON body.
fn frame(api_key: &str, body: &str) -> Vec<u8> {
    let k = api_key.trim();
    let preamble = if k.is_empty() {
        "\n".to_string()
    } else {
        format!("api-key: {k}\n")
    };
    let mut buf = Vec::with_capacity(preamble.len() + body.len());
    buf.extend_from_slice(preamble.as_bytes());
    buf.extend_from_slice(body.as_bytes());
    buf
}

#[derive(Deserialize)]
struct RelayResponse {
    status: Option<String>,
    signature: Option<String>,
    error: Option<String>,
    message: Option<String>,
}

#[derive(Debug)]
enum ParsedReply {
    Accepted { signature: Option<String> },
    RelayError { message: String },
    NonJson { body: String },
}

fn parse_reply(bytes: &[u8]) -> ParsedReply {
    let text = String::from_utf8_lossy(bytes);
    match serde_json::from_str::<RelayResponse>(&text) {
        Ok(r) => {
            if let Some(err) = r.error.or(r.message) {
                ParsedReply::RelayError { message: err }
            } else if r.status.as_deref() == Some("accepted") {
                ParsedReply::Accepted { signature: r.signature }
            } else {
                ParsedReply::RelayError { message: text.to_string() }
            }
        }
        Err(_) => ParsedReply::NonJson { body: text.to_string() },
    }
}

/// rustls 0.23 danger verifier: accepts any server certificate. AllenHark docs
/// instruct skipping server-cert verification (relay uses Let's Encrypt; TLS is
/// for encryption, not auth - auth is the optional in-stream api-key).
#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

/// Build a quinn client endpoint that skips server-cert verification and sets NO
/// ALPN (per AllenHark docs), bound to an ephemeral local UDP socket. Uses an
/// explicit ring provider (`builder_with_provider`) so it does not depend on a
/// process-default CryptoProvider being installed (important for unit tests).
fn build_quic_endpoint() -> anyhow::Result<quinn::Endpoint> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    crypto.alpn_protocols.clear(); // no ALPN

    let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))?;
    let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

    let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse()?)?;
    endpoint.set_default_client_config(client_config);
    Ok(endpoint)
}

/// TLS SNI for AllenHark Relay. Fixed; the per-region IP goes in `endpoint_url`.
const ALLENHARK_SERVER_NAME: &str = "relay.allenhark.com";

pub struct AllenHarkSender {
    id: u8,
    name: String,
    addr: SocketAddr,
    /// host:port string (no secret) - returned by `endpoint_url()`.
    endpoint_display: String,
    /// Optional API key. Empty = anonymous. Sent only in the stream preamble.
    api_key: String,
    endpoint: quinn::Endpoint,
    conn: Arc<tokio::sync::Mutex<Option<quinn::Connection>>>,
}

impl AllenHarkSender {
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let endpoint_display = endpoint_url.into();
        let addr: SocketAddr = endpoint_display
            .parse()
            .with_context(|| format!("parse allenhark endpoint_url as ip:port: {endpoint_display}"))?;
        let endpoint = build_quic_endpoint()?;
        Ok(Self {
            id,
            name: name.into(),
            addr,
            endpoint_display,
            api_key: api_key.into(),
            endpoint,
            conn: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    /// Return a live QUIC connection, (re)connecting if absent or closed.
    async fn connection(&self) -> anyhow::Result<quinn::Connection> {
        let mut guard = self.conn.lock().await;
        if let Some(c) = guard.as_ref() {
            if c.close_reason().is_none() {
                return Ok(c.clone());
            }
        }
        let c = self
            .endpoint
            .connect(self.addr, ALLENHARK_SERVER_NAME)?
            .await?;
        *guard = Some(c.clone());
        Ok(c)
    }

    /// Open a stream, write the framed request, read the reply bytes.
    async fn try_send(&self, frame_bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
        let conn = self.connection().await?;
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(frame_bytes).await?;
        send.finish()?;
        let resp = recv.read_to_end(64 * 1024).await?;
        Ok(resp)
    }

    /// Fire-and-forget pre-warm: establish the persistent connection at startup.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let endpoint = self.endpoint.clone();
        let addr = self.addr;
        let conn = self.conn.clone();
        handle.spawn(async move {
            let mut g = conn.lock().await;
            let need = g.as_ref().map_or(true, |c| c.close_reason().is_some());
            if need {
                if let Ok(connecting) = endpoint.connect(addr, ALLENHARK_SERVER_NAME) {
                    if let Ok(c) = connecting.await {
                        *g = Some(c);
                    }
                }
            }
        });
    }
}

#[async_trait]
impl TxSender for AllenHarkSender {
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
        "ALLENHARK"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();
        let frame_bytes = frame(&self.api_key, &build_body(tx));
        let url = self.endpoint_display.clone();

        let send_at = Instant::now();
        let result = self.try_send(&frame_bytes).await;
        let send_ack_at = Some(Instant::now());

        match result {
            Ok(bytes) => match parse_reply(&bytes) {
                ParsedReply::Accepted { signature: returned } => {
                    let returned_sig = returned.as_deref().and_then(|s| s.parse().ok());
                    SendOutcome {
                        send_at,
                        send_ack_at,
                        signature: returned_sig.unwrap_or(signature),
                        http_status: None,
                        rpc_err_code: None,
                        rpc_err_message: None,
                        provider_request_id: None,
                        error: None,
                        endpoint_url_used: Some(url),
                    }
                }
                ParsedReply::RelayError { message } => SendOutcome {
                    send_at,
                    send_ack_at,
                    signature,
                    http_status: None,
                    rpc_err_code: None,
                    rpc_err_message: Some(message.clone()),
                    provider_request_id: None,
                    error: Some(message),
                    endpoint_url_used: Some(url),
                },
                ParsedReply::NonJson { body } => SendOutcome {
                    send_at,
                    send_ack_at,
                    signature,
                    http_status: None,
                    rpc_err_code: None,
                    rpc_err_message: Some(format!("non-JSON relay reply: {body}")),
                    provider_request_id: None,
                    error: Some(format!("non-JSON relay reply: {body}")),
                    endpoint_url_used: Some(url),
                },
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
    fn build_body_has_base64_tx_and_simulate_false() {
        use base64::Engine as _;
        let tx = sample_tx();
        let body = build_body(&tx);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["simulate"], false);
        let expected = base64::engine::general_purpose::STANDARD
            .encode(bincode::serialize(&tx).unwrap());
        assert_eq!(v["tx"], expected);
    }

    #[test]
    fn frame_anonymous_starts_with_newline() {
        let f = frame("", "BODY");
        assert!(f.starts_with(b"\n"));
        assert!(f.ends_with(b"BODY"));
    }

    #[test]
    fn frame_with_key_has_api_key_preamble() {
        let f = frame("sk_live_123", "BODY");
        let s = String::from_utf8(f).unwrap();
        assert!(s.starts_with("api-key: sk_live_123\n"));
        assert!(s.ends_with("BODY"));
    }

    #[test]
    fn parse_reply_accepted_returns_signature() {
        match parse_reply(br#"{"status":"accepted","request_id":"r1","signature":"5Sig"}"#) {
            ParsedReply::Accepted { signature } => assert_eq!(signature.as_deref(), Some("5Sig")),
            other => panic!("expected Accepted, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_error_returns_message() {
        match parse_reply(br#"{"error":"tip_missing"}"#) {
            ParsedReply::RelayError { message } => assert_eq!(message, "tip_missing"),
            other => panic!("expected RelayError, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_non_json_is_captured() {
        match parse_reply(b"upstream reset") {
            ParsedReply::NonJson { body } => assert_eq!(body, "upstream reset"),
            other => panic!("expected NonJson, got {:?}", other),
        }
    }

    // `#[tokio::test]`: `AllenHarkSender::new` eagerly builds the quinn endpoint
    // (`build_quic_endpoint` -> `Endpoint::client`), which looks up the ambient
    // Tokio runtime - so construction must run inside a Tokio context.
    #[tokio::test]
    async fn protocol_label_and_endpoint_url() {
        let s = AllenHarkSender::new(6, "allenhark-fra", "84.32.223.83:4433", "").unwrap();
        assert_eq!(s.protocol(), "ALLENHARK");
        assert_eq!(s.id(), 6);
        assert_eq!(s.name(), "allenhark-fra");
        assert_eq!(s.endpoint_url(), "84.32.223.83:4433");
    }

    #[tokio::test]
    async fn endpoint_url_never_contains_api_key() {
        let s = AllenHarkSender::new(6, "allenhark-fra", "84.32.223.83:4433", "SECRET-KEY-XYZ").unwrap();
        assert!(!s.endpoint_url().contains("SECRET-KEY-XYZ"));
    }

    // `#[tokio::test]`: `build_quic_endpoint` binds the UDP socket and spawns the
    // quinn endpoint driver via the Tokio runtime, so it must run inside a Tokio
    // context (quinn's `Endpoint::client` looks up the ambient runtime).
    #[tokio::test]
    async fn quic_endpoint_builds() {
        // Builds the rustls skip-verify config + binds an ephemeral UDP socket.
        // No network I/O; just verifies the client config is constructible.
        assert!(build_quic_endpoint().is_ok());
    }
}
