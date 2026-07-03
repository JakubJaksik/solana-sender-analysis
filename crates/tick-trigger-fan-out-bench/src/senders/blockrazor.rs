//! BlockRazor "Fast" sender (HTTP v2).
//!
//! BlockRazor's fastest documented Solana path is plain HTTP v2:
//! `POST <base>/v2/sendTransaction?auth=<token>&mode=fast` with
//! `Content-Type: text/plain` and the BARE base64 tx string as the body
//! (NOT JSON-RPC). A mandatory in-tx tip (min 100_000 lamports = 0.0001 SOL, to a
//! BlockRazor tip account) is added by the preparer. `mode=fast`
//! is hardcoded (sandwichMitigation mode breaks durable nonce). The auth token
//! is a URL query param (`?auth=`), kept private (never in endpoint_url/logs).
//! A Jito-style throttle respects BlockRazor's default 3 TPS.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use solana_sdk::transaction::Transaction;
use std::time::{Duration, Instant};

/// True if a send at `now` falls within `interval` of the previous send (`last`)
/// and should be throttled. Respects BlockRazor's default 3 TPS.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// BlockRazor v2 send URL: `<base>/v2/sendTransaction?auth=<key>&mode=fast`.
/// `base` is the host root (no path). The key is a secret - used only here.
fn send_url(base: &str, api_key: &str) -> String {
    format!("{base}/v2/sendTransaction?auth={api_key}&mode=fast")
}

/// BlockRazor health URL (connection pre-warm): `<base>/health`.
fn health_url(base: &str) -> String {
    format!("{base}/health")
}

/// BlockRazor v2 body = the BARE base64(bincode(tx)) string (sent with
/// `Content-Type: text/plain`; NOT JSON-RPC, NOT a JSON object).
fn build_body(tx: &Transaction) -> String {
    use base64::Engine as _;
    let serialized = bincode::serialize(tx).unwrap_or_default();
    base64::engine::general_purpose::STANDARD.encode(&serialized)
}

#[derive(Debug)]
enum ParsedReply {
    Ok { signature: Option<String> },
    Error { message: String },
}

/// Parse a BlockRazor v2 reply. Errors arrive as JSON (`{"error"|"message":...}`);
/// a success may be JSON (`{"signature"|"result":...}`) or the bare signature
/// string. A non-empty 2xx body with no recognizable JSON is treated as the
/// signature; anything else is an error. (v2 HTTP response shape is not fully
/// documented - robust to both forms.)
fn parse_reply(status: u16, body: &str) -> ParsedReply {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(err) = v
            .get("error")
            .and_then(|e| e.as_str())
            .or_else(|| v.get("message").and_then(|m| m.as_str()))
        {
            return ParsedReply::Error { message: err.to_string() };
        }
        if let Some(sig) = v
            .get("signature")
            .and_then(|s| s.as_str())
            .or_else(|| v.get("result").and_then(|s| s.as_str()))
        {
            return ParsedReply::Ok { signature: Some(sig.to_string()) };
        }
    }
    let trimmed = body.trim();
    if status < 400 && !trimmed.is_empty() {
        ParsedReply::Ok { signature: Some(trimmed.to_string()) }
    } else {
        ParsedReply::Error { message: format!("HTTP {status}: {trimmed}") }
    }
}

pub struct BlockRazorSender {
    id: u8,
    name: String,
    /// BASE host root (no path), e.g. http://frankfurt.solana.blockrazor.xyz:443.
    /// The send URL and health URL are derived from it.
    endpoint: String,
    /// Secret auth token, sent as the `?auth=` query param; never logged.
    api_key: String,
    client: reqwest::Client,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl BlockRazorSender {
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        min_send_interval_ms: u64,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(8)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self {
            id,
            name: name.into(),
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            client,
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        }
    }

    /// Fire-and-forget pre-warm: GET `<base>/health` to warm the keep-alive
    /// connection so the first real send doesn't pay TCP+TLS handshake.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let client = self.client.clone();
        let url = health_url(&self.endpoint);
        handle.spawn(async move {
            let _ = client.get(&url).send().await;
        });
    }
}

#[async_trait]
impl TxSender for BlockRazorSender {
    fn id(&self) -> u8 {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn endpoint_url(&self) -> &str {
        &self.endpoint
    }
    fn protocol(&self) -> &'static str {
        "BLOCKRAZOR"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();

        // Jito-style local throttle (BlockRazor default 3 TPS).
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

        let body = build_body(tx);
        let url = send_url(&self.endpoint, &self.api_key);
        let redacted = self.endpoint.clone(); // base, NO token - for SendOutcome

        let send_at = Instant::now();
        let result = self
            .client
            .post(&url)
            .header("Content-Type", "text/plain")
            .body(body)
            .send()
            .await;
        let send_ack_at = Some(Instant::now());

        match result {
            Err(e) => SendOutcome {
                send_at,
                send_ack_at: None,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some(format!("network: {}", e)),
                endpoint_url_used: Some(redacted),
            },
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                match parse_reply(status, &body_text) {
                    ParsedReply::Ok { signature: returned } => {
                        let returned_sig = returned.as_deref().and_then(|s| s.parse().ok());
                        SendOutcome {
                            send_at,
                            send_ack_at,
                            signature: returned_sig.unwrap_or(signature),
                            http_status: Some(status),
                            rpc_err_code: None,
                            rpc_err_message: None,
                            provider_request_id: None,
                            error: None,
                            endpoint_url_used: Some(redacted),
                        }
                    }
                    ParsedReply::Error { message } => SendOutcome {
                        send_at,
                        send_ack_at,
                        signature,
                        http_status: Some(status),
                        rpc_err_code: None,
                        rpc_err_message: Some(message.clone()),
                        provider_request_id: None,
                        error: Some(message),
                        endpoint_url_used: Some(redacted),
                    },
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
    fn build_body_is_bare_base64_not_json() {
        use base64::Engine as _;
        let tx = sample_tx();
        let body = build_body(&tx);
        let expected = base64::engine::general_purpose::STANDARD
            .encode(bincode::serialize(&tx).unwrap());
        assert_eq!(body, expected);
        assert!(!body.starts_with('{'), "must be a bare base64 string, not JSON");
    }

    #[test]
    fn parse_reply_json_error() {
        match parse_reply(200, r#"{"error":"Authentication information is missing"}"#) {
            ParsedReply::Error { message } => assert!(message.contains("Authentication")),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_json_signature() {
        match parse_reply(200, r#"{"signature":"5Sig"}"#) {
            ParsedReply::Ok { signature } => assert_eq!(signature.as_deref(), Some("5Sig")),
            other => panic!("expected Ok, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_bare_signature_on_2xx() {
        match parse_reply(200, "5BareSigString\n") {
            ParsedReply::Ok { signature } => assert_eq!(signature.as_deref(), Some("5BareSigString")),
            other => panic!("expected Ok, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_error_on_4xx() {
        match parse_reply(429, "rate limit exceeded") {
            ParsedReply::Error { message } => assert!(message.contains("429")),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn throttled_false_when_interval_zero() {
        let now = Instant::now();
        assert!(!throttled(Some(now), now, Duration::ZERO));
    }

    #[test]
    fn throttled_false_when_no_previous_send() {
        assert!(!throttled(None, Instant::now(), Duration::from_millis(350)));
    }

    #[test]
    fn throttled_true_within_interval_false_beyond() {
        let now = Instant::now();
        let interval = Duration::from_millis(350);
        assert!(throttled(Some(now - Duration::from_millis(100)), now, interval));
        assert!(!throttled(Some(now - Duration::from_millis(500)), now, interval));
    }

    #[test]
    fn send_url_builds_v2_path_with_auth_and_mode() {
        assert_eq!(
            send_url("http://frankfurt.solana.blockrazor.xyz:443", "TOK"),
            "http://frankfurt.solana.blockrazor.xyz:443/v2/sendTransaction?auth=TOK&mode=fast"
        );
    }

    #[test]
    fn health_url_appends_health() {
        assert_eq!(
            health_url("http://frankfurt.solana.blockrazor.xyz:443"),
            "http://frankfurt.solana.blockrazor.xyz:443/health"
        );
    }

    #[test]
    fn protocol_label_is_blockrazor() {
        let s = BlockRazorSender::new(7, "blockrazor-fra", "http://frankfurt.solana.blockrazor.xyz:443", "", 350);
        assert_eq!(s.protocol(), "BLOCKRAZOR");
        assert_eq!(s.id(), 7);
        assert_eq!(s.name(), "blockrazor-fra");
        assert_eq!(s.endpoint_url(), "http://frankfurt.solana.blockrazor.xyz:443");
    }

    #[test]
    fn endpoint_url_never_contains_api_key() {
        let s = BlockRazorSender::new(7, "blockrazor-fra", "http://frankfurt.solana.blockrazor.xyz:443", "SECRET-TOKEN-XYZ", 350);
        assert!(!s.endpoint_url().contains("SECRET-TOKEN-XYZ"));
    }
}
