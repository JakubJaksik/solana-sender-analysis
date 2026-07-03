//! Syncro Sender (P2P.org) `sendTransaction` sender (HTTP JSON-RPC).
//!
//! P2P.org's Solana tx-send service: server-side SWQoS multi-path routing. We
//! POST a single pre-signed tx as base64 with `skipPreflight=true`,
//! `maxRetries=0` - drop-in like the Helius/Triton senders. A tip to a P2P tip
//! account (added by the preparer as a System transfer) is MANDATORY (P2P's
//! service fee, charged only on landing). The optional API key is sent as
//! `Authorization: Bearer <key>` (private endpoint); empty key = public keyless
//! endpoint. A Jito-style local throttle (`min_send_interval_ms`) caps send rate
//! to respect the public 1 RPS/IP cap. The key is a secret: header only, never
//! in the URL or logs.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use solana_sdk::transaction::Transaction;
use std::time::{Duration, Instant};

/// Bearer auth value for the optional API key. `None` when the key is
/// empty/whitespace (public keyless endpoint). The key is a secret - used only
/// as a header, never placed in the URL or logged.
fn bearer_header(api_key: &str) -> Option<String> {
    let k = api_key.trim();
    if k.is_empty() {
        None
    } else {
        Some(format!("Bearer {k}"))
    }
}

/// True if a send at `now` falls within `interval` of the previous send
/// (`last`) and should therefore be throttled. Mirrors the Jito sender's local
/// rate-limit; respects e.g. Syncro's public 1 RPS/IP cap.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

#[derive(Serialize)]
struct SendRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'static str,
    params: (&'a str, SendOptions),
}

#[derive(Serialize)]
struct SendOptions {
    encoding: &'static str,
    #[serde(rename = "skipPreflight")]
    skip_preflight: bool,
    #[serde(rename = "maxRetries")]
    max_retries: u32,
}

/// Build the JSON-RPC `sendTransaction` request body for a pre-signed tx:
/// `base64(bincode(tx))` + `{encoding:"base64", skipPreflight:true, maxRetries:0}`.
fn build_body(tx: &Transaction) -> String {
    use base64::Engine as _;
    let serialized = bincode::serialize(tx).unwrap_or_default();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&serialized);
    serde_json::to_string(&SendRequest {
        jsonrpc: "2.0",
        id: 1,
        method: "sendTransaction",
        params: (
            &b64,
            SendOptions { encoding: "base64", skip_preflight: true, max_retries: 0 },
        ),
    })
    .unwrap_or_default()
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    result: Option<String>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[derive(Debug)]
enum ParsedReply {
    Ok { signature: Option<String> },
    RpcError { code: i32, message: String },
    NonJson { body: String },
}

fn parse_reply(body: &str) -> ParsedReply {
    match serde_json::from_str::<JsonRpcResponse>(body) {
        Ok(r) => match r.error {
            Some(err) => ParsedReply::RpcError { code: err.code, message: err.message },
            None => ParsedReply::Ok { signature: r.result },
        },
        Err(_) => ParsedReply::NonJson { body: body.to_string() },
    }
}

pub struct SyncroSender {
    id: u8,
    name: String,
    /// Base URL - NO secret (the api_key is sent as a header).
    endpoint: String,
    /// Optional API key (secret). Empty = public keyless endpoint. Sent only as
    /// the `Authorization: Bearer` header; never placed in the URL or logged.
    api_key: String,
    client: reqwest::Client,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl SyncroSender {
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

    /// Fire-and-forget connection pre-warm: a lightweight `getHealth` so the
    /// first real send reuses a warm keep-alive connection. Adds the Bearer
    /// header iff an API key is configured.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let bearer = bearer_header(&self.api_key);
        handle.spawn(async move {
            let mut req = client
                .post(&endpoint)
                .header("Content-Type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"method":"getHealth"}"#);
            if let Some(b) = bearer {
                req = req.header("Authorization", b);
            }
            let _ = req.send().await;
        });
    }
}

#[async_trait]
impl TxSender for SyncroSender {
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
        "SYNCRO"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();

        // Jito-style local throttle (respects e.g. Syncro public 1 RPS/IP).
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
        let url = self.endpoint.clone();

        let send_at = Instant::now();
        let mut req = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .body(body);
        if let Some(b) = bearer_header(&self.api_key) {
            req = req.header("Authorization", b);
        }
        let result = req.send().await;
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
                endpoint_url_used: Some(url),
            },
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                match parse_reply(&body_text) {
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
                            endpoint_url_used: Some(url),
                        }
                    }
                    ParsedReply::RpcError { code, message } => SendOutcome {
                        send_at,
                        send_ack_at,
                        signature,
                        http_status: Some(status),
                        rpc_err_code: Some(code),
                        rpc_err_message: Some(message.clone()),
                        provider_request_id: None,
                        error: Some(message),
                        endpoint_url_used: Some(url),
                    },
                    ParsedReply::NonJson { body } => SendOutcome {
                        send_at,
                        send_ack_at,
                        signature,
                        http_status: Some(status),
                        rpc_err_code: None,
                        rpc_err_message: Some(format!("non-JSONRPC body: {}", body)),
                        provider_request_id: None,
                        error: Some(format!("HTTP {} body: {}", status, body)),
                        endpoint_url_used: Some(url),
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
    fn build_body_matches_syncro_send_transaction_shape() {
        use base64::Engine as _;
        let tx = sample_tx();
        let body = build_body(&tx);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["method"], "sendTransaction");
        assert_eq!(v["params"][1]["encoding"], "base64");
        assert_eq!(v["params"][1]["skipPreflight"], true);
        assert_eq!(v["params"][1]["maxRetries"], 0);
        assert!(v["params"][1].get("preflightCommitment").is_none());
        let expected_b64 = base64::engine::general_purpose::STANDARD
            .encode(bincode::serialize(&tx).unwrap());
        assert_eq!(v["params"][0], expected_b64);
    }

    #[test]
    fn bearer_header_none_when_empty() {
        assert_eq!(bearer_header(""), None);
        assert_eq!(bearer_header("   "), None);
    }

    #[test]
    fn bearer_header_some_when_set() {
        assert_eq!(bearer_header("abc-123"), Some("Bearer abc-123".to_string()));
    }

    #[test]
    fn throttled_false_when_interval_zero() {
        let now = Instant::now();
        assert!(!throttled(Some(now), now, Duration::ZERO));
    }

    #[test]
    fn throttled_false_when_no_previous_send() {
        assert!(!throttled(None, Instant::now(), Duration::from_millis(1000)));
    }

    #[test]
    fn throttled_true_within_interval_false_beyond() {
        let now = Instant::now();
        let interval = Duration::from_millis(1000);
        // last send 100ms ago -> within 1000ms -> throttled
        assert!(throttled(Some(now - Duration::from_millis(100)), now, interval));
        // last send 1500ms ago -> beyond -> not throttled
        assert!(!throttled(Some(now - Duration::from_millis(1500)), now, interval));
    }

    #[test]
    fn parse_reply_ok_returns_signature() {
        let body = r#"{"jsonrpc":"2.0","result":"5SigabcDEF","id":1}"#;
        match parse_reply(body) {
            ParsedReply::Ok { signature } => assert_eq!(signature.as_deref(), Some("5SigabcDEF")),
            other => panic!("expected Ok, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_error_returns_code_and_message() {
        let body = r#"{"jsonrpc":"2.0","error":{"code":-32007,"message":"Insufficient tip"},"id":1}"#;
        match parse_reply(body) {
            ParsedReply::RpcError { code, message } => {
                assert_eq!(code, -32007);
                assert_eq!(message, "Insufficient tip");
            }
            other => panic!("expected RpcError, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_non_json_is_captured() {
        match parse_reply("429 Too Many Requests") {
            ParsedReply::NonJson { body } => assert_eq!(body, "429 Too Many Requests"),
            other => panic!("expected NonJson, got {:?}", other),
        }
    }

    #[test]
    fn protocol_label_is_syncro() {
        let s = SyncroSender::new(4, "syncro-fra", "http://sfls-geo-fra.l2.p2p.org/public", "", 1000);
        assert_eq!(s.protocol(), "SYNCRO");
        assert_eq!(s.id(), 4);
        assert_eq!(s.name(), "syncro-fra");
        assert_eq!(s.endpoint_url(), "http://sfls-geo-fra.l2.p2p.org/public");
    }

    #[test]
    fn endpoint_url_never_contains_api_key() {
        let s = SyncroSender::new(4, "syncro-fra", "https://sfls.l2.p2p.org", "SECRET-KEY-XYZ", 20);
        assert!(!s.endpoint_url().contains("SECRET-KEY-XYZ"));
    }
}
