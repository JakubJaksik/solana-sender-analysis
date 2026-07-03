//! 0slot.trade (ZeroSlot) `sendTransaction` sender (HTTP JSON-RPC).
//!
//! 0slot's `staked_conn` endpoint connects directly to their validator node. We
//! POST a single pre-signed tx as base64 with `skipPreflight=true`,
//! `maxRetries=0` - drop-in like the Syncro/Triton senders. A tip (>= 0.001 SOL,
//! added by the preparer as a System transfer to a 0slot tip account) is
//! MANDATORY. The API key is a URL query param (`?api-key=<key>`), NOT a header;
//! it is kept private and never logged - `endpoint_url()` returns the base host
//! without the key. Only `sendTransaction` is allowed (other RPC methods 403),
//! so the connection pre-warm uses the `/health` path, not `getHealth`. A
//! Jito-style local throttle (`min_send_interval_ms`) respects 0slot's 5 RPS.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use solana_sdk::transaction::Transaction;
use std::time::{Duration, Instant};

/// True if a send at `now` falls within `interval` of the previous send (`last`)
/// and should be throttled. Mirrors the Jito/Syncro local rate-limit; respects
/// 0slot's 5 RPS cap.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// Build the POST URL with the API key as a query parameter: `<base>?api-key=<key>`
/// (or `&api-key=` if `base` already has a query). The key is a secret - used only
/// here, never returned by `endpoint_url()`.
fn send_url(base: &str, api_key: &str) -> String {
    let sep = if base.contains('?') { '&' } else { '?' };
    format!("{base}{sep}api-key={api_key}")
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

pub struct ZeroSlotSender {
    id: u8,
    name: String,
    /// Base URL (no key), e.g. http://de1.0slot.trade. The key is appended as a
    /// `?api-key=` query param at send time (see `send_url`).
    endpoint: String,
    /// Secret API key. Sent only as the `?api-key=` query param; never logged or
    /// returned by `endpoint_url()`.
    api_key: String,
    client: reqwest::Client,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl ZeroSlotSender {
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

    /// Fire-and-forget connection pre-warm: a GET to `<endpoint>/health` (no key,
    /// not an RPC method - 0slot 403s non-`sendTransaction` methods) so the first
    /// real send reuses a warm keep-alive connection.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let client = self.client.clone();
        let health = format!("{}/health", self.endpoint);
        handle.spawn(async move {
            let _ = client.get(&health).send().await;
        });
    }
}

#[async_trait]
impl TxSender for ZeroSlotSender {
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
        "0SLOT"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();

        // Jito-style local throttle (0slot allows max 5 calls/sec).
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
        let post_url = send_url(&self.endpoint, &self.api_key); // contains the key
        let redacted = self.endpoint.clone(); // base, NO key - for SendOutcome

        let send_at = Instant::now();
        let result = self
            .client
            .post(&post_url)
            .header("Content-Type", "application/json")
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
                            endpoint_url_used: Some(redacted),
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
                        endpoint_url_used: Some(redacted),
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

    #[test]
    fn throttled_false_when_interval_zero() {
        let now = Instant::now();
        assert!(!throttled(Some(now), now, Duration::ZERO));
    }

    #[test]
    fn throttled_false_when_no_previous_send() {
        assert!(!throttled(None, Instant::now(), Duration::from_millis(200)));
    }

    #[test]
    fn throttled_true_within_interval_false_beyond() {
        let now = Instant::now();
        let interval = Duration::from_millis(200);
        assert!(throttled(Some(now - Duration::from_millis(50)), now, interval));
        assert!(!throttled(Some(now - Duration::from_millis(400)), now, interval));
    }

    #[test]
    fn send_url_appends_query_when_none_present() {
        assert_eq!(send_url("http://de1.0slot.trade", "K"), "http://de1.0slot.trade?api-key=K");
    }

    #[test]
    fn send_url_uses_ampersand_when_query_present() {
        assert_eq!(send_url("http://de1.0slot.trade?x=1", "K"), "http://de1.0slot.trade?x=1&api-key=K");
    }

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
    fn build_body_matches_zeroslot_send_transaction_shape() {
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
    fn parse_reply_ok_returns_signature() {
        let body = r#"{"jsonrpc":"2.0","result":"5SigabcDEF","id":1}"#;
        match parse_reply(body) {
            ParsedReply::Ok { signature } => assert_eq!(signature.as_deref(), Some("5SigabcDEF")),
            other => panic!("expected Ok, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_error_returns_code_and_message() {
        // 0slot rate-limit error shape.
        let body = r#"{"id":"1","jsonrpc":"2.0","error":{"code":419,"message":"Rate limit exceeaded"}}"#;
        match parse_reply(body) {
            ParsedReply::RpcError { code, message } => {
                assert_eq!(code, 419);
                assert_eq!(message, "Rate limit exceeaded");
            }
            other => panic!("expected RpcError, got {:?}", other),
        }
    }

    #[test]
    fn parse_reply_non_json_is_captured() {
        match parse_reply("upstream timeout") {
            ParsedReply::NonJson { body } => assert_eq!(body, "upstream timeout"),
            other => panic!("expected NonJson, got {:?}", other),
        }
    }

    #[test]
    fn protocol_label_is_zeroslot() {
        let s = ZeroSlotSender::new(5, "0slot-de1", "http://de1.0slot.trade", "", 200);
        assert_eq!(s.protocol(), "0SLOT");
        assert_eq!(s.id(), 5);
        assert_eq!(s.name(), "0slot-de1");
        assert_eq!(s.endpoint_url(), "http://de1.0slot.trade");
    }

    #[test]
    fn endpoint_url_never_contains_api_key() {
        let s = ZeroSlotSender::new(5, "0slot-de1", "http://de1.0slot.trade", "SECRET-KEY-XYZ", 200);
        assert!(!s.endpoint_url().contains("SECRET-KEY-XYZ"));
    }
}
