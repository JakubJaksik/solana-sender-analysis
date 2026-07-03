//! NextBlock (nextblock.io) HTTP REST sender - `POST /api/v2/submit`.
//!
//! NextBlock is purpose-built Solana tx-submission infra (server-side fan-out
//! across its validator network + Jito bundles + normal leader sends), NOT a
//! generic JSON-RPC node. The HTTP REST path:
//!   * `POST https://<region>.nextblock.io/api/v2/submit`,
//!     `Content-Type: application/json`,
//!     body `{"transaction":{"content":"<base64(bincode(tx))>"},
//!           "skipPreFlight":true,"frontRunningProtection":false}`.
//!   * Auth header **`Authorization: <api-key>`** - the RAW key, **no `Bearer`**
//!     prefix (differs from Syncro). The key is secret: header only, never in
//!     the URL or logs.
//!   * Success: `{"signature":"<sig>","uuid":"..."}`. We also take the signature
//!     locally from `tx.signatures[0]`. Error: `{code,message}`.
//!   * `frontRunningProtection:false` (anti-MEV OFF) for raw latency.
//!
//! Tip is MANDATORY - one in-tx `SystemProgram::transfer` to a NextBlock tip
//! wallet (added by the preparer via `tip_accounts_for(Nextblock)`), minimum
//! **1_000_000 lamports** (0.001 SOL, per NextBlock docs). One tip is cluster-
//! wide (same 8 wallets across all regions).
//!
//! Single-host Frankfurt (NextBlock fans out server-side; docs say pick the
//! closest region - no client multi-region fan-out). `outbound_ips` bind the
//! source IP (reqwest `local_address`, rotated per send) for rate-limit spread;
//! `min_send_interval_ms` paces distinct triggers. The faster QUIC path
//! (`:11100`, ALPN `nb-tx/1`) is a separate sender.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use solana_sdk::transaction::Transaction;
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// True if a send at `now` is within `interval` of the previous send and should
/// be throttled. Paces distinct triggers under NextBlock's per-key rate limit.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// One reqwest client per outbound source IP (bound via `local_address`), or a
/// single default client when none configured. Mirrors the Jito/Astralane grid.
fn build_clients(outbound_ips: &[String]) -> Vec<reqwest::Client> {
    fn base() -> reqwest::ClientBuilder {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(8)
            .tcp_keepalive(Duration::from_secs(30))
    }
    if outbound_ips.is_empty() {
        return vec![base().build().expect("reqwest client")];
    }
    outbound_ips
        .iter()
        .map(|s| {
            let ip = IpAddr::from_str(s).unwrap_or_else(|_| panic!("invalid outbound_ip {s:?}"));
            base()
                .local_address(Some(ip))
                .build()
                .expect("reqwest client with local_address")
        })
        .collect()
}

#[derive(Serialize)]
struct TxMessage<'a> {
    content: &'a str,
}

#[derive(Serialize)]
struct SubmitRequest<'a> {
    transaction: TxMessage<'a>,
    #[serde(rename = "skipPreFlight")]
    skip_pre_flight: bool,
    #[serde(rename = "frontRunningProtection")]
    front_running_protection: bool,
}

/// Build the `/api/v2/submit` JSON body for a pre-signed tx:
/// `{"transaction":{"content":"<base64(bincode(tx))>"},"skipPreFlight":true,
/// "frontRunningProtection":false}`. Matches the official gRPC example (minus
/// the deprecated/reserved `useStakedRPCs`/`fastBestEffort`/`tip` fields).
fn build_body(tx: &Transaction) -> String {
    let raw = bincode::serialize(tx).unwrap_or_default();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
    serde_json::to_string(&SubmitRequest {
        transaction: TxMessage { content: &b64 },
        skip_pre_flight: true,
        front_running_protection: false,
    })
    .unwrap_or_default()
}

#[derive(Deserialize)]
struct SubmitResponse {
    signature: Option<String>,
    uuid: Option<String>,
    code: Option<i32>,
    message: Option<String>,
}

/// Outcome of parsing a NextBlock reply body.
#[derive(Debug)]
enum ParsedReply {
    Ok { signature: Option<String>, uuid: Option<String> },
    ApiError { code: Option<i32>, message: String },
    NonJson { body: String },
}

fn parse_reply(body: &str) -> ParsedReply {
    match serde_json::from_str::<SubmitResponse>(body) {
        Ok(r) => {
            if let Some(msg) = r.message {
                ParsedReply::ApiError { code: r.code, message: msg }
            } else {
                ParsedReply::Ok { signature: r.signature, uuid: r.uuid }
            }
        }
        Err(_) => ParsedReply::NonJson { body: body.to_string() },
    }
}

pub struct NextblockSender {
    id: u8,
    name: String,
    /// Full `/api/v2/submit` URL - no secret (auth is the `Authorization` header).
    endpoint: String,
    /// Raw API key (secret) - sent only as the `Authorization` header value.
    api_key: String,
    clients: Vec<reqwest::Client>,
    ip_cursor: AtomicUsize,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl NextblockSender {
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint: impl Into<String>,
        outbound_ips: Vec<String>,
        api_key: impl Into<String>,
        min_send_interval_ms: u64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            clients: build_clients(&outbound_ips),
            ip_cursor: AtomicUsize::new(0),
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        }
    }

    fn next_client(&self) -> &reqwest::Client {
        let idx = self.ip_cursor.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        &self.clients[idx]
    }

    /// Fire-and-forget pre-warm: a cheap GET to the host so the first real send
    /// reuses a warm keep-alive connection. `reqwest::Client` clones share the
    /// pool; we warm every bound client.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        for client in &self.clients {
            let client = client.clone();
            let endpoint = self.endpoint.clone();
            let api_key = self.api_key.clone();
            handle.spawn(async move {
                let _ = client
                    .get(&endpoint)
                    .header("Authorization", api_key)
                    .send()
                    .await;
            });
        }
    }
}

#[async_trait]
impl TxSender for NextblockSender {
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
        "NEXTBLOCK"
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

        let body = build_body(tx);
        let url = self.endpoint.clone();

        let send_at = Instant::now();
        let result = self
            .next_client()
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Authorization", &self.api_key)
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
                error: Some(format!("network: {e}")),
                endpoint_url_used: Some(url),
            },
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                match parse_reply(&body_text) {
                    ParsedReply::Ok { signature: returned, uuid } => {
                        let returned_sig = returned.as_deref().and_then(|s| s.parse().ok());
                        SendOutcome {
                            send_at,
                            send_ack_at,
                            signature: returned_sig.unwrap_or(signature),
                            http_status: Some(status),
                            rpc_err_code: None,
                            rpc_err_message: None,
                            provider_request_id: uuid,
                            error: None,
                            endpoint_url_used: Some(url),
                        }
                    }
                    ParsedReply::ApiError { code, message } => SendOutcome {
                        send_at,
                        send_ack_at,
                        signature,
                        http_status: Some(status),
                        rpc_err_code: code,
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
                        rpc_err_message: Some(format!("non-JSON body: {body}")),
                        provider_request_id: None,
                        error: Some(format!("HTTP {status} body: {body}")),
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
    fn build_body_matches_nextblock_submit_shape() {
        let tx = sample_tx();
        let body = build_body(&tx);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["skipPreFlight"], true);
        assert_eq!(v["frontRunningProtection"], false);
        // transaction.content is base64(bincode(tx))
        let expected = base64::engine::general_purpose::STANDARD
            .encode(bincode::serialize(&tx).unwrap());
        assert_eq!(v["transaction"]["content"], expected);
        // no deprecated/reserved fields
        assert!(v.get("tip").is_none());
        assert!(v.get("useStakedRPCs").is_none());
        assert!(v.get("fastBestEffort").is_none());
    }

    #[test]
    fn parse_reply_ok_returns_signature_and_uuid() {
        let body = r#"{"signature":"5SigABC","uuid":"u-123"}"#;
        match parse_reply(body) {
            ParsedReply::Ok { signature, uuid } => {
                assert_eq!(signature.as_deref(), Some("5SigABC"));
                assert_eq!(uuid.as_deref(), Some("u-123"));
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn parse_reply_error_returns_message() {
        let body = r#"{"code":3,"message":"fee too low; transaction contains low tip"}"#;
        match parse_reply(body) {
            ParsedReply::ApiError { code, message } => {
                assert_eq!(code, Some(3));
                assert!(message.contains("low tip"));
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[test]
    fn parse_reply_non_json_is_captured() {
        match parse_reply("429 Too Many Requests") {
            ParsedReply::NonJson { body } => assert_eq!(body, "429 Too Many Requests"),
            other => panic!("expected NonJson, got {other:?}"),
        }
    }

    #[test]
    fn empty_outbound_ips_yields_one_client() {
        let s = NextblockSender::new(
            5,
            "nextblock-fra",
            "https://frankfurt.nextblock.io/api/v2/submit",
            vec![],
            "k",
            0,
        );
        assert_eq!(s.clients.len(), 1);
    }

    #[test]
    fn outbound_ips_build_one_client_each_and_rotate() {
        let s = NextblockSender::new(
            5,
            "nextblock-fra",
            "https://frankfurt.nextblock.io/api/v2/submit",
            vec!["127.0.0.1".into(), "127.0.0.2".into()],
            "k",
            0,
        );
        assert_eq!(s.clients.len(), 2);
        let a = s.ip_cursor.fetch_add(1, Ordering::Relaxed) % s.clients.len();
        let b = s.ip_cursor.fetch_add(1, Ordering::Relaxed) % s.clients.len();
        let c = s.ip_cursor.fetch_add(1, Ordering::Relaxed) % s.clients.len();
        assert_eq!((a, b, c), (0, 1, 0));
    }

    #[test]
    fn endpoint_url_never_contains_api_key() {
        let s = NextblockSender::new(
            5,
            "nextblock-fra",
            "https://frankfurt.nextblock.io/api/v2/submit",
            vec![],
            "SECRET-KEY-XYZ",
            0,
        );
        assert!(!s.endpoint_url().contains("SECRET-KEY-XYZ"));
    }

    #[test]
    fn protocol_label_is_nextblock() {
        let s = NextblockSender::new(5, "nextblock-fra", "https://x/api/v2/submit", vec![], "k", 0);
        assert_eq!(s.protocol(), "NEXTBLOCK");
        assert_eq!(s.id(), 5);
        assert_eq!(s.name(), "nextblock-fra");
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
}
