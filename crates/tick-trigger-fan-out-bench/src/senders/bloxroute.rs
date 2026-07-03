//! bloXroute Solana Trader API sender - `POST /api/v2/submit`, multi-region
//! fan-out over plaintext HTTP.
//!
//! bloXroute is the ORIGIN of the `/api/v2/submit` + `frontRunningProtection` +
//! `useStakedRPCs` shape (NextBlock mirrors it). Fastest documented non-QUIC
//! path is **plaintext HTTP** - bloXroute docs say verbatim "USE HTTP, NOT
//! HTTPS, FOR THE BEST PERFORMANCE" (skip the TLS handshake). There is no
//! publicly-documented QUIC endpoint (no host/port/ALPN, mTLS-only), so this is
//! the fastest implementable path.
//!
//!   * `POST http://<region>.solana.dex.blxrbdn.com/api/v2/submit`,
//!     `Content-Type: application/json`,
//!     body `{"transaction":{"content":"<base64(bincode(tx))>"},
//!           "skipPreFlight":true,"frontRunningProtection":false,
//!           "submitProtection":"SP_LOW","useStakedRPCs":true}`.
//!   * `useStakedRPCs:true` = fastest staked-leader path; REQUIRES the in-tx tip
//!     and is mutually exclusive with `frontRunningProtection`/`revertProtection`
//!     (both kept off). `SP_LOW` = no MEV protection, direct to Jito, no delay.
//!   * Auth header **`Authorization: <api-key>`** - RAW token, **no `Bearer`**
//!     (the bloXroute portal token). Header only; never in the URL or logs.
//!   * Success: HTTP 200 `{"signature":"<sig>"}` (no uuid). We also take the
//!     signature locally from `tx.signatures[0]`. Error: 429 / JSON `{code,message}`.
//!
//! **Multi-region fan-out** (docs RECOMMEND it - server-side BDN propagation is
//! NOT a substitute): the SAME signed tx is POSTed to N regional hosts
//! concurrently, first reply wins (germany + global + amsterdam for FRA). ONE
//! cluster-wide tip (one signed tx → dedup → one lands → one tip; rotation over
//! the 17 wallets is only for write-lock contention). Tip is MANDATORY in-tx
//! (preparer via `tip_accounts_for(Bloxroute)`), min 1_000_000 lamports.
//!
//! `outbound_ips` bind source IPs (reqwest `local_address`, host×IP grid,
//! rotated per send); `min_send_interval_ms` paces distinct triggers.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use solana_sdk::transaction::Transaction;
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

fn substitute_region(template: &str, region: &str) -> String {
    template.replace("{region}", region)
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
    #[serde(rename = "submitProtection")]
    submit_protection: &'static str,
    #[serde(rename = "useStakedRPCs")]
    use_staked_rpcs: bool,
}

/// `/api/v2/submit` body: base64(bincode(tx)) in `transaction.content`, plus the
/// bloXroute fast-path flags (`useStakedRPCs:true` + `SP_LOW`, anti-MEV off).
fn build_body(tx: &Transaction) -> String {
    let raw = bincode::serialize(tx).unwrap_or_default();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
    serde_json::to_string(&SubmitRequest {
        transaction: TxMessage { content: &b64 },
        skip_pre_flight: true,
        front_running_protection: false,
        submit_protection: "SP_LOW",
        use_staked_rpcs: true,
    })
    .unwrap_or_default()
}

#[derive(Deserialize)]
struct SubmitResponse {
    signature: Option<String>,
    code: Option<i32>,
    message: Option<String>,
}

enum FanoutReply {
    Success {
        host: String,
        send_ack_at: Instant,
        http_status: u16,
        signature: Option<String>,
    },
    Error {
        host: String,
        send_ack_at: Instant,
        http_status: Option<u16>,
        rpc_err_code: Option<i32>,
        rpc_err_message: Option<String>,
        error: String,
    },
}

pub struct BloxrouteSender {
    id: u8,
    name: String,
    /// `{region}` host template incl. `/api/v2/submit` - no secret.
    endpoint_template: String,
    /// Raw API key (secret) - sent only as the `Authorization` header value.
    api_key: String,
    /// Per-region full submit URLs.
    hosts: Vec<String>,
    /// `grid[host_idx][ip_idx]` - a client per (host, source IP).
    grid: Vec<Vec<reqwest::Client>>,
    ip_count: usize,
    ip_cursor: AtomicUsize,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl BloxrouteSender {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint_template: impl Into<String>,
        regions: Vec<String>,
        outbound_ips: Vec<String>,
        api_key: impl Into<String>,
        min_send_interval_ms: u64,
    ) -> Self {
        let endpoint_template = endpoint_template.into();
        let hosts: Vec<String> = if regions.is_empty() {
            vec![endpoint_template.clone()]
        } else {
            regions
                .iter()
                .map(|r| substitute_region(&endpoint_template, r))
                .collect()
        };
        let ip_count = outbound_ips.len().max(1);
        let grid: Vec<Vec<reqwest::Client>> =
            hosts.iter().map(|_| build_clients(&outbound_ips)).collect();
        Self {
            id,
            name: name.into(),
            endpoint_template,
            api_key: api_key.into(),
            hosts,
            grid,
            ip_count,
            ip_cursor: AtomicUsize::new(0),
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        }
    }

    fn next_ip_idx(&self) -> usize {
        self.ip_cursor.fetch_add(1, Ordering::Relaxed) % self.ip_count
    }

    /// Fire-and-forget pre-warm: a cheap GET (with the Authorization header) per
    /// (host, source IP) so the first real send reuses a warm keep-alive conn.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        for host_idx in 0..self.hosts.len() {
            let url = self.hosts[host_idx].clone();
            for client in &self.grid[host_idx] {
                let client = client.clone();
                let url = url.clone();
                let api_key = self.api_key.clone();
                handle.spawn(async move {
                    let _ = client.get(&url).header("Authorization", api_key).send().await;
                });
            }
        }
    }
}

fn parse_reply(host: String, status: u16, body: String, send_ack_at: Instant) -> FanoutReply {
    if (200..300).contains(&status) {
        match serde_json::from_str::<SubmitResponse>(&body) {
            Ok(r) if r.message.is_none() => FanoutReply::Success {
                host,
                send_ack_at,
                http_status: status,
                signature: r.signature,
            },
            Ok(r) => FanoutReply::Error {
                host,
                send_ack_at,
                http_status: Some(status),
                rpc_err_code: r.code,
                rpc_err_message: r.message.clone(),
                error: r.message.unwrap_or_else(|| "error".into()),
            },
            // 2xx with a non-JSON body: treat as accepted (signature is local).
            Err(_) => FanoutReply::Success {
                host,
                send_ack_at,
                http_status: status,
                signature: None,
            },
        }
    } else {
        let (code, msg) = match serde_json::from_str::<SubmitResponse>(&body) {
            Ok(r) => (r.code, r.message.unwrap_or_else(|| body.clone())),
            Err(_) => (None, body.clone()),
        };
        FanoutReply::Error {
            host,
            send_ack_at,
            http_status: Some(status),
            rpc_err_code: code,
            rpc_err_message: Some(msg.clone()),
            error: format!("HTTP {status}: {msg}"),
        }
    }
}

#[async_trait]
impl TxSender for BloxrouteSender {
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
        "BLOXROUTE"
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
        if self.hosts.is_empty() {
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

        let body: Arc<String> = Arc::new(build_body(tx));
        let ip_idx = self.next_ip_idx();
        let (tx_first, mut rx_first) =
            tokio::sync::mpsc::channel::<FanoutReply>(self.hosts.len());

        for host_idx in 0..self.hosts.len() {
            let url = self.hosts[host_idx].clone();
            let client = self.grid[host_idx][ip_idx % self.ip_count].clone();
            let body = body.clone();
            let api_key = self.api_key.clone();
            let tx_first = tx_first.clone();
            tokio::spawn(async move {
                let result = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", api_key)
                    .body((*body).clone())
                    .send()
                    .await;
                let send_ack_at = Instant::now();
                let reply = match result {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        let text = resp.text().await.unwrap_or_default();
                        parse_reply(url, status, text, send_ack_at)
                    }
                    Err(e) => FanoutReply::Error {
                        host: url,
                        send_ack_at,
                        http_status: None,
                        rpc_err_code: None,
                        rpc_err_message: None,
                        error: format!("network: {e}"),
                    },
                };
                let _ = tx_first.send(reply).await;
            });
        }
        drop(tx_first);

        let Some(first) = rx_first.recv().await else {
            return SendOutcome {
                send_at,
                send_ack_at: None,
                signature,
                http_status: None,
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: Some("all fan-out tasks dropped without reply".into()),
                endpoint_url_used: None,
            };
        };

        match first {
            FanoutReply::Success { host, send_ack_at, http_status, signature: returned } => {
                let returned_sig = returned.as_deref().and_then(|s| s.parse().ok());
                SendOutcome {
                    send_at,
                    send_ack_at: Some(send_ack_at),
                    signature: returned_sig.unwrap_or(signature),
                    http_status: Some(http_status),
                    rpc_err_code: None,
                    rpc_err_message: None,
                    provider_request_id: None,
                    error: None,
                    endpoint_url_used: Some(host),
                }
            }
            FanoutReply::Error { host, send_ack_at, http_status, rpc_err_code, rpc_err_message, error } => {
                SendOutcome {
                    send_at,
                    send_ack_at: Some(send_ack_at),
                    signature,
                    http_status,
                    rpc_err_code,
                    rpc_err_message,
                    provider_request_id: None,
                    error: Some(error),
                    endpoint_url_used: Some(host),
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
    fn build_body_matches_bloxroute_submit_shape() {
        let tx = sample_tx();
        let body = build_body(&tx);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["skipPreFlight"], true);
        assert_eq!(v["frontRunningProtection"], false);
        assert_eq!(v["submitProtection"], "SP_LOW");
        assert_eq!(v["useStakedRPCs"], true);
        let expected = base64::engine::general_purpose::STANDARD
            .encode(bincode::serialize(&tx).unwrap());
        assert_eq!(v["transaction"]["content"], expected);
    }

    #[test]
    fn substitute_region_replaces_placeholder() {
        assert_eq!(
            substitute_region("http://{region}.solana.dex.blxrbdn.com/api/v2/submit", "germany"),
            "http://germany.solana.dex.blxrbdn.com/api/v2/submit"
        );
    }

    #[test]
    fn new_builds_host_per_region_and_grid_per_ip() {
        let s = BloxrouteSender::new(
            7,
            "bloxroute-fra",
            "http://{region}.solana.dex.blxrbdn.com/api/v2/submit",
            vec!["germany".into(), "global".into(), "amsterdam".into()],
            vec!["127.0.0.1".into(), "127.0.0.2".into()],
            "k",
            0,
        );
        assert_eq!(s.hosts.len(), 3);
        assert_eq!(s.hosts[0], "http://germany.solana.dex.blxrbdn.com/api/v2/submit");
        assert_eq!(s.ip_count, 2);
        assert_eq!(s.grid.len(), 3);
        assert_eq!(s.grid[0].len(), 2);
    }

    #[test]
    fn empty_regions_uses_template_as_single_host() {
        let s = BloxrouteSender::new(
            7,
            "bloxroute-fra",
            "http://germany.solana.dex.blxrbdn.com/api/v2/submit",
            vec![],
            vec![],
            "k",
            0,
        );
        assert_eq!(s.hosts.len(), 1);
        assert_eq!(s.ip_count, 1);
    }

    #[test]
    fn parse_reply_ok_returns_signature() {
        match parse_reply("h".into(), 200, r#"{"signature":"5SigABC"}"#.into(), Instant::now()) {
            FanoutReply::Success { signature, .. } => assert_eq!(signature.as_deref(), Some("5SigABC")),
            _ => panic!("expected Success"),
        }
    }

    #[test]
    fn parse_reply_non_2xx_is_error() {
        match parse_reply("h".into(), 429, "Too Many Requests".into(), Instant::now()) {
            FanoutReply::Error { http_status, error, .. } => {
                assert_eq!(http_status, Some(429));
                assert!(error.contains("429"));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn parse_reply_json_error_body_captures_message() {
        match parse_reply("h".into(), 400, r#"{"code":3,"message":"missing tip"}"#.into(), Instant::now()) {
            FanoutReply::Error { rpc_err_code, rpc_err_message, .. } => {
                assert_eq!(rpc_err_code, Some(3));
                assert_eq!(rpc_err_message.as_deref(), Some("missing tip"));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn endpoint_url_never_contains_api_key() {
        let s = BloxrouteSender::new(
            7,
            "bloxroute-fra",
            "http://germany.solana.dex.blxrbdn.com/api/v2/submit",
            vec![],
            vec![],
            "SECRET-TOKEN",
            0,
        );
        assert!(!s.endpoint_url().contains("SECRET-TOKEN"));
    }

    #[test]
    fn protocol_label_is_bloxroute() {
        let s = BloxrouteSender::new(7, "bloxroute-fra", "http://x/api/v2/submit", vec![], vec![], "k", 0);
        assert_eq!(s.protocol(), "BLOXROUTE");
        assert_eq!(s.id(), 7);
        assert_eq!(s.name(), "bloxroute-fra");
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
    async fn send_with_no_hosts_returns_error() {
        let mut s = BloxrouteSender::new(7, "bloxroute", "http://x/api/v2/submit", vec![], vec![], "k", 0);
        s.hosts.clear();
        let tx = sample_tx();
        let outcome = s.send(&tx).await;
        assert_eq!(outcome.error.as_deref(), Some("no regions configured"));
    }
}
