//! Astralane **Iris / Gateway** sender - raw-binary `/irisb` fast path, with
//! multi-region fan-out and per-source-IP binding.
//!
//! Astralane's lowest-overhead HTTP route is `/irisb`:
//!   * `POST {host}/irisb?api-key=<KEY>&method=sendTransaction`,
//!     `Content-Type: application/octet-stream`, body = RAW `bincode(tx)`
//!     (NO base64 - zero encoding overhead, like Triton `/sendtx`). <= 1232 B.
//!   * Fire-and-forget: on 2xx we don't read the body and take the signature
//!     locally from `tx.signatures[0]`; 4xx/5xx (e.g. 429 rate-limit,
//!     `UNKNOWN_API_KEY`) are read off the success path for diagnostics.
//!   * Auth = **`?api-key=<KEY>` URL query** (hyphen). Secret: stripped from
//!     `endpoint_url()`/`endpoint_url_used` and scrubbed from error strings.
//!
//! **Tip is MANDATORY** - one in-tx `SystemProgram::transfer` to an `astra*`
//! tip account (added by the preparer via `tip_accounts_for(Astralane)`,
//! tier-gated minimum; there is no auto-tip and `swqos-only` does NOT waive it).
//! One signed tx → one landed copy (server-side dedup) → one tip, regardless of
//! fan-out, so the tip is NOT varied per region.
//!
//! **Fan-out**: Astralane fans out server-side to ~8 upcoming leaders per
//! gateway; client-side multi-target is still recommended (edge + closest). We
//! support a `{region}` host template + `regions` list, fanning the SAME tx to
//! all hosts concurrently and racing the first reply (Nozomi-style). One tip
//! covers all targets.
//!
//! **Source-IP binding**: Astralane gates access by an account-level IP
//! allowlist, so each send must egress from a whitelisted IP. Like the Jito
//! sender we hold a grid of `reqwest` clients (one per host × per `outbound_ips`
//! entry, bound via `local_address`) and rotate the source IP per send. Leave
//! `outbound_ips` empty to use the OS-default egress.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use solana_sdk::transaction::Transaction;
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Astralane raw-binary fast-path route.
const SEND_PATH: &str = "/irisb";

/// Substitute the `{region}` placeholder in a host template.
fn substitute_region(template: &str, region: &str) -> String {
    template.replace("{region}", region)
}

/// Full `/irisb` POST URL incl. the secret key:
/// `{host}/irisb?api-key=<key>&method=sendTransaction`. PRIVATE - never logged.
fn build_send_url(host: &str, api_key: &str) -> String {
    let base = host.trim_end_matches('/');
    format!("{base}{SEND_PATH}?api-key={api_key}&method=sendTransaction")
}

/// Keep-alive warm URL: `{host}/irisb?api-key=<key>` (GET - answered non-2xx,
/// which is fine; the point is to prime the keep-alive connection). PRIVATE.
fn build_warm_url(host: &str, api_key: &str) -> String {
    let base = host.trim_end_matches('/');
    format!("{base}{SEND_PATH}?api-key={api_key}")
}

/// Strip the `?...` query (api-key + method) for safe logging.
fn redact_query(url: &str) -> String {
    match url.split_once('?') {
        Some((before, _)) => before.to_string(),
        None => url.to_string(),
    }
}

/// Scrub the secret key out of a message (reqwest error Display embeds the URL).
/// No-op when key is empty.
fn scrub(msg: String, api_key: &str) -> String {
    if api_key.is_empty() {
        msg
    } else {
        msg.replace(api_key, "***")
    }
}

/// True if a send at `now` is within `interval` of the previous send and should
/// be throttled. Paces DISTINCT triggers under the per-key TPS cap (HTTP 429 on
/// overflow). The fan-out below is one distinct tx and is not throttled.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// One `reqwest` client per outbound source IP (bound via `local_address`), or a
/// single default client when none configured. Mirrors the Jito grid builder so
/// sends can egress from the whitelisted FRA IPs. Binding is lazy (at connect),
/// so an IP not on this host fails only at send time, never at build.
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

/// One regional fan-out result; the first reply (success OR error) wins the race.
enum FanoutReply {
    Success {
        host_display: String,
        send_ack_at: Instant,
        http_status: u16,
    },
    Error {
        host_display: String,
        send_ack_at: Instant,
        http_status: Option<u16>,
        rpc_err_message: Option<String>,
        error: String,
    },
}

pub struct AstralaneSender {
    id: u8,
    name: String,
    /// Host template (no key) - returned by `endpoint_url()` for logging.
    endpoint_template: String,
    /// Secret key (query-param auth). Used to build URLs and scrub errors.
    api_key: String,
    /// Per-region full `/irisb` POST URLs incl. `?api-key=<key>` - PRIVATE.
    hosts: Vec<String>,
    /// Per-region redacted POST URLs (no key) - for `endpoint_url_used`.
    hosts_display: Vec<String>,
    /// Per-region keep-alive warm URLs incl. `?api-key=<key>` - PRIVATE.
    warm_urls: Vec<String>,
    /// `grid[host_idx][ip_idx]` - a client per (host, source IP).
    grid: Vec<Vec<reqwest::Client>>,
    ip_count: usize,
    ip_cursor: AtomicUsize,
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl AstralaneSender {
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
        let api_key = api_key.into();
        let region_hosts: Vec<String> = if regions.is_empty() {
            vec![endpoint_template.clone()]
        } else {
            regions
                .iter()
                .map(|r| substitute_region(&endpoint_template, r))
                .collect()
        };
        let hosts: Vec<String> = region_hosts.iter().map(|h| build_send_url(h, &api_key)).collect();
        let hosts_display: Vec<String> = hosts.iter().map(|h| redact_query(h)).collect();
        let warm_urls: Vec<String> = region_hosts.iter().map(|h| build_warm_url(h, &api_key)).collect();
        let ip_count = outbound_ips.len().max(1);
        let grid: Vec<Vec<reqwest::Client>> =
            hosts.iter().map(|_| build_clients(&outbound_ips)).collect();
        Self {
            id,
            name: name.into(),
            endpoint_template,
            api_key,
            hosts,
            hosts_display,
            warm_urls,
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

    /// Fire-and-forget pre-warm: a cheap GET per (host, source IP) so the first
    /// real send reuses a warm keep-alive connection on each bound client.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        for host_idx in 0..self.hosts.len() {
            let url = self.warm_urls[host_idx].clone();
            for client in &self.grid[host_idx] {
                let client = client.clone();
                let url = url.clone();
                handle.spawn(async move {
                    let _ = client.get(&url).send().await;
                });
            }
        }
    }
}

#[async_trait]
impl TxSender for AstralaneSender {
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
        "ASTRALANE"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();

        // Local throttle: pace DISTINCT triggers under the per-key TPS cap.
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

        // Raw wire bytes - bincode(tx), unencoded octet-stream.
        let body: Arc<Vec<u8>> = Arc::new(bincode::serialize(tx).unwrap_or_default());
        let ip_idx = self.next_ip_idx();
        let (tx_first, mut rx_first) =
            tokio::sync::mpsc::channel::<FanoutReply>(self.hosts.len());

        for host_idx in 0..self.hosts.len() {
            let url = self.hosts[host_idx].clone();
            let host_display = self.hosts_display[host_idx].clone();
            let client = self.grid[host_idx][ip_idx % self.ip_count].clone();
            let body = body.clone();
            let tx_first = tx_first.clone();
            let api_key = self.api_key.clone();
            tokio::spawn(async move {
                let result = client
                    .post(&url)
                    .header("Content-Type", "application/octet-stream")
                    .body((*body).clone())
                    .send()
                    .await;
                let send_ack_at = Instant::now();
                let reply = match result {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        if (200..300).contains(&status) {
                            FanoutReply::Success { host_display, send_ack_at, http_status: status }
                        } else {
                            let text = scrub(resp.text().await.unwrap_or_default(), &api_key);
                            FanoutReply::Error {
                                host_display,
                                send_ack_at,
                                http_status: Some(status),
                                rpc_err_message: Some(text.clone()),
                                error: format!("HTTP {status}: {text}"),
                            }
                        }
                    }
                    Err(e) => FanoutReply::Error {
                        host_display,
                        send_ack_at,
                        http_status: None,
                        rpc_err_message: None,
                        error: scrub(format!("network: {e}"), &api_key),
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
            FanoutReply::Success { host_display, send_ack_at, http_status } => SendOutcome {
                send_at,
                send_ack_at: Some(send_ack_at),
                signature,
                http_status: Some(http_status),
                rpc_err_code: None,
                rpc_err_message: None,
                provider_request_id: None,
                error: None,
                endpoint_url_used: Some(host_display),
            },
            FanoutReply::Error { host_display, send_ack_at, http_status, rpc_err_message, error } => {
                SendOutcome {
                    send_at,
                    send_ack_at: Some(send_ack_at),
                    signature,
                    http_status,
                    rpc_err_code: None,
                    rpc_err_message,
                    provider_request_id: None,
                    error: Some(error),
                    endpoint_url_used: Some(host_display),
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
    fn substitute_region_replaces_placeholder() {
        assert_eq!(
            substitute_region("https://{region}.gateway.astralane.io", "fr"),
            "https://fr.gateway.astralane.io"
        );
    }

    #[test]
    fn build_send_url_appends_irisb_route_key_and_method() {
        assert_eq!(
            build_send_url("https://fr.gateway.astralane.io", "KEY-123"),
            "https://fr.gateway.astralane.io/irisb?api-key=KEY-123&method=sendTransaction"
        );
        // trailing slash trimmed
        assert_eq!(
            build_send_url("https://fr.gateway.astralane.io/", "KEY-123"),
            "https://fr.gateway.astralane.io/irisb?api-key=KEY-123&method=sendTransaction"
        );
    }

    #[test]
    fn redact_query_strips_key_and_method() {
        let red = redact_query(
            "https://fr.gateway.astralane.io/irisb?api-key=SECRET-KEY&method=sendTransaction",
        );
        assert_eq!(red, "https://fr.gateway.astralane.io/irisb");
        assert!(!red.contains("SECRET-KEY"));
    }

    #[test]
    fn scrub_removes_key_from_error_text() {
        let msg = "network: error for url (https://x/irisb?api-key=SECRET-KEY)".to_string();
        let safe = scrub(msg, "SECRET-KEY");
        assert!(!safe.contains("SECRET-KEY"));
        assert!(safe.contains("***"));
        assert_eq!(scrub("untouched".into(), ""), "untouched");
    }

    #[test]
    fn new_builds_host_per_region_and_grid_per_ip() {
        let s = AstralaneSender::new(
            3,
            "astralane-fra",
            "https://{region}.gateway.astralane.io",
            vec!["fr".into(), "ams".into()],
            vec!["127.0.0.1".into(), "127.0.0.2".into(), "127.0.0.3".into()],
            "KEY-123",
            0,
        );
        assert_eq!(s.hosts.len(), 2);
        assert_eq!(
            s.hosts[0],
            "https://fr.gateway.astralane.io/irisb?api-key=KEY-123&method=sendTransaction"
        );
        // grid is host × ip
        assert_eq!(s.ip_count, 3);
        assert_eq!(s.grid.len(), 2);
        assert_eq!(s.grid[0].len(), 3);
        assert_eq!(s.grid[1].len(), 3);
        // display never contains the key
        assert!(s.hosts_display.iter().all(|h| !h.contains("KEY-123")));
    }

    #[test]
    fn empty_outbound_ips_yields_ip_count_one() {
        let s = AstralaneSender::new(
            3,
            "astralane-fra",
            "https://fr.gateway.astralane.io",
            vec![],
            vec![],
            "k",
            0,
        );
        assert_eq!(s.ip_count, 1);
        assert_eq!(s.hosts.len(), 1);
        assert_eq!(s.grid[0].len(), 1);
    }

    #[test]
    fn ip_cursor_rotates_round_robin() {
        let s = AstralaneSender::new(
            3,
            "astralane-fra",
            "https://fr.gateway.astralane.io",
            vec!["fr".into()],
            vec!["127.0.0.1".into(), "127.0.0.2".into()],
            "k",
            0,
        );
        let a = s.next_ip_idx();
        let b = s.next_ip_idx();
        let c = s.next_ip_idx();
        assert_eq!((a, b, c), (0, 1, 0));
    }

    #[test]
    fn endpoint_url_is_template_without_key() {
        let s = AstralaneSender::new(
            3,
            "astralane-fra",
            "https://{region}.gateway.astralane.io",
            vec!["fr".into()],
            vec![],
            "SECRET-KEY",
            0,
        );
        assert_eq!(s.endpoint_url(), "https://{region}.gateway.astralane.io");
        assert!(!s.endpoint_url().contains("SECRET-KEY"));
    }

    #[test]
    fn protocol_label_is_astralane() {
        let s = AstralaneSender::new(3, "astralane-fra", "https://fr.gateway.astralane.io", vec![], vec![], "k", 0);
        assert_eq!(s.protocol(), "ASTRALANE");
        assert_eq!(s.id(), 3);
        assert_eq!(s.name(), "astralane-fra");
    }

    #[test]
    fn throttled_true_within_interval_false_beyond() {
        let now = Instant::now();
        let interval = Duration::from_millis(1000);
        assert!(throttled(Some(now - Duration::from_millis(100)), now, interval));
        assert!(!throttled(Some(now - Duration::from_millis(1500)), now, interval));
        assert!(!throttled(Some(now), now, Duration::ZERO));
        assert!(!throttled(None, now, interval));
    }

    #[tokio::test]
    async fn send_throttles_second_call_within_interval() {
        let s = AstralaneSender::new(
            3,
            "astralane-fra",
            "https://127.0.0.1:1",
            vec!["fr".into()],
            vec![],
            "k",
            10_000,
        );
        let tx = sample_tx();
        let first = s.send(&tx).await;
        assert_ne!(first.error.as_deref(), Some("throttled_local"));
        let second = s.send(&tx).await;
        assert_eq!(second.error.as_deref(), Some("throttled_local"));
        assert!(second.endpoint_url_used.is_none());
    }

    #[tokio::test]
    async fn send_with_no_hosts_returns_error() {
        let mut s = AstralaneSender::new(3, "astralane", "https://x", vec![], vec![], "k", 0);
        s.hosts.clear();
        let tx = sample_tx();
        let outcome = s.send(&tx).await;
        assert_eq!(outcome.error.as_deref(), Some("no regions configured"));
    }
}
