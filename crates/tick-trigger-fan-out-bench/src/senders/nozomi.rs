//! Temporal **Nozomi** sender - direct-HTTP fast path with multi-region fan-out.
//!
//! Nozomi's lowest-latency submit path is `POST /api/sendTransaction2`:
//!   * `Content-Type: text/plain`, body = `base64(bincode(tx))` as PLAINTEXT
//!     (NOT JSON-RPC wrapped). Strips the envelope parse, like Triton `/sendtx`
//!     - but Nozomi wants base64 text, not raw octet-stream.
//!   * Success = **HTTP 200, EMPTY body, no signature** → we take the signature
//!     locally from `tx.signatures[0]` (fire-and-forget). 4xx/5xx carry a
//!     text/plain error (400 = insufficient tip, 429 = rate-limited) which we
//!     read only off the success path.
//!   * Auth is a **URL query param `?c=<API_KEY>`** on every request - NOT a
//!     header (differs from Syncro's Bearer). The key is secret: it lives only
//!     in the private URL, is stripped from `endpoint_url()`/`endpoint_url_used`
//!     and scrubbed from network-error strings.
//!
//! **Multi-region fan-out** (docs explicitly recommend it): the SAME signed tx
//! is POSTed to N regional hosts concurrently; the first reply wins (latency /
//! landing redundancy). Because every region gets byte-identical bytes (same
//! signature), Solana leader-level dedup lands exactly ONE copy → exactly ONE
//! tip is charged. So - unlike Jito - the tip is ONE cluster-wide in-tx
//! transfer (added by the preparer via `tip_accounts_for(Nozomi)`, min
//! 1_000_000 lamports); it is NOT varied per region.
//!
//! Config: `endpoint_url` is a host template with `{region}` (e.g.
//! `http://{region}.nozomi.temporal.xyz`), `regions` the list (e.g.
//! `["fra2","ams1","lon1"]`), `api_key` the Nozomi key. Plain `http://` (port
//! 80) to a direct host skips TLS and is blessed for datacenter/VPS colos.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use base64::Engine as _;
use solana_sdk::transaction::Transaction;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Nozomi fast-path send route (appended to the per-region host).
const SEND_PATH: &str = "/api/sendTransaction2";
/// Keep-alive ping route (used by `spawn_warmup`).
const PING_PATH: &str = "/ping";

/// Substitute the `{region}` placeholder in a host template.
fn substitute_region(template: &str, region: &str) -> String {
    template.replace("{region}", region)
}

/// Full fast-path POST URL incl. the secret key: `{host}/api/sendTransaction2?c=<key>`.
/// PRIVATE - used only as the POST target; never logged.
fn build_send_url(host: &str, api_key: &str) -> String {
    let base = host.trim_end_matches('/');
    format!("{base}{SEND_PATH}?c={api_key}")
}

/// Keep-alive ping URL: `{host}/ping?c=<key>`. PRIVATE.
fn build_ping_url(host: &str, api_key: &str) -> String {
    let base = host.trim_end_matches('/');
    format!("{base}{PING_PATH}?c={api_key}")
}

/// Strip the `?c=<key>` query for safe logging - keep scheme + host + path.
fn redact_query(url: &str) -> String {
    match url.split_once('?') {
        Some((before, _)) => before.to_string(),
        None => url.to_string(),
    }
}

/// Scrub the secret key out of an arbitrary message (reqwest error Display can
/// embed the request URL incl. the `?c=<key>` query). No-op when key is empty.
fn scrub(msg: String, api_key: &str) -> String {
    if api_key.is_empty() {
        msg
    } else {
        msg.replace(api_key, "***")
    }
}

/// The plaintext body for `/api/sendTransaction2`: `base64(bincode(tx))`.
fn tx_to_base64(tx: &Transaction) -> String {
    let raw = bincode::serialize(tx).unwrap_or_default();
    base64::engine::general_purpose::STANDARD.encode(&raw)
}

/// True if a send at `now` falls within `interval` of the previous send and
/// should be throttled. Mirrors the Syncro/Jito local rate-limit. Lets a 1 TPS
/// Nozomi key pace DISTINCT triggers to avoid 429s and the 30-min QoS penalty.
/// The multi-region fan-out of ONE tx is a single distinct tx (exempt from
/// Nozomi's per-region limit), so the throttle paces triggers, not the fan-out.
fn throttled(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    interval > Duration::ZERO && matches!(last, Some(prev) if now.duration_since(prev) < interval)
}

/// One regional fan-out result. Mirrors the Jito sender: the first reply
/// (success OR error) wins the race; on-chain landing is judged later by the
/// recorder matching the signature.
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

pub struct NozomiSender {
    id: u8,
    name: String,
    /// Host template (no key) - returned by `endpoint_url()` for logging.
    endpoint_template: String,
    /// Secret key (query-param auth). Used to build URLs and to scrub errors.
    api_key: String,
    /// Per-region full POST URLs incl. `?c=<key>` - PRIVATE.
    hosts: Vec<String>,
    /// Per-region redacted POST URLs (no key) - for `endpoint_url_used`.
    hosts_display: Vec<String>,
    /// Per-region keep-alive ping URLs incl. `?c=<key>` - PRIVATE.
    warm_urls: Vec<String>,
    client: reqwest::Client,
    /// Local throttle: minimum interval between distinct sends. Zero = off.
    min_send_interval: Duration,
    last_send_at: parking_lot::Mutex<Option<Instant>>,
}

impl NozomiSender {
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint_template: impl Into<String>,
        regions: Vec<String>,
        api_key: impl Into<String>,
        min_send_interval_ms: u64,
    ) -> Self {
        let endpoint_template = endpoint_template.into();
        let api_key = api_key.into();
        // No regions → treat the template as a single concrete host.
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
        let warm_urls: Vec<String> = region_hosts.iter().map(|h| build_ping_url(h, &api_key)).collect();
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
            endpoint_template,
            api_key,
            hosts,
            hosts_display,
            warm_urls,
            client,
            min_send_interval: Duration::from_millis(min_send_interval_ms),
            last_send_at: parking_lot::Mutex::new(None),
        }
    }

    /// Fire-and-forget pre-warm: a cheap `GET /ping` per region so the first
    /// real send reuses a warm keep-alive connection instead of paying the
    /// TCP(+TLS) handshake on the hot path. `reqwest::Client` clones share the
    /// pool.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        for url in &self.warm_urls {
            let client = self.client.clone();
            let url = url.clone();
            handle.spawn(async move {
                let _ = client.get(&url).send().await;
            });
        }
    }
}

#[async_trait]
impl TxSender for NozomiSender {
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
        "NOZOMI"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        let signature = tx.signatures.first().copied().unwrap_or_default();

        // Local throttle: pace DISTINCT triggers to respect the 1 TPS key cap
        // and avoid the QoS penalty. The fan-out below is one distinct tx
        // (exempt from per-region limits), so it is not throttled.
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

        let body: Arc<String> = Arc::new(tx_to_base64(tx));
        let (tx_first, mut rx_first) =
            tokio::sync::mpsc::channel::<FanoutReply>(self.hosts.len());

        for i in 0..self.hosts.len() {
            let url = self.hosts[i].clone();
            let host_display = self.hosts_display[i].clone();
            let body = body.clone();
            let client = self.client.clone();
            let tx_first = tx_first.clone();
            let api_key = self.api_key.clone();
            tokio::spawn(async move {
                let result = client
                    .post(&url)
                    .header("Content-Type", "text/plain")
                    .body((*body).clone())
                    .send()
                    .await;
                let send_ack_at = Instant::now();
                let reply = match result {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        if (200..300).contains(&status) {
                            // Success: empty body - don't read it.
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
            substitute_region("http://{region}.nozomi.temporal.xyz", "fra2"),
            "http://fra2.nozomi.temporal.xyz"
        );
    }

    #[test]
    fn build_send_url_appends_route_and_key() {
        assert_eq!(
            build_send_url("http://fra2.nozomi.temporal.xyz", "KEY-123"),
            "http://fra2.nozomi.temporal.xyz/api/sendTransaction2?c=KEY-123"
        );
        // trailing slash on host is trimmed (no // before the route)
        assert_eq!(
            build_send_url("http://fra2.nozomi.temporal.xyz/", "KEY-123"),
            "http://fra2.nozomi.temporal.xyz/api/sendTransaction2?c=KEY-123"
        );
    }

    #[test]
    fn build_ping_url_appends_ping_and_key() {
        assert_eq!(
            build_ping_url("http://fra2.nozomi.temporal.xyz", "KEY-123"),
            "http://fra2.nozomi.temporal.xyz/ping?c=KEY-123"
        );
    }

    #[test]
    fn redact_query_strips_the_key() {
        let red = redact_query("http://fra2.nozomi.temporal.xyz/api/sendTransaction2?c=SECRET-KEY");
        assert_eq!(red, "http://fra2.nozomi.temporal.xyz/api/sendTransaction2");
        assert!(!red.contains("SECRET-KEY"));
    }

    #[test]
    fn scrub_removes_key_from_error_text() {
        let msg = "network: error for url (http://x/api/sendTransaction2?c=SECRET-KEY)".to_string();
        let safe = scrub(msg, "SECRET-KEY");
        assert!(!safe.contains("SECRET-KEY"));
        assert!(safe.contains("***"));
        // empty key is a no-op (no panic, no spurious replacement)
        assert_eq!(scrub("untouched".into(), ""), "untouched");
    }

    #[test]
    fn tx_to_base64_is_base64_of_bincode() {
        let tx = sample_tx();
        let expected = base64::engine::general_purpose::STANDARD
            .encode(bincode::serialize(&tx).unwrap());
        assert_eq!(tx_to_base64(&tx), expected);
    }

    #[test]
    fn new_builds_one_host_per_region_with_key() {
        let s = NozomiSender::new(
            8,
            "nozomi-fra",
            "http://{region}.nozomi.temporal.xyz",
            vec!["fra2".into(), "ams1".into(), "lon1".into()],
            "KEY-123",
            0,
        );
        assert_eq!(s.hosts.len(), 3);
        assert_eq!(s.hosts[0], "http://fra2.nozomi.temporal.xyz/api/sendTransaction2?c=KEY-123");
        assert_eq!(s.warm_urls[0], "http://fra2.nozomi.temporal.xyz/ping?c=KEY-123");
        // display forms never contain the key
        assert!(s.hosts_display.iter().all(|h| !h.contains("KEY-123")));
    }

    #[test]
    fn new_with_no_regions_uses_template_as_single_host() {
        let s = NozomiSender::new(
            8,
            "nozomi-single",
            "http://fra2.nozomi.temporal.xyz",
            vec![],
            "KEY-123",
            0,
        );
        assert_eq!(s.hosts.len(), 1);
        assert_eq!(s.hosts[0], "http://fra2.nozomi.temporal.xyz/api/sendTransaction2?c=KEY-123");
    }

    #[test]
    fn endpoint_url_is_template_without_key() {
        let s = NozomiSender::new(
            8,
            "nozomi-fra",
            "http://{region}.nozomi.temporal.xyz",
            vec!["fra2".into()],
            "SECRET-KEY",
            0,
        );
        assert_eq!(s.endpoint_url(), "http://{region}.nozomi.temporal.xyz");
        assert!(!s.endpoint_url().contains("SECRET-KEY"));
    }

    #[test]
    fn protocol_label_is_nozomi() {
        let s = NozomiSender::new(8, "nozomi-fra", "http://{region}.nozomi.temporal.xyz", vec!["fra2".into()], "k", 0);
        assert_eq!(s.protocol(), "NOZOMI");
        assert_eq!(s.id(), 8);
        assert_eq!(s.name(), "nozomi-fra");
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
        assert!(throttled(Some(now - Duration::from_millis(100)), now, interval));
        assert!(!throttled(Some(now - Duration::from_millis(1500)), now, interval));
    }

    #[tokio::test]
    async fn send_throttles_second_call_within_interval() {
        let s = NozomiSender::new(
            8,
            "nozomi-fra",
            "http://127.0.0.1:1",
            vec!["fra2".into()],
            "k",
            10_000,
        );
        let tx = sample_tx();
        // First call is not throttled (it will network-fail against the dead
        // address, but it is NOT the throttle path).
        let first = s.send(&tx).await;
        assert_ne!(first.error.as_deref(), Some("throttled_local"));
        // Second call within the 10s interval is short-circuited locally.
        let second = s.send(&tx).await;
        assert_eq!(second.error.as_deref(), Some("throttled_local"));
        assert!(second.endpoint_url_used.is_none());
    }

    #[tokio::test]
    async fn send_with_no_hosts_returns_error() {
        // Empty template + no regions still yields one host; to hit the
        // "no regions" guard we construct then clear - simulate via a sender
        // whose host list is empty by using an explicitly empty template list.
        let mut s = NozomiSender::new(8, "nozomi", "http://x", vec![], "k", 0);
        s.hosts.clear();
        s.hosts_display.clear();
        let tx = sample_tx();
        let outcome = s.send(&tx).await;
        assert_eq!(outcome.error.as_deref(), Some("no regions configured"));
    }
}
