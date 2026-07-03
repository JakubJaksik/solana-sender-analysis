//! Triton One `sendtx` sender - the new direct-HTTP fast path (Yellowstone Jet).
//!
//! Unlike `triton.rs` (JSON-RPC `sendTransaction`, which wraps the tx in a
//! `{jsonrpc,id,method,params}` envelope the server must parse), `sendtx` POSTs
//! the RAW bincode wire bytes of the signed tx straight to Jet:
//!   * `POST {endpoint}/sendtx?response=none`
//!   * `Content-Type: application/octet-stream` - body is the unencoded
//!     `bincode(tx)` (server feeds it directly to `handle_raw_transaction`;
//!     0% encoding overhead vs ~+33% for base64). Must be <= 1232 bytes.
//!   * `response=none` → 200 with EMPTY body - we never read it on success and
//!     take the signature locally from `tx.signatures[0]` (fire-and-forget).
//!   * SWQoS is on by default server-side (Jet); no tip, priority fee drives
//!     inclusion - same as the JSON-RPC Triton path.
//!
//! Transport: the OSS Jet server is hyper/TCP (HTTP/1.1 + h2), NOT HTTP/3. The
//! fastest *verified* path is plain `reqwest` with ALPN-negotiated h2 over the
//! Triton TLS edge (do NOT force `http2_prior_knowledge` - that is cleartext-h2
//! only). `tcp_nodelay` flushes the small tx immediately; the connection is
//! pre-warmed off the hot path.
//!
//! Auth (two forms, selected by config, single code path):
//!   * header auth (recommended/documented): `endpoint_url` = host base, token
//!     in `api_key` → sent as `x-token: <token>`.
//!   * token-in-path: `endpoint_url` = `https://<ep>.rpcpool.com/<TOKEN>`,
//!     `api_key` empty → token already in the URL, no `x-token` header.
//!
//! The token is NEVER logged - `endpoint_url()` returns a redacted (scheme +
//! host) form, and `x-token` is only ever a request header.

use super::{SendOutcome, TxSender};
use async_trait::async_trait;
use solana_sdk::transaction::Transaction;
use std::time::{Duration, Instant};

/// Strip any secret token (URL path) for safe logging: keep scheme + host only.
/// `https://name.mainnet.rpcpool.com/TOKEN/sendtx` -> `https://name.mainnet.rpcpool.com`.
fn redact_endpoint(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let host = rest.split('/').next().unwrap_or(rest);
            format!("{scheme}://{host}")
        }
        None => url.split('/').next().unwrap_or(url).to_string(),
    }
}

/// Build the POST target: append the `sendtx` route and the fire-and-forget
/// `response=none` query to the configured endpoint base. Works for both the
/// header-auth form (`endpoint` = host) and the token-in-path form
/// (`endpoint` = `https://<ep>.rpcpool.com/<TOKEN>`). A trailing slash on the
/// base is trimmed so we never produce `//sendtx`.
fn build_post_url(endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    format!("{base}/sendtx?response=none")
}

/// The raw wire bytes posted to `sendtx`: `bincode(tx)`, unencoded. This is the
/// exact octet-stream body - no base64/base58 (the server rejects `encoding=`
/// combined with `application/octet-stream` and decodes octet-stream verbatim).
fn wire_bytes(tx: &Transaction) -> Vec<u8> {
    bincode::serialize(tx).unwrap_or_default()
}

/// The `x-token` header value: `Some(token)` when a non-empty token is
/// configured (header-auth form), `None` otherwise (token-in-path form, where
/// the token already lives in the URL).
fn auth_token(token: &str) -> Option<&str> {
    let t = token.trim();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

pub struct TritonSendTxSender {
    id: u8,
    name: String,
    /// Full POST target incl. token-in-path (if used) + `/sendtx?response=none`.
    /// Private - used ONLY as the POST/warm target.
    endpoint: String,
    /// Token-redacted (scheme + host). Returned by `endpoint_url()` and used in
    /// `SendOutcome.endpoint_url_used` so the token never reaches logs/records.
    endpoint_display: String,
    /// `x-token` header value. Empty = token-in-path form (no header). Secret -
    /// only ever sent as a request header, never logged.
    token: String,
    client: reqwest::Client,
}

impl TritonSendTxSender {
    pub fn new(
        id: u8,
        name: impl Into<String>,
        endpoint_url: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        let endpoint = build_post_url(&endpoint_url.into());
        let endpoint_display = redact_endpoint(&endpoint);
        // Default ALPN: negotiates h2 over the Triton TLS edge. NOT
        // `http2_prior_knowledge` (cleartext-h2 only - would break the TLS
        // handshake). `tcp_nodelay` flushes the small tx without Nagle delay.
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
            endpoint,
            endpoint_display,
            token: token.into(),
            client,
        }
    }

    /// Fire-and-forget connection pre-warm: a cheap `GET /sendtx` (the server
    /// answers 405 to non-POST, which is fine - the point is to pay the
    /// TCP+TLS+h2 handshake off the hot path so the first real send reuses a
    /// warm keep-alive connection). `reqwest::Client` clones share the pool.
    pub fn spawn_warmup(&self, handle: &tokio::runtime::Handle) {
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let token = self.token.clone();
        handle.spawn(async move {
            let mut req = client.get(&endpoint);
            if let Some(t) = auth_token(&token) {
                req = req.header("x-token", t);
            }
            let _ = req.send().await;
        });
    }
}

#[async_trait]
impl TxSender for TritonSendTxSender {
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
        "TRITON_SENDTX"
    }

    async fn send(&self, tx: &Transaction) -> SendOutcome {
        // `response=none` returns an empty body, so the signature is taken
        // locally - it is exactly `tx.signatures[0]`, the on-chain signature.
        let signature = tx.signatures.first().copied().unwrap_or_default();
        let body = wire_bytes(tx);
        let redacted = self.endpoint_display.clone();

        let mut req = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/octet-stream")
            .body(body);
        if let Some(t) = auth_token(&self.token) {
            req = req.header("x-token", t);
        }

        let send_at = Instant::now();
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
                error: Some(format!("network: {e}")),
                endpoint_url_used: Some(redacted),
            },
            Ok(resp) => {
                let status = resp.status().as_u16();
                if (200..300).contains(&status) {
                    // Success: empty body (response=none). Don't read it - the
                    // status line is all that matters on the hot path.
                    SendOutcome {
                        send_at,
                        send_ack_at,
                        signature,
                        http_status: Some(status),
                        rpc_err_code: None,
                        rpc_err_message: None,
                        provider_request_id: None,
                        error: None,
                        endpoint_url_used: Some(redacted),
                    }
                } else {
                    // Error: read the text/plain body for diagnostics (off the
                    // success path). 4xx/5xx carry the Jet error message.
                    let body_text = resp.text().await.unwrap_or_default();
                    SendOutcome {
                        send_at,
                        send_ack_at,
                        signature,
                        http_status: Some(status),
                        rpc_err_code: None,
                        rpc_err_message: Some(body_text.clone()),
                        provider_request_id: None,
                        error: Some(format!("HTTP {status}: {body_text}")),
                        endpoint_url_used: Some(redacted),
                    }
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
    fn build_post_url_appends_sendtx_route_and_response_none() {
        assert_eq!(
            build_post_url("https://my-app.mainnet.rpcpool.com/TOKEN"),
            "https://my-app.mainnet.rpcpool.com/TOKEN/sendtx?response=none"
        );
        assert_eq!(
            build_post_url("https://my-app.mainnet.rpcpool.com"),
            "https://my-app.mainnet.rpcpool.com/sendtx?response=none"
        );
    }

    #[test]
    fn build_post_url_trims_trailing_slash() {
        assert_eq!(
            build_post_url("https://my-app.mainnet.rpcpool.com/"),
            "https://my-app.mainnet.rpcpool.com/sendtx?response=none"
        );
    }

    #[test]
    fn wire_bytes_are_raw_bincode_not_base64() {
        let tx = sample_tx();
        let expected = bincode::serialize(&tx).unwrap();
        assert_eq!(wire_bytes(&tx), expected);
        // Raw tx is well under the 1232-byte octet-stream limit.
        assert!(wire_bytes(&tx).len() <= 1232);
    }

    #[test]
    fn auth_token_present_only_when_non_empty() {
        assert_eq!(auth_token("SECRET-TOKEN-123"), Some("SECRET-TOKEN-123"));
        assert_eq!(auth_token("  tok  "), Some("tok"));
        assert_eq!(auth_token(""), None);
        assert_eq!(auth_token("   "), None);
    }

    #[test]
    fn redact_endpoint_strips_token_path() {
        let url = "https://my-app.mainnet.rpcpool.com/SECRET-TOKEN-123/sendtx?response=none";
        let red = redact_endpoint(url);
        assert_eq!(red, "https://my-app.mainnet.rpcpool.com");
        assert!(!red.contains("SECRET-TOKEN-123"));
    }

    #[test]
    fn endpoint_url_redacts_token_in_path() {
        let s = TritonSendTxSender::new(
            8,
            "triton-sendtx-fra",
            "https://my-app.mainnet.rpcpool.com/SECRET-TOKEN-123",
            "",
        );
        assert_eq!(s.endpoint_url(), "https://my-app.mainnet.rpcpool.com");
        assert!(!s.endpoint_url().contains("SECRET-TOKEN-123"));
    }

    #[test]
    fn endpoint_url_never_contains_header_token() {
        let s = TritonSendTxSender::new(
            8,
            "triton-sendtx-fra",
            "https://my-app.mainnet.rpcpool.com",
            "SECRET-TOKEN-XYZ",
        );
        assert!(!s.endpoint_url().contains("SECRET-TOKEN-XYZ"));
    }

    #[test]
    fn protocol_label_is_triton_sendtx() {
        let s = TritonSendTxSender::new(
            8,
            "triton-sendtx-fra",
            "https://x.mainnet.rpcpool.com/t",
            "",
        );
        assert_eq!(s.protocol(), "TRITON_SENDTX");
        assert_eq!(s.id(), 8);
        assert_eq!(s.name(), "triton-sendtx-fra");
    }
}
