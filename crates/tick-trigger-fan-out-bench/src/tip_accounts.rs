//! Per-vendor tip account lists + round-robin rotator.
//!
//! Each vendor publishes a small set of tip accounts; rotating per tx
//! balances load and avoids the "all txs hit the same account" anti-pattern
//! that vendors sometimes rate-limit on. Future phases (Jito, Nozomi,
//! bloXroute, etc.) get their own slices here.

use crate::config::SenderKind;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Helius `/fast` SWQoS tip accounts (mainnet). Published in their docs.
/// Rotating through these distributes load and avoids one account becoming
/// a bottleneck.
const HELIUS_TIP_ACCOUNTS_STR: &[&str] = &[
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvAfrLT",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
    "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
];

pub fn helius_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        HELIUS_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded helius tip account parses"))
            .collect()
    })
}

/// Jito Block Engine tip accounts (mainnet). 8 accounts; rotation spreads
/// write-lock contention so concurrent tips don't serialize on a single
/// account. Source: `getTipAccounts` RPC response on Jito searcher API.
const JITO_TIP_ACCOUNTS_STR: &[&str] = &[
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
];

pub fn jito_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        JITO_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded jito tip account parses"))
            .collect()
    })
}

/// Syncro Sender (P2P.org) tip accounts (mainnet). 9 accounts; rotation spreads
/// write-lock contention. Source: docs.p2p.org/docs/syncro-sender-quick-start (2026-06).
const SYNCRO_TIP_ACCOUNTS_STR: &[&str] = &[
    "BPZrtYhdoAhiHWV5EgGLoV7bZFbMamBZurGDq4DmST8v",
    "7D5pdbkV75Sr73M1YFNZwXMed6DenwkdfbJwVWrX6drQ",
    "ELpn2NryEW4B3psG36eSjF45YcGMQpGGuu9J2AgAccbV",
    "FnckAPC9PitnRpGZM2M4WLwb3w9odRLJ7EDRZDngjvd6",
    "3ZnDTgvVfwzqwWoqAUmDkgVtXvXqjmeb5t9zxD5pMbmv",
    "3SLDFcdCzMbcFNguZhzmV4zqEAUvcPoKY13akpE4Tq1p",
    "48tT6LJqrsoFrLpzZSHkjGdGTWtsJ1PvjgWZjh8qF1RK",
    "7GM9fpVMHHcrK4cgzfVdzJvjiy1bSyfwSYzhxvgbfVLg",
    "CBd8GE3ffMJKf3iCCcNNBEifMxH1WpgtTzRnXPxxbjGE",
];

pub fn syncro_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        SYNCRO_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded syncro tip account parses"))
            .collect()
    })
}

/// 0slot.trade tip accounts (mainnet). 5 accounts; rotation spreads write-lock
/// contention. Source: 0slot onboarding (staked_conn), 2026-06. Min tip 0.001 SOL.
const ZEROSLOT_TIP_ACCOUNTS_STR: &[&str] = &[
    "Eb2KpSC8uMt9GmzyAEm5Eb1AAAgTjRaXWFjKyFXHZxF3",
    "FCjUJZ1qozm1e8romw216qyfQMaaWKxWsuySnumVCCNe",
    "ENxTEjSQ1YabmUpXAdCgevnHQ9MHdLv8tzFiuiYJqa13",
    "6rYLG55Q9RpsPGvqdPNJs4z5WTxJVatMB8zV3WJhs5EK",
    "Cix2bHfqPcKcM233mzxbLk14kSggUUiz2A87fJtGivXr",
];

pub fn zeroslot_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        ZEROSLOT_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded 0slot tip account parses"))
            .collect()
    })
}

/// AllenHark Relay tip accounts (mainnet). 11 accounts; rotation spreads
/// write-lock contention. Source: AllenHark console (2026-06). Min tip 0.001 SOL.
const ALLENHARK_TIP_ACCOUNTS_STR: &[&str] = &[
    "hark1zxc5Rz3K8Kquz79WPWFEgNCFeJnsMJ16f22uNP",
    "harkm2BTWxZuszoNpZnfe84jRbQTg6KGHaQBmWzDGQQ",
    "hark4CwtTnN2y9FaxjcFBAJdJqQrpouu5pgEixfqdEz",
    "harkoJfnM6dxrJydx5eVmDVwAgwC94KbhuxF69UbXwP",
    "hark6hUDUTekc1DGxWdJcuyDZwf6pJdCxd4SXAVtta6",
    "harkoTvFpKSrEQduYrNHXCurARVT19Ud3BnFhVxabos",
    "harkEpXoJv5qVzHaN7HSuUAd6PHjyMcFMcDYBMDJCEQ",
    "harkyXDdZSoJGyCxa24t2QXx1poPyp8YfghbtpzGSzK",
    "harkR2YJ4Dpt4UDJTcBirjnSPBhNpQFcoFkNpCkVqNk",
    "harkRBygM8pHYe4K8eBjfxyEX19oJn3LepFjvNbLbyi",
    "harkYFxB6DuUFNwDLvA5CQ66KpfRvFgUoVypMagNcmd",
];

pub fn allenhark_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        ALLENHARK_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded allenhark tip account parses"))
            .collect()
    })
}

/// BlockRazor "Fast" tip accounts (mainnet). 14 rotating accounts; rotation
/// spreads write-lock contention. Source: blockrazor.gitbook.io (2026-06).
/// Min tip 0.0001 SOL (100_000 lamports).
const BLOCKRAZOR_TIP_ACCOUNTS_STR: &[&str] = &[
    "FjmZZrFvhnqqb9ThCuMVnENaM3JGVuGWNyCAxRJcFpg9",
    "6No2i3aawzHsjtThw81iq1EXPJN6rh8eSJCLaYZfKDTG",
    "A9cWowVAiHe9pJfKAj3TJiN9VpbzMUq6E4kEvf5mUT22",
    "Gywj98ophM7GmkDdaWs4isqZnDdFCW7B46TXmKfvyqSm",
    "68Pwb4jS7eZATjDfhmTXgRJjCiZmw1L7Huy4HNpnxJ3o",
    "4ABhJh5rZPjv63RBJBuyWzBK3g9gWMUQdTZP2kiW31V9",
    "B2M4NG5eyZp5SBQrSdtemzk5TqVuaWGQnowGaCBt8GyM",
    "5jA59cXMKQqZAVdtopv8q3yyw9SYfiE3vUCbt7p8MfVf",
    "5YktoWygr1Bp9wiS1xtMtUki1PeYuuzuCF98tqwYxf61",
    "295Avbam4qGShBYK7E9H5Ldew4B3WyJGmgmXfiWdeeyV",
    "EDi4rSy2LZgKJX74mbLTFk4mxoTgT6F7HxxzG2HBAFyK",
    "BnGKHAC386n4Qmv9xtpBVbRaUTKixjBe3oagkPFKtoy6",
    "Dd7K2Fp7AtoN8xCghKDRmyqr5U169t48Tw5fEd3wT9mq",
    "AP6qExwrbRgBAVaehg4b5xHENX815sMabtBzUzVB4v8S",
];

pub fn blockrazor_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        BLOCKRAZOR_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded blockrazor tip account parses"))
            .collect()
    })
}

/// Temporal Nozomi tip accounts (mainnet). 17 public accounts; rotation spreads
/// write-lock contention. The tip is ONE in-tx SystemProgram::transfer to one of
/// these (min 1_000_000 lamports), charged only on landing. Same tip serves all
/// fan-out regions (one signed tx → one landed copy → one tip). Source:
/// use.temporal.xyz/nozomi (+ official temporalxyz examples), 2026-06.
const NOZOMI_TIP_ACCOUNTS_STR: &[&str] = &[
    "TEMPaMeCRFAS9EKF53Jd6KpHxgL47uWLcpFArU1Fanq",
    "noz3jAjPiHuBPqiSPkkugaJDkJscPuRhYnSpbi8UvC4",
    "noz3str9KXfpKknefHji8L1mPgimezaiUyCHYMDv1GE",
    "noz6uoYCDijhu1V7cutCpwxNiSovEwLdRHPwmgCGDNo",
    "noz9EPNcT7WH6Sou3sr3GGjHQYVkN3DNirpbvDkv9YJ",
    "nozc5yT15LazbLTFVZzoNZCwjh3yUtW86LoUyqsBu4L",
    "nozFrhfnNGoyqwVuwPAW4aaGqempx4PU6g6D9CJMv7Z",
    "nozievPk7HyK1Rqy1MPJwVQ7qQg2QoJGyP71oeDwbsu",
    "noznbgwYnBLDHu8wcQVCEw6kDrXkPdKkydGJGNXGvL7",
    "nozNVWs5N8mgzuD3qigrCG2UoKxZttxzZ85pvAQVrbP",
    "nozpEGbwx4BcGp6pvEdAh1JoC2CQGZdU6HbNP1v2p6P",
    "nozrhjhkCr3zXT3BiT4WCodYCUFeQvcdUkM7MqhKqge",
    "nozrwQtWhEdrA6W8dkbt9gnUaMs52PdAv5byipnadq3",
    "nozUacTVWub3cL4mJmGCYjKZTnE9RbdY5AP46iQgbPJ",
    "nozWCyTPppJjRuw2fpzDhhWbW355fzosWSzrrMYB1Qk",
    "nozWNju6dY353eMkMqURqwQEoM3SFgEKC6psLCSfUne",
    "nozxNBgWohjR75vdspfxR5H9ceC7XXH99xpxhVGt3Bb",
];

pub fn nozomi_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        NOZOMI_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded nozomi tip account parses"))
            .collect()
    })
}

/// Astralane Iris/Gateway tip accounts (mainnet). 17 `astra*` vanity accounts;
/// rotation spreads write-lock contention. The tip is ONE mandatory in-tx
/// SystemProgram::transfer to one of these (tier-gated min; 10_000 lamports
/// floor, 1_000_000 on the Free tier). The primary/canonical account (in every
/// official example) is `astra4uejePWneqNaJKuFFA8oonqCE1sqF6b45kDMZm`; the full
/// list is from Astralane's `llms-full.txt` corpus - verify the extras against
/// https://tips.astralane.io/tip_floor before relying on them. Source:
/// astralane.gitbook.io/docs (2026-06).
const ASTRALANE_TIP_ACCOUNTS_STR: &[&str] = &[
    "astra4uejePWneqNaJKuFFA8oonqCE1sqF6b45kDMZm",
    "astrazznxsGUhWShqgNtAdfrzP2G83DzcWVJDxwV9bF",
    "astra9xWY93QyfG6yM8zwsKsRodscjQ2uU2HKNL5prk",
    "astraRVUuTHjpwEVvNBeQEgwYx9w9CFyfxjYoobCZhL",
    "astraEJ2fEj8Xmy6KLG7B3VfbKfsHXhHrNdCQx7iGJK",
    "astraubkDw81n4LuutzSQ8uzHCv4BhPVhfvTcYv8SKC",
    "astraZW5GLFefxNPAatceHhYjfA1ciq9gvfEg2S47xk",
    "astrawVNP4xDBKT7rAdxrLYiTSTdqtUr63fSMduivXK",
    "AstrA1ejL4UeXC2SBP4cpeEmtcFPZVLxx3XGKXyCW6to",
    "AsTra79FET4aCKWspPqeSFvjJNyp96SvAnrmyAxqg5b7",
    "AstrABAu8CBTyuPXpV4eSCJ5fePEPnxN8NqBaPKQ9fHR",
    "AsTRADtvb6tTmrsqULQ9Wji9PigDMjhfEMza6zkynEvV",
    "AsTRAEoyMofR3vUPpf9k68Gsfb6ymTZttEtsAbv8Bk4d",
    "AStrAJv2RN2hKCHxwUMtqmSxgdcNZbihCwc1mCSnG83W",
    "Astran35aiQUF57XZsmkWMtNCtXGLzs8upfiqXxth2bz",
    "AStRAnpi6kFrKypragExgeRoJ1QnKH7pbSjLAKQVWUum",
    "ASTRaoF93eYt73TYvwtsv6fMWHWbGmMUZfVZPo3CRU9C",
];

pub fn astralane_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        ASTRALANE_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded astralane tip account parses"))
            .collect()
    })
}

/// NextBlock (nextblock.io) tip wallets (mainnet). 8 accounts; rotation spreads
/// per-slot wallet limits (NextBlock explicitly recommends randomizing). The tip
/// is ONE mandatory in-tx SystemProgram::transfer to one of these, minimum
/// 1_000_000 lamports (0.001 SOL, per NextBlock docs). Source:
/// github.com/nextblock-ag/nextblock-proto README (2026-06).
const NEXTBLOCK_TIP_ACCOUNTS_STR: &[&str] = &[
    "NextbLoCkVtMGcV47JzewQdvBpLqT9TxQFozQkN98pE",
    "NexTbLoCkWykbLuB1NkjXgFWkX9oAtcoagQegygXXA2",
    "NeXTBLoCKs9F1y5PJS9CKrFNNLU1keHW71rfh7KgA1X",
    "NexTBLockJYZ7QD7p2byrUa6df8ndV2WSd8GkbWqfbb",
    "neXtBLock1LeC67jYd1QdAa32kbVeubsfPNTJC1V5At",
    "nEXTBLockYgngeRmRrjDV31mGSekVPqZoMGhQEZtPVG",
    "NEXTbLoCkB51HpLBLojQfpyVAMorm3zzKg7w9NFdqid",
    "nextBLoCkPMgmG8ZgJtABeScP35qLa2AMCNKntAP7Xc",
];

pub fn nextblock_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        NEXTBLOCK_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded nextblock tip account parses"))
            .collect()
    })
}

/// bloXroute Solana Trader API tip wallets (mainnet). 17 accounts; rotation
/// spreads write-lock contention (bloXroute recommends randomizing). The tip is
/// ONE mandatory in-tx SystemProgram::transfer to one of these, minimum
/// 1_000_000 lamports (0.001 SOL). `useStakedRPCs` REQUIRES the tip. Verified
/// verbatim against docs.bloxroute.com tipping-addresses (2026-06).
const BLOXROUTE_TIP_ACCOUNTS_STR: &[&str] = &[
    "3UQUKjhMKaY2S6bjcQD6yHB7utcZt5bfarRCmctpRtUd",
    "FogxVNs6Mm2w9rnGL1vkARSwJxvLE8mujTv3LK8RnUhF",
    "bLx7MvxGaKdKL7mEbpk9tC79z6MnBSJoJkuaEAPu6Nd",
    "bLx7XBqSg3LUPVf1bRgCnkJmgVZR8QEgDJBPqcRLHvp",
    "bLx8KeZxinPwy6kkUgyzMLeqb2ARNsWjADG1dhSsVba",
    "bLxADBknoNj8WAGw2W6GBYeq848Xx6ajhaymV1YvrHm",
    "bLxAc88vRBwvcUQJEgcxNfBLvHPikY4csNsUmPeWea2",
    "bLxQ88oCiTsL8Xj4YWekKi1hjrgmbE3J3FFZ2xZHR3h",
    "bLxS7NoLuynNRJ4mCnEE2YbtwJFttYsEyp2ME7rp2yt",
    "bLxW6mCov7VEbrKc3S9tcBRcfSzRnLCbNp3Dfn3SJG5",
    "bLxXSGXs4mYPTC5okZXed1qzvjNwNJ48QJ82hT2V7w7",
    "bLxYi3vojbbB7hVzVDVTdBLVPhp7GJ3ZB3BwdK5sFXi",
    "bLxhLPgBXtUpX4b1bH3HatuMGMSKT9GnwtuCGiMSAqe",
    "bLxpY1mniuFW4PgkNA4JiNxoeKHFszryi6tNgyZAiAA",
    "bLxuETxd2tgWxBALNwPzAfHhsik4BzD3nrEBCiPNZQD",
    "bLxuL2gK5FW7xfahvwLrxLyW76vcCpNsKQY2CmnE6kV",
    "bLxv4Hnub7nDJWHs8s17o9bGU65Bnx6Yqp2fqtMgHmm",
];

pub fn bloxroute_tip_accounts() -> &'static [Pubkey] {
    static CACHED: OnceLock<Vec<Pubkey>> = OnceLock::new();
    CACHED.get_or_init(|| {
        BLOXROUTE_TIP_ACCOUNTS_STR
            .iter()
            .map(|s| Pubkey::from_str(s).expect("hardcoded bloxroute tip account parses"))
            .collect()
    })
}

/// Return the tip account list for a given sender kind. Empty slice means
/// no tip account (sender protocol does not use them).
pub fn tip_accounts_for(kind: SenderKind) -> &'static [Pubkey] {
    match kind {
        SenderKind::Helius => helius_tip_accounts(),
        SenderKind::Jito => jito_tip_accounts(),
        // Triton (Jet/SWQoS) needs no tip - inclusion is driven by priority fee.
        SenderKind::Triton => &[],
        // Triton sendtx (same Jet/SWQoS engine, direct-HTTP fast path) - no tip.
        SenderKind::TritonSendTx => &[],
        // Nozomi: mandatory in-tx tip (>= 1_000_000 lamports) to one of 17 accounts.
        SenderKind::Nozomi => nozomi_tip_accounts(),
        // Astralane (HTTP /irisb): mandatory in-tx tip to one of 17 astra* accounts.
        SenderKind::Astralane => astralane_tip_accounts(),
        // Astralane QUIC: same astra* tip accounts as the HTTP path.
        SenderKind::AstralaneQuic => astralane_tip_accounts(),
        // NextBlock: mandatory in-tx tip (>= 1_000_000 lamports) to one of 8 wallets.
        SenderKind::Nextblock => nextblock_tip_accounts(),
        // NextBlock QUIC: same 8 NextBlock tip wallets as the HTTP path.
        SenderKind::NextblockQuic => nextblock_tip_accounts(),
        // bloXroute: mandatory in-tx tip (>= 1_000_000 lamports) to one of 17 wallets.
        SenderKind::Bloxroute => bloxroute_tip_accounts(),
        // bloXroute QUIC: same 17 bloXroute tip wallets as the HTTP path.
        SenderKind::BloxrouteQuic => bloxroute_tip_accounts(),
        // Syncro: mandatory tip (P2P service fee) to one of 9 P2P tip accounts.
        SenderKind::Syncro => syncro_tip_accounts(),
        // 0slot: mandatory tip (>= 0.001 SOL) to one of 5 0slot tip accounts.
        SenderKind::ZeroSlot => zeroslot_tip_accounts(),
        // AllenHark: mandatory tip (>= 0.001 SOL) to one of 11 hark… wallets.
        SenderKind::AllenHark => allenhark_tip_accounts(),
        // BlockRazor: mandatory in-tx tip (>= 100_000 lamports) to a BlockRazor tip account.
        SenderKind::BlockRazor => blockrazor_tip_accounts(),
    }
}

/// Round-robin rotator over a tip account list. Single-threaded merger
/// safety + lock-free read make it cheap on the hot path (a single
/// `fetch_add` + modulo).
pub struct TipAccountRotator {
    accounts: Vec<Pubkey>,
    cursor: AtomicUsize,
}

impl TipAccountRotator {
    pub fn new(accounts: Vec<Pubkey>) -> Self {
        Self {
            accounts,
            cursor: AtomicUsize::new(0),
        }
    }

    /// Returns the next account in rotation. `None` if the list is empty.
    pub fn next(&self) -> Option<Pubkey> {
        if self.accounts.is_empty() {
            return None;
        }
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed) % self.accounts.len();
        Some(self.accounts[idx])
    }

    pub fn len(&self) -> usize {
        self.accounts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helius_list_loads_and_parses() {
        let list = helius_tip_accounts();
        assert!(list.len() >= 10);
        // All distinct.
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), list.len());
    }

    #[test]
    fn rotator_cycles_in_order() {
        let r = TipAccountRotator::new(helius_tip_accounts().to_vec());
        let a = r.next().unwrap();
        let b = r.next().unwrap();
        assert_ne!(a, b);
        // After len() calls we wrap back to start.
        for _ in 0..(r.len() - 2) {
            r.next();
        }
        let wrapped = r.next().unwrap();
        assert_eq!(wrapped, a);
    }

    #[test]
    fn empty_rotator_returns_none() {
        let r = TipAccountRotator::new(vec![]);
        assert!(r.next().is_none());
    }

    #[test]
    fn jito_list_loads_and_parses() {
        let list = jito_tip_accounts();
        assert_eq!(list.len(), 8);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 8, "jito tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_jito_returns_jito_list() {
        let helius = tip_accounts_for(SenderKind::Helius);
        let jito = tip_accounts_for(SenderKind::Jito);
        assert_eq!(jito, jito_tip_accounts());
        // Helius and Jito sets must be disjoint.
        for h in helius {
            assert!(!jito.contains(h), "tip account {} present in both lists", h);
        }
    }

    #[test]
    fn tip_accounts_for_triton_is_empty() {
        assert!(tip_accounts_for(SenderKind::Triton).is_empty());
    }

    #[test]
    fn tip_accounts_for_triton_sendtx_is_empty() {
        assert!(tip_accounts_for(SenderKind::TritonSendTx).is_empty());
    }

    #[test]
    fn tip_accounts_for_nozomi_has_seventeen_distinct() {
        let list = tip_accounts_for(SenderKind::Nozomi);
        assert_eq!(list.len(), 17);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 17, "nozomi tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_astralane_has_seventeen_distinct() {
        let list = tip_accounts_for(SenderKind::Astralane);
        assert_eq!(list.len(), 17);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 17, "astralane tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_nextblock_has_eight_distinct() {
        let list = tip_accounts_for(SenderKind::Nextblock);
        assert_eq!(list.len(), 8);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 8, "nextblock tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_nextblock_quic_matches_nextblock() {
        assert_eq!(
            tip_accounts_for(SenderKind::NextblockQuic),
            tip_accounts_for(SenderKind::Nextblock)
        );
    }

    #[test]
    fn tip_accounts_for_bloxroute_has_seventeen_distinct() {
        let list = tip_accounts_for(SenderKind::Bloxroute);
        assert_eq!(list.len(), 17);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 17, "bloxroute tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_bloxroute_quic_matches_bloxroute() {
        assert_eq!(
            tip_accounts_for(SenderKind::BloxrouteQuic),
            tip_accounts_for(SenderKind::Bloxroute)
        );
    }

    #[test]
    fn tip_accounts_for_astralane_quic_matches_astralane() {
        assert_eq!(
            tip_accounts_for(SenderKind::AstralaneQuic),
            tip_accounts_for(SenderKind::Astralane)
        );
    }

    #[test]
    fn tip_accounts_for_syncro_has_nine_distinct() {
        let list = tip_accounts_for(SenderKind::Syncro);
        assert_eq!(list.len(), 9);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 9, "syncro tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_zeroslot_has_five_distinct() {
        let list = tip_accounts_for(SenderKind::ZeroSlot);
        assert_eq!(list.len(), 5);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "0slot tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_allenhark_has_eleven_distinct() {
        let list = tip_accounts_for(SenderKind::AllenHark);
        assert_eq!(list.len(), 11);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 11, "allenhark tip accounts must be distinct");
    }

    #[test]
    fn tip_accounts_for_blockrazor_distinct() {
        let list = tip_accounts_for(SenderKind::BlockRazor);
        assert_eq!(list.len(), 14);
        let mut sorted = list.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), list.len(), "blockrazor tip accounts must be distinct");
    }
}
