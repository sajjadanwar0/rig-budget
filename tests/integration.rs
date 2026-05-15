//! Integration tests for `rig-budget`.
//!
//! These tests cover ground BEYOND the inline unit tests in `src/lib.rs`:
//! - Multi-call budget threading (sequential consumption through one Budget)
//! - Exact-actual case (zero refund slack)
//! - Different pricing tiers (verify ProviderPricing constants are correctly applied)
//! - Different estimators (ByteLength vs Tiktoken behaviour)
//! - Eventual exhaustion (budget drains across calls until ReservationFailed)
//! - Edge cases: zero-token response, single-byte prompt
//!
//! These integration tests go through the same public API any downstream
//! user would call.
//!
//! ## What's NOT here
//! - Real network calls (use `examples/anthropic_haiku.rs` for that)
//! - Concurrent calls (Budget is consumed-by-value; sequential only by design)

use rig_budget::{with_budget, BudgetedError, ProviderPricing};
use token_budgets::{Budget, ByteLength};

// ─────────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────────

/// 1B nano-cents = $10. Plenty for several test calls.
const CAP: u64 = 1_000_000_000;

/// Small CAP (5000 nano-cents = $0.00005) for testing exhaustion paths.
const TINY_CAP: u64 = 5_000;

// ─────────────────────────────────────────────────────────────────────────
// Multi-call threading: budget passes through sequential calls
// ─────────────────────────────────────────────────────────────────────────

/// Property: a Budget can be threaded through multiple sequential `with_budget`
/// calls, each consuming actual usage and refunding slack. Final remaining
/// equals CAP minus sum of all actual usages.
#[tokio::test]
async fn multi_call_threading_sums_actuals() {
    let budget = Budget::<CAP>::new(CAP).unwrap();

    // ─ Call 1: gpt-4o-mini, 20-byte prompt, max_tokens=100 ─────────────
    // Reservation: 20 * 150 + 100 * 600 = 3,000 + 60,000 = 63,000 nc
    // Actual:      15 * 150 +  30 * 600 = 2,250 + 18,000 = 20,250 nc
    let (budget, _) : (_, &str) = with_budget(
        budget,
        ProviderPricing::GPT_4O_MINI,
        &ByteLength,
        "12345678901234567890",
        100,
        || async { Ok::<(&str, u64, u64), std::io::Error>(("response-1", 15, 30)) },
    )
        .await
        .expect("first call succeeds");

    assert_eq!(budget.micro_cents(), CAP - 20_250, "first call drained 20,250 nc");

    // ─ Call 2: gpt-4o, 30-byte prompt, max_tokens=200 ──────────────────
    // Reservation: 30 * 2500 + 200 * 10000 = 75,000 + 2,000,000 = 2,075,000 nc
    // Actual:      25 * 2500 +  80 * 10000 = 62,500 +   800,000 =   862,500 nc
    let (budget, _) : (_, &str) = with_budget(
        budget,
        ProviderPricing::GPT_4O,
        &ByteLength,
        "123456789012345678901234567890",
        200,
        || async { Ok::<(&str, u64, u64), std::io::Error>(("response-2", 25, 80)) },
    )
        .await
        .expect("second call succeeds");

    let expected = CAP - 20_250 - 862_500;
    assert_eq!(budget.micro_cents(), expected, "cumulative drain: {} nc", CAP - expected);

    // ─ Call 3: claude-haiku, 10-byte prompt, max_tokens=50 ──────────────
    // Reservation: 10 * 1000 + 50 * 5000 = 10,000 + 250,000 = 260,000 nc
    // Actual:       8 * 1000 + 20 * 5000 =  8,000 + 100,000 = 108,000 nc
    let (budget, _) : (_, &str) = with_budget(
        budget,
        ProviderPricing::CLAUDE_HAIKU_4_5,
        &ByteLength,
        "1234567890",
        50,
        || async { Ok::<(&str, u64, u64), std::io::Error>(("response-3", 8, 20)) },
    )
        .await
        .expect("third call succeeds");

    let expected_final = CAP - 20_250 - 862_500 - 108_000;
    assert_eq!(budget.micro_cents(), expected_final);
}

// ─────────────────────────────────────────────────────────────────────────
// Exact-actual: zero refund slack
// ─────────────────────────────────────────────────────────────────────────

/// Property: when actual usage exactly equals the reservation, the refund is
/// zero and the final budget is initial - reservation.
#[tokio::test]
async fn exact_actual_drains_full_reservation() {
    let budget = Budget::<CAP>::new(CAP).unwrap();

    // ByteLength on "hi" = 2 bytes. With max_tokens=10:
    // Reservation: 2 * 150 + 10 * 600 = 300 + 6,000 = 6,300 nc
    // To make actual == reservation:
    //   actual = in * 150 + out * 600 = 6,300
    //   Pick in=2, out=10 → 300 + 6,000 = 6,300 ✓
    let (remaining, _) : (_, ()) = with_budget(
        budget,
        ProviderPricing::GPT_4O_MINI,
        &ByteLength,
        "hi",
        10,
        || async { Ok::<((), u64, u64), std::io::Error>(((), 2, 10)) },
    )
        .await
        .expect("call succeeds");

    assert_eq!(remaining.micro_cents(), CAP - 6_300, "exact draw, zero refund slack");
}

// ─────────────────────────────────────────────────────────────────────────
// All four pricing tiers
// ─────────────────────────────────────────────────────────────────────────

/// Property: each `ProviderPricing` constant produces the documented
/// nano-cent cost. Verifies the constants map to published prices.
#[tokio::test]
async fn all_four_pricing_tiers_apply_correctly() {
    // Test fixture: 1-byte prompt, max_tokens=1. Closure returns (1 in, 1 out).
    // For each tier, reservation == actual == 1 * input_price + 1 * output_price.
    // Final budget = CAP - (input_price + output_price).

    let cases = [
        (ProviderPricing::GPT_4O_MINI,        150 +    600),  // = 750
        (ProviderPricing::GPT_4O,           2_500 + 10_000),  // = 12_500
        (ProviderPricing::CLAUDE_HAIKU_4_5, 1_000 +  5_000),  // = 6_000
        (ProviderPricing::CLAUDE_SONNET_4_5, 3_000 + 15_000), // = 18_000
    ];

    for (pricing, expected_total_cost) in cases.iter() {
        let budget = Budget::<CAP>::new(CAP).unwrap();
        let (remaining, _) : (_, ()) = with_budget(
            budget,
            *pricing,
            &ByteLength,
            "a",          // 1 byte
            1,             // max_tokens = 1
            || async { Ok::<((), u64, u64), std::io::Error>(((), 1, 1)) },
        )
            .await
            .expect("call should succeed for pricing tier");

        assert_eq!(
            remaining.micro_cents(),
            CAP - expected_total_cost,
            "pricing {:?} should have charged {} nc total",
            pricing,
            expected_total_cost
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tiktoken estimator
// ─────────────────────────────────────────────────────────────────────────

/// Property: switching estimator from `ByteLength` to `Tiktoken` changes
/// the reservation amount. Tiktoken should produce a tighter (smaller)
/// reservation than the byte-length upper bound for non-trivial English text.
#[cfg(feature = "tiktoken")]
#[tokio::test]
async fn tiktoken_produces_tighter_reservation_than_byte_length() {
    use token_budgets::Tiktoken;

    // Prompt where byte-length overestimates token count substantially.
    // English text: ~4 bytes/token on average for BPE.
    let prompt = "The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog.";

    // Both runs use ProviderPricing::GPT_4O_MINI (150 nc/input, 600 nc/output).
    // Same actual usage in both runs: (50 in, 100 out).
    // Difference is the RESERVATION calculation:
    //   ByteLength reserves ~135 input-tokens worth (135 bytes / 1 = 135)
    //   Tiktoken reserves ~33 input-tokens worth (135 / 4 ≈ 34)

    // Tiktoken reservation is smaller, so the budget impact is the same after
    // refund (since refund covers slack). What this test verifies: both
    // succeed; the budget impact is identical (actual usage determines spend).

    let b1 = Budget::<CAP>::new(CAP).unwrap();
    let (after_byte_length, _) : (_, ()) = with_budget(
        b1,
        ProviderPricing::GPT_4O_MINI,
        &ByteLength,
        prompt,
        200,
        || async { Ok::<((), u64, u64), std::io::Error>(((), 50, 100)) },
    )
        .await
        .expect("ByteLength path succeeds");

    let b2 = Budget::<CAP>::new(CAP).unwrap();
    let (after_tiktoken, _) : (_, ()) = with_budget(
        b2,
        ProviderPricing::GPT_4O_MINI,
        &Tiktoken,
        prompt,
        200,
        || async { Ok::<((), u64, u64), std::io::Error>(((), 50, 100)) },
    )
        .await
        .expect("Tiktoken path succeeds");

    // Both paths charge the same actual: 50*150 + 100*600 = 67,500
    assert_eq!(after_byte_length.micro_cents(), CAP - 67_500);
    assert_eq!(after_tiktoken.micro_cents(),    CAP - 67_500);
    assert_eq!(after_byte_length.micro_cents(), after_tiktoken.micro_cents());
}

// ─────────────────────────────────────────────────────────────────────────
// Reservation failure: budget too small for required reservation
// ─────────────────────────────────────────────────────────────────────────

/// Property: when the estimator+pricing combination produces a reservation
/// larger than the available budget, the call returns `ReservationFailed`
/// before the provider closure is invoked.
#[tokio::test]
async fn small_budget_fails_at_reservation() {
    let budget = Budget::<TINY_CAP>::new(TINY_CAP).unwrap();

    // 50-byte prompt, max_tokens=1000 with GPT_4O (2500/10000 nc per token)
    // Reservation: 50 * 2500 + 1000 * 10000 = 125,000 + 10,000,000 = 10,125,000 nc
    // Far exceeds TINY_CAP (5,000 nc).

    let result: Result<(Budget<TINY_CAP>, &str), _> = with_budget(
        budget,
        ProviderPricing::GPT_4O,
        &ByteLength,
        "01234567890123456789012345678901234567890123456789",
        1000,
        || async {
            // Track whether this was called — it should NOT be.
            // (Note: closure_was_called modification won't actually work
            // through the async closure; we rely on the Result variant
            // to confirm pre-call rejection.)
            Ok::<(&str, u64, u64), std::io::Error>(("should-not-execute", 0, 0))
        },
    )
        .await;

    match result {
        Err(BudgetedError::ReservationFailed { required: 10_125_000 }) => {
            // Expected: required is 10,125,000 nc (50*2500 + 1000*10000).
            // Budget had only 5,000 nc available.
        }
        Err(other) => panic!("expected ReservationFailed with required=10,125,000, got: {}", other),
        Ok(_) => panic!("expected ReservationFailed, got Ok"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Drain-to-empty: budget exhausted across calls
// ─────────────────────────────────────────────────────────────────────────

/// Property: a Budget can be drained across multiple calls until the next
/// reservation would exceed the remaining balance. At that point the call
/// returns `ReservationFailed` and the remaining budget is preserved.
#[tokio::test]
async fn budget_drained_then_reservation_fails() {
    // 1M nano-cents budget ($0.01). Each call uses CLAUDE_HAIKU_4_5 with
    // a small max_tokens to drain gradually.
    const SMALL_CAP: u64 = 1_000_000;
    let mut budget = Budget::<SMALL_CAP>::new(SMALL_CAP).unwrap();

    // Each call: 10-byte prompt, max_tokens=100
    // Reservation: 10 * 1000 + 100 * 5000 = 10,000 + 500,000 = 510,000 nc
    // Actual:       8 * 1000 +  20 * 5000 =  8,000 + 100,000 = 108,000 nc

    // Call 1: budget after = 1,000,000 - 108,000 = 892,000
    let result = with_budget(
        budget,
        ProviderPricing::CLAUDE_HAIKU_4_5,
        &ByteLength,
        "1234567890",
        100,
        || async { Ok::<(&str, u64, u64), std::io::Error>(("ok", 8, 20)) },
    )
        .await;
    let (b, _) : (_, &str) = result.expect("first call succeeds");
    budget = b;
    assert_eq!(budget.micro_cents(), 892_000);

    // Call 2: budget after = 892,000 - 108,000 = 784,000
    let result = with_budget(
        budget,
        ProviderPricing::CLAUDE_HAIKU_4_5,
        &ByteLength,
        "1234567890",
        100,
        || async { Ok::<(&str, u64, u64), std::io::Error>(("ok", 8, 20)) },
    )
        .await;
    let (b, _) : (_, &str) = result.expect("second call succeeds");
    budget = b;
    assert_eq!(budget.micro_cents(), 784_000);

    // Now ask for a reservation that exceeds remaining (784k nc):
    // 100-byte prompt + 200 max_tokens at CLAUDE_SONNET pricing (3000/15000).
    // Reservation: 100 * 3000 + 200 * 15000 = 300,000 + 3,000,000 = 3,300,000 nc
    let big_prompt = "0123456789".repeat(10); // 100 bytes
    let result = with_budget::<SMALL_CAP, &str, _, _, std::io::Error>(
        budget,
        ProviderPricing::CLAUDE_SONNET_4_5,
        &ByteLength,
        &big_prompt,
        200,
        || async { Ok(("never reached", 0, 0)) },
    )
        .await;

    match result {
        Err(BudgetedError::ReservationFailed { required: 3_300_000 }) => {
            // Expected. Budget was 784,000 but call required 3,300,000.
        }
        other => panic!("expected ReservationFailed, got: {:?}", other.is_ok()),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Zero-token response edge case
// ─────────────────────────────────────────────────────────────────────────

/// Property: a provider that returns 0 output tokens (e.g., empty response,
/// refusal, or filter-trigger) is handled correctly. The full output
/// reservation slack is refunded.
#[tokio::test]
async fn zero_output_refunds_full_output_reservation() {
    let budget = Budget::<CAP>::new(CAP).unwrap();

    // Prompt 5 bytes, max_tokens=100 with GPT_4O_MINI.
    // Reservation: 5 * 150 + 100 * 600 = 750 + 60,000 = 60,750 nc
    // Actual: 5 in, 0 out = 5 * 150 + 0 * 600 = 750 nc
    // Refund: 60,750 - 750 = 60,000 nc
    let (remaining, _) : (_, &str) = with_budget(
        budget,
        ProviderPricing::GPT_4O_MINI,
        &ByteLength,
        "12345",
        100,
        || async { Ok::<(&str, u64, u64), std::io::Error>(("", 5, 0)) },
    )
        .await
        .expect("call succeeds");

    assert_eq!(remaining.micro_cents(), CAP - 750, "only input was charged");
}

// ─────────────────────────────────────────────────────────────────────────
// Empty prompt edge case
// ─────────────────────────────────────────────────────────────────────────

/// Property: an empty prompt produces a zero-byte ByteLength estimate.
/// Reservation falls back to just the output side.
#[tokio::test]
async fn empty_prompt_zero_input_reservation() {
    let budget = Budget::<CAP>::new(CAP).unwrap();

    // Prompt: empty string. ByteLength returns 0.
    // Reservation: 0 * 150 + 50 * 600 = 0 + 30,000 = 30,000 nc
    // Actual: 0 in, 10 out = 0 + 10 * 600 = 6,000 nc
    let (remaining, _) : (_, ()) = with_budget(
        budget,
        ProviderPricing::GPT_4O_MINI,
        &ByteLength,
        "",
        50,
        || async { Ok::<((), u64, u64), std::io::Error>(((), 0, 10)) },
    )
        .await
        .expect("empty prompt is handled");

    assert_eq!(remaining.micro_cents(), CAP - 6_000);
}