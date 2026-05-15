//! rig-budget: a Budget-threading helper that composes with rig
//! (or any other Rust LLM client).
//!
//! Rather than wrapping rig's `CompletionModel` trait (which has
//! provider-specific response types and no uniform usage interface),
//! this crate provides a generic `with_budget` helper. The caller
//! supplies an async closure that performs the actual LLM call and
//! returns the response plus the (prompt_tokens, completion_tokens)
//! tuple reported by the provider. The wrapper handles affine-typed
//! reserve / confirm / refund accounting around the closure.
//!
//! This design demonstrates that the Token Budgets discipline
//! composes with rig without requiring upstream modification to rig
//! itself, while remaining independent of rig's evolving type
//! signatures.

use token_budgets::{Budget, TokenEstimator};
use std::future::Future;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BudgetedError {
    #[error("budget reservation failed: required {required} nano-cents")]
    ReservationFailed { required: u64 },

    #[error("A1 violation: reservation {reservation} < actual {actual}")]
    A1Violation { reservation: u64, actual: u64 },

    #[error("provider call failed: {0}")]
    CallFailed(String),

    #[error("refund accounting failed")]
    RefundFailed,
}

/// Per-call pricing in nano-cents per token.
#[derive(Clone, Copy, Debug)]
pub struct ProviderPricing {
    pub input_nc_per_token: u64,
    pub output_nc_per_token: u64,
}

impl ProviderPricing {
    /// gpt-4o-mini: $0.15 in / $0.60 out per M tokens.
    pub const GPT_4O_MINI: Self = Self {
        input_nc_per_token: 150,
        output_nc_per_token: 600,
    };

    /// gpt-4o: $2.50 in / $10.00 out per M tokens.
    pub const GPT_4O: Self = Self {
        input_nc_per_token: 2500,
        output_nc_per_token: 10000,
    };

    /// claude-haiku-4-5: $1.00 in / $5.00 out per M tokens.
    pub const CLAUDE_HAIKU_4_5: Self = Self {
        input_nc_per_token: 1000,
        output_nc_per_token: 5000,
    };

    /// claude-sonnet-4-5: $3.00 in / $15.00 out per M tokens.
    pub const CLAUDE_SONNET_4_5: Self = Self {
        input_nc_per_token: 3000,
        output_nc_per_token: 15000,
    };
}

/// Thread a Budget through a single LLM completion call.
///
/// The caller supplies a closure that performs the actual LLM call
/// and returns `(response, prompt_tokens, completion_tokens)`. This
/// crate handles:
///   1. Estimating the reservation via `estimator` over `prompt`
///   2. Reserving from `budget` (consumed by value; affine)
///   3. Awaiting the caller's `call` closure
///   4. Confirming the receipt against the reported usage
///   5. Returning the refunded Budget alongside the response
///
/// On reservation failure the Budget is consumed (affine discipline);
/// on A1 violation the receipt is forfeited and the after-reserve
/// balance is lost; on caller error the receipt is forfeited.
pub async fn with_budget<const CAP: u64, R, F, Fut, E>(
    budget: Budget<CAP>,
    pricing: ProviderPricing,
    estimator: &dyn TokenEstimator,
    prompt: &str,
    max_tokens: u32,
    call: F,
) -> Result<(Budget<CAP>, R), BudgetedError>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<(R, u64, u64), E>>,
    E: std::fmt::Display,
{
    // 1. Estimate
    let est_in = estimator.estimate(prompt);
    let reservation = est_in
        .saturating_mul(pricing.input_nc_per_token)
        .saturating_add((max_tokens as u64).saturating_mul(pricing.output_nc_per_token));

    // 2. Reserve (consumes budget by value)
    let (after_reserve, receipt) = budget
        .spend_with_receipt(reservation)
        .map_err(|_| BudgetedError::ReservationFailed { required: reservation })?;

    // 3. Await the caller's closure
    match call().await {
        Ok((response, in_tokens, out_tokens)) => {
            // 4. Compute actual and check A1
            let actual = in_tokens
                .saturating_mul(pricing.input_nc_per_token)
                .saturating_add(out_tokens.saturating_mul(pricing.output_nc_per_token));

            if actual > reservation {
                receipt.forfeit();
                return Err(BudgetedError::A1Violation {
                    reservation,
                    actual,
                });
            }

            // 5. Confirm and refund
            let refund = receipt
                .confirm(actual)
                .map_err(|_| BudgetedError::RefundFailed)?;
            let final_budget = refund
                .apply_to(after_reserve)
                .map_err(|_| BudgetedError::RefundFailed)?;
            Ok((final_budget, response))
        }
        Err(e) => {
            // Provider call failed: forfeit receipt, propagate error
            receipt.forfeit();
            Err(BudgetedError::CallFailed(format!("{}", e)))
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use token_budgets::ByteLength;

    const CAP: u64 = 1_000_000_000;

    #[tokio::test]
    async fn happy_path_refunds_correctly() {
        let budget = Budget::<CAP>::new(CAP).unwrap();
        let pricing = ProviderPricing::GPT_4O_MINI;
        // Estimator returns 50 tokens for our prompt.
        // Reservation = 50 * 150 + 100 * 600 = 7,500 + 60,000 = 67,500 nc
        // Actual will be 30 tokens in + 50 tokens out
        //   = 30 * 150 + 50 * 600 = 4,500 + 30,000 = 34,500 nc
        // Refund = 67,500 - 34,500 = 33,000
        // Final budget = CAP - 34,500 = 999_965_500

        let prompt = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwx"; // 50 chars
        let (remaining, _resp): (_, &str) = with_budget(
            budget,
            pricing,
            &ByteLength,
            prompt,
            100,
            || async {
                Ok::<(&str, u64, u64), std::io::Error>(("mock response", 30, 50))
            },
        )
            .await
            .unwrap();

        assert_eq!(remaining.micro_cents(), CAP - 34_500);
    }

    #[tokio::test]
    async fn over_actual_returns_a1_violation() {
        let budget = Budget::<CAP>::new(CAP).unwrap();
        let pricing = ProviderPricing::GPT_4O_MINI;
        // Reservation: 5 chars * 150 + 10 * 600 = 750 + 6,000 = 6,750
        // Actual: 100 input tokens * 150 + 200 output * 600 = 15,000 + 120,000 = 135,000
        // 135_000 > 6_750 → A1Violation

        let result: Result<(Budget<CAP>, &str), _> = with_budget(
            budget,
            pricing,
            &ByteLength,
            "hello",
            10,
            || async {
                Ok::<(&str, u64, u64), std::io::Error>(("oops", 100, 200))
            },
        )
            .await;

        match result {
            Err(BudgetedError::A1Violation { reservation: 6750, actual: 135_000 }) => {}
            Err(other_err) => panic!("expected A1Violation, got different error: {}", other_err),
            Ok(_) => panic!("expected A1Violation, got Ok"),
        }
    }

    #[tokio::test]
    async fn provider_failure_forfeits_receipt() {
        let budget = Budget::<CAP>::new(CAP).unwrap();
        let pricing = ProviderPricing::GPT_4O_MINI;

        let result: Result<(Budget<CAP>, &str), _> = with_budget(
            budget,
            pricing,
            &ByteLength,
            "test prompt",
            64,
            || async {
                Err::<(&str, u64, u64), _>(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "provider returned 500",
                ))
            },
        )
            .await;

        assert!(matches!(result, Err(BudgetedError::CallFailed(_))));
    }

    #[test]
    fn pricing_constants_sane() {
        assert!(
            ProviderPricing::GPT_4O.input_nc_per_token > ProviderPricing::GPT_4O_MINI.input_nc_per_token
        );
        assert!(
            ProviderPricing::CLAUDE_SONNET_4_5.input_nc_per_token
                > ProviderPricing::CLAUDE_HAIKU_4_5.input_nc_per_token
        );
    }
}