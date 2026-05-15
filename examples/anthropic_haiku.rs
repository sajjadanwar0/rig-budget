//! Worked example: threading a `Budget` through a mock LLM call.
//!
//! Demonstrates the `with_budget` API end-to-end without depending on any
//! specific LLM client. To run against a real provider (rig, reqwest, etc.),
//! swap out the `mock_provider_call` function — the wrapper-level code is
//! identical regardless of which provider you use.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example anthropic_haiku
//! ```
//!
//! No API key needed; this example uses a deterministic mock that simulates
//! a claude-haiku-4-5 call returning ~80 input tokens and ~40 output tokens.
//!
//! ## To use with real rig
//!
//! Add `rig-core` to `[dev-dependencies]` in `Cargo.toml`, then replace the
//! body of `mock_provider_call` below with:
//!
//! ```ignore
//! use rig::{completion::Prompt, providers::anthropic};
//!
//! let client = anthropic::Client::new(&api_key);
//! let agent = client.agent("claude-haiku-4-5-20251001").build();
//! let response_text = agent.prompt(prompt).await?;
//!
//! // Production: parse usage from rig's CompletionResponse instead of
//! // approximating from byte length.
//! let approx_input  = prompt.len() as u64;
//! let approx_output = response_text.len() as u64;
//! Ok((response_text, approx_input, approx_output))
//! ```
//!
//! See `tests/integration.rs` for the test-suite version of this API and
//! `token-budgets-experiments/refund-live/src/bin/refund-live-1000.rs` for
//! the production reqwest-based pattern.
//!
//! ## What this example demonstrates
//!
//! - The affine API consumes the `Budget` by value and returns it post-spend.
//! - The receipt/refund cycle correctly reconciles reserved vs actual usage.
//! - `BudgetedError` variants signal where in the pipeline a problem occurred.

use anyhow::Result;
use rig_budget::{with_budget, BudgetedError, ProviderPricing};
use token_budgets::{Budget, ByteLength};

/// Compile-time cap: 1B nano-cents = $10.
///
/// `Budget<MAX>` is a financial cap in nano-cents (1 nc = $0.00000001). The
/// cap is a u64 const-generic so it's enforced at compile time; A2
/// (overflow-safety on spend) is mechanically proven in
/// `token-budgets-formals/verus/` for any MAX <= u64::MAX.
const CAP: u64 = 1_000_000_000;

/// Stand-in for a real LLM call. In production, this is where you'd call
/// rig, reqwest-against-Anthropic, or any other provider.
///
/// Returns: (response_text, prompt_tokens, completion_tokens)
async fn mock_provider_call(prompt: &str) -> Result<(String, u64, u64), String> {
    // Pretend network roundtrip.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Simulate Claude Haiku response for our test prompt.
    let response = format!(
        "Affine types allow each binding to be used AT MOST once, while linear \
         types require AT LEAST one use — affine relaxes the obligation of \
         consumption while keeping the no-duplication guarantee. ({} chars in)",
        prompt.len()
    );

    // Pretend the provider reported these usage figures. Real code would
    // parse the API response's `usage` field. For demonstration, we use
    // byte-length-divided-by-4 as a rough BPE approximation.
    let input_tokens = (prompt.len() as u64 + 3) / 4;
    let output_tokens = (response.len() as u64 + 3) / 4;

    Ok((response, input_tokens, output_tokens))
}

#[tokio::main]
async fn main() -> Result<()> {
    // ─── 1. Initialize the budget ────────────────────────────────────────
    let budget = Budget::<CAP>::new(CAP).expect("CAP fits in u64");
    println!(
        "[budget] Initialized: {} nano-cents (${:.2})",
        budget.micro_cents(),
        budget.micro_cents() as f64 / 100_000_000.0
    );

    // ─── 2. Prepare the call ─────────────────────────────────────────────
    let prompt = "In one sentence: what is the difference between affine and \
                  linear types in programming language theory?";

    // Pricing for claude-haiku-4-5: $1.00/M input, $5.00/M output
    let pricing = ProviderPricing::CLAUDE_HAIKU_4_5;
    let max_output_tokens: u32 = 256;

    // Pre-compute the reservation that with_budget will compute internally
    // (informational only — this is what gets reserved before the call).
    let est_input_tokens = prompt.len() as u64; // ByteLength estimate
    let reservation_nc = est_input_tokens * pricing.input_nc_per_token
        + (max_output_tokens as u64) * pricing.output_nc_per_token;
    println!(
        "[budget] Will reserve ~{} nano-cents (≈${:.6}):",
        reservation_nc,
        reservation_nc as f64 / 100_000_000.0
    );
    println!(
        "  - input:  {} bytes × {} nc/tok = {} nc",
        est_input_tokens,
        pricing.input_nc_per_token,
        est_input_tokens * pricing.input_nc_per_token
    );
    println!(
        "  - output: {} max_tokens × {} nc/tok = {} nc",
        max_output_tokens,
        pricing.output_nc_per_token,
        (max_output_tokens as u64) * pricing.output_nc_per_token
    );

    // ─── 3. Thread the budget through the provider call ────────────────
    let result = with_budget(
        budget,
        pricing,
        &ByteLength,
        prompt,
        max_output_tokens,
        || async { mock_provider_call(prompt).await },
    )
        .await;

    // ─── 4. Inspect the result ───────────────────────────────────────────
    match result {
        Ok((returned_budget, response)) => {
            let spent = CAP - returned_budget.micro_cents();
            println!("\n[response] {}", response);
            println!(
                "\n[budget] Remaining: {} nano-cents (≈${:.6})",
                returned_budget.micro_cents(),
                returned_budget.micro_cents() as f64 / 100_000_000.0
            );
            println!(
                "[budget] Spent on this call: {} nano-cents (≈${:.6})",
                spent,
                spent as f64 / 100_000_000.0
            );
            println!(
                "[budget] Reservation slack refunded: {} nano-cents",
                reservation_nc - spent
            );
        }
        Err(BudgetedError::ReservationFailed { required }) => {
            eprintln!(
                "[budget] Reservation rejected: needed {} nano-cents (≈${:.6}) \
                 but budget was lower",
                required,
                required as f64 / 100_000_000.0
            );
        }
        Err(BudgetedError::A1Violation { reservation, actual }) => {
            eprintln!(
                "[A1 VIOLATION] Provider reported {} nano-cents of usage but only {} \
                 were reserved (overshoot ratio: {:.2}x)",
                actual,
                reservation,
                actual as f64 / reservation as f64
            );
            eprintln!("Possible causes:");
            eprintln!("  - Byte-length estimator is unsound for this tokenizer");
            eprintln!("  - Provider's reported usage is incorrect (see PYAI-003)");
            eprintln!("  - max_tokens setting was below actual completion length");
        }
        Err(BudgetedError::CallFailed(msg)) => {
            eprintln!("[provider] Call failed: {}", msg);
            eprintln!(
                "Receipt was forfeited; the reserved amount is NOT refunded back \
                 to the budget."
            );
        }
        Err(BudgetedError::RefundFailed) => {
            eprintln!(
                "[budget] Refund accounting failed — this is an internal-consistency \
                 violation and should not occur under normal conditions."
            );
        }
    }

    Ok(())
}