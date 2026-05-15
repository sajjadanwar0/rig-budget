# rig-budget

> Integration helper threading the [`token-budgets`](https://github.com/sajjadanwar0/token-budgets)
> affine-resource discipline through the [rig](https://github.com/0xPlaygrounds/rig)
> Rust LLM framework — or any other Rust LLM client.

[![Status](https://img.shields.io/badge/status-experimental-yellow)](#status)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue)](LICENSE-MIT)

This crate is **not** a wrapper around rig's `CompletionModel` trait.
It exposes a single generic helper, `with_budget`, that the caller
parameterizes with their own provider-call closure. This design
choice keeps `rig-budget` independent of rig's evolving type
signatures and makes the helper composable with any LLM client.

## What it provides

The `with_budget` function threads a `Budget<CAP>` through a single
provider call:

```rust
use rig_budget::{with_budget, ProviderPricing};
use token_budgets::{Budget, ByteLength};

const CAP: u64 = 1_000_000_000;  // $10 cap (in nano-cents)

let budget = Budget::<CAP>::new(CAP)?;
let pricing = ProviderPricing::CLAUDE_HAIKU_4_5;
let prompt = "What is the capital of France?";
let max_output_tokens: u32 = 256;

let (returned_budget, response) = with_budget(
budget,
pricing,
&ByteLength,
prompt,
max_output_tokens,
|| async {
// Your provider call here — rig, reqwest, whatever.
// Must return (response, input_tokens, output_tokens).
Ok::<(String, u64, u64), String>(("Paris.".to_string(), 8, 2))
},
).await?;

println!("Response: {}", response);
println!("Remaining: {} nano-cents", returned_budget.micro_cents());
```

The helper:

1. **Estimates** the input-token count via the supplied
   `&dyn TokenEstimator`.
2. **Computes** a sound reservation = `input_estimate * pricing.input_nc_per_token + max_tokens * pricing.output_nc_per_token`.
3. **Reserves** from the budget (consumes by value; affine
   discipline).
4. **Awaits** the caller's closure.
5. **Verifies A1**: that actual ≤ reserved. If violated, the receipt
   is forfeited and `BudgetedError::A1Violation` is returned.
6. **Confirms** the receipt with actual usage and returns the
   refunded budget alongside the response.

## API

### `with_budget`

```rust
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
```

The closure `F` returns `(response, input_tokens, output_tokens)`.
The wrapper handles all budget bookkeeping.

### `ProviderPricing`

Pre-configured pricing constants (per-token rates in nano-cents):

| Constant | Input nc/tok | Output nc/tok | Equivalent $/M tokens |
|---|---:|---:|---|
| `GPT_4O_MINI` | 150 | 600 | $0.15 in / $0.60 out |
| `GPT_4O` | 2,500 | 10,000 | $2.50 in / $10.00 out |
| `CLAUDE_HAIKU_4_5` | 1,000 | 5,000 | $1.00 in / $5.00 out |
| `CLAUDE_SONNET_4_5` | 3,000 | 15,000 | $3.00 in / $15.00 out |

For other models, construct directly:

```rust
let custom = ProviderPricing {
    input_nc_per_token: 750,
    output_nc_per_token: 2000,
};
```

### `BudgetedError`

```rust
pub enum BudgetedError {
    /// The estimator+pricing combination required a reservation
    /// larger than the available budget.
    ReservationFailed { required: u64 },

    /// Actual provider usage exceeded the reservation. The estimator
    /// returned an unsound upper bound, the provider misreported, or
    /// max_tokens was set too low. The receipt is forfeited; the
    /// post-reserve balance is lost.
    A1Violation { reservation: u64, actual: u64 },

    /// The closure returned Err. The receipt is forfeited.
    CallFailed(String),

    /// Refund accounting inconsistency. Should not occur under
    /// normal operation; if it does, file an issue.
    RefundFailed,
}
```

## Why a closure-based helper, not a wrapper struct

The rig ecosystem ships provider-specific response types
(`AnthropicCompletionResponse`, `OpenAiCompletionResponse`, etc.)
that don't share a uniform "where to find token usage" interface.
Wrapping each provider would require:

1. A per-provider extractor function.
2. Tracking rig's API changes across versions.
3. Conditional compilation for each provider's feature flag.

The closure-based design pushes all of that complexity to the
caller, who knows their specific provider's response shape. The
helper handles the budget arithmetic and the A1 check.

This makes `rig-budget` a thin layer (~150 lines of public surface)
that composes cleanly with whatever LLM client the caller uses.

## Examples

### `examples/anthropic_haiku.rs`

A self-contained example using a mock provider closure. Compiles
with no extra dependencies and demonstrates the API end-to-end.

```bash
cargo run --example anthropic_haiku
```

To adapt for real rig: add `rig-core` to `[dev-dependencies]` and
replace the mock closure with a rig prompt call. See the example's
doc-comment for the exact 5-line substitution.

## Tests

```bash
cargo test
```

This runs:
- Inline unit tests in `src/lib.rs` (happy path, A1 violation,
  provider failure, pricing constants)
- Integration tests in `tests/integration.rs`:
    - `multi_call_threading_sums_actuals` — three sequential calls
      summing actuals correctly
    - `exact_actual_drains_full_reservation` — zero refund slack
    - `all_four_pricing_tiers_apply_correctly` — pricing constant
      arithmetic
    - `tiktoken_produces_tighter_reservation_than_byte_length` —
      feature-gated estimator comparison
    - `small_budget_fails_at_reservation` — pre-call rejection
    - `budget_drained_then_reservation_fails` — eventual exhaustion
    - `zero_output_refunds_full_output_reservation` — zero-completion
    - `empty_prompt_zero_input_reservation` — empty-prompt edge case

## Status

| Aspect | State |
|---|---|
| `with_budget` API | ✅ Stable across this version |
| `ProviderPricing` constants | ✅ Current as of model release dates listed |
| Integration tests | ✅ 8 tests covering edge cases |
| Real-rig example | ⚠️ Mock provider only; rig API integration is left to caller |
| crates.io publish | ⏳ Pending arXiv preprint + EMSE submission |

This crate is part of an EMSE submission artifact and may evolve as
review feedback comes in.

## Honest scope

What this crate **does** provide:
- A clean, thin integration point for the Token Budgets discipline.
- Pre-configured pricing for four common models.
- A1 verification at the receipt-confirm step.

What it does **NOT** provide:
- A rig-version-pinned integration. The caller's closure does the
  rig-specific work; this crate doesn't track rig's API.
- Automatic token-count extraction from provider responses. The
  caller must extract token counts and pass them as
  `(input_tokens, output_tokens)`.
- Streaming-response handling. For per-chunk refund during streaming,
  use `Budget::spend_streaming` and `StreamingReceipt` from the main
  [`token-budgets`](https://github.com/sajjadanwar0/token-budgets) crate.
- Pool-aware budgeting. For multi-tenant budget pools, use
  `BudgetPool` from [`token-budgets`](https://github.com/sajjadanwar0/token-budgets).

## Related repositories

| Repository | What it contains |
|---|---|
| [`token-budgets`](https://github.com/sajjadanwar0/token-budgets) | Main affine-API library + 167-entry catalog |
| [`token-budgets-extensions`](https://github.com/sajjadanwar0/token-budgets-extensions) | Adaptive estimator, Verus skeleton |
| [`token-budgets-formals`](https://github.com/sajjadanwar0/token-budgets-formals) | 5-tier mechanization (TLAPS / TLC / Coq / Dafny / Verus) |
| [`token-budgets-experiments`](https://github.com/sajjadanwar0/token-budgets-experiments) | Empirical validation (5,424 live API row-events) |
| [`rig-budget`](https://github.com/sajjadanwar0/rig-budget) | This repo |

## Paper

```bibtex
@article{khan-token-budgets-2026,
  author  = {Khan, Sajjad},
  title   = {Token Budgets: An Affine-Resource Discipline for LLM Cost Caps in Rust},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```

This crate corresponds to §VI (integration patterns) in the paper.

## License

Dual MIT/Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.