# Environment Design

This document describes the design of `CashApplicationEnv`, a minimal
Gymnasium environment for the initial research milestone. The design is
intentionally small and is expected to evolve.

## Problem framing

Each episode presents a single **reconciliation case**: one incoming
customer payment and a set of open invoices. The agent decides how to
resolve the payment. The key research element is that the evidence needed to
act correctly (the *remittance*) may be hidden, so the agent may have to
**investigate before acting** — a sequential decision under uncertainty.

## Action space

`Discrete(8)`:

| Action | Name | Effect |
|-------:|------|--------|
| 0 | `INVESTIGATE` | Reveal hidden remittance evidence (small cost). |
| 1 | `SEARCH_AR` | Search the open AR for an amount/customer/date match (small cost). |
| 2–5 | `ALLOCATE_CANDIDATE_0..3` | Apply candidate allocation #0..#3. |
| 6 | `LEAVE_UNAPPLIED` | Finish without applying the cash. |
| 7 | `ESCALATE` | Hand the case to a human reviewer (small cost). |

Candidate allocations come from one of two sources:

- **Pre-authored** per scenario (for the remittance-driven cases), including
  both supported and unsupported options so the agent cannot succeed by always
  picking the first candidate.
- **AR matcher** (for missing-remittance cases): after `SEARCH_AR`, the
  candidate list is generated at runtime from the customer's open invoices —
  every combination (up to four invoices) dated within ±7 days of the payment
  whose balances sum to the payment amount. See `environment/ar_matcher.py`.

Choosing a candidate index that does not exist, or one the ledger rejects, is
an **unsupported allocation**: it yields no reward and does not change state.

### AR auto-matching (missing remittance)

A missing remittance does not mean no information. `SEARCH_AR` lets the agent
gather *indirect* evidence from the AR sub-ledger. The matcher uses amount
reconciliation as the primary signal, with customer and a ±7-day date window
as filters and date proximity as a tie-breaker. The key design rule: the agent
should auto-match only when there is **exactly one** reconciling combination;
**zero or multiple** combinations mean the evidence is ambiguous and the safe
action is to escalate / leave unapplied rather than guess.

## Observation space

`Box(low=0, high=1e7, shape=(17,), dtype=float32)`. A fixed-size numeric
summary (money in dollars for scale):

| Index | Field |
|------:|-------|
| 0 | payment amount |
| 1 | remaining unapplied amount |
| 2 | number of open invoices |
| 3 | total outstanding across invoices |
| 4 | investigated flag (0/1) |
| 5 | remittance visible flag (0/1) |
| 6 | number of visible remittance references |
| 7 | remittance reliable flag (0/1, only if visible) |
| 8 | number of candidate allocations currently available |
| 9–11 | per-invoice outstanding balance (first 3 slots; larger ARs summarised below) |
| 12 | steps taken this episode |
| 13 | open AR invoices for the paying customer |
| 14 | AR invoices within the ±7-day date window |
| 15 | AR searched flag (0/1) |
| 16 | number of AR candidate matches found |

**The observation never exposes the hidden ground-truth resolution.** When a
remittance is hidden, indices 5–7 read as not-visible / zero until an
`INVESTIGATE` action reveals it. Index 16 reads 0 until `SEARCH_AR` is used.
For large ARs the per-invoice slots (9–11) show only the first three invoices;
the agent reasons about the AR through the summary fields (13–16) instead.

## Episode termination

- **Terminated** when the agent allocates a valid candidate, leaves the cash
  unapplied, or escalates.
- **Truncated** at `max_steps` (default 8) to prevent infinite loops, e.g.
  repeated investigation.

## Reward function (INITIAL, UNTUNED)

> These values are **initial design choices**. They have **not** been tuned
> or empirically validated. They exist only to provide a learning signal for
> future experiments and will likely change.

| Event | Reward |
|-------|-------:|
| Correct reconciliation (terminal) | +1.00 |
| Incorrect reconciliation (terminal) | −1.00 |
| Investigate (per action) | −0.05 |
| Search AR (per action) | −0.05 |
| Escalate (per action) | −0.10 |
| Unsupported / rejected allocation | 0.00 |

Correctness is judged against the hidden ground truth:

- For determinable cases, an allocation is **correct** only if the applied
  (invoice, amount) multiset exactly matches the ground-truth resolution.
- For the undeterminable case (missing remittance), leaving the cash
  unapplied or escalating is **correct**; any allocation is **incorrect**.
- Escalating a *determinable* case is a safe fallback: it incurs the
  escalation cost but no large penalty (outcome `escalated`).

## Accounting guarantees

All monetary values are integer cents. The `Ledger` enforces:

- invoices cannot be over-allocated (balances never negative),
- a payment cannot be over-applied (unapplied balance never negative),
- duplicate allocations are rejected,
- conservation: `applied + unapplied == original payment`.

Batch allocation is atomic: if any line is invalid, none are applied.
