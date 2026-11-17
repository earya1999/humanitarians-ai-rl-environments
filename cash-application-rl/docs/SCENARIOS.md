# Scenarios

Eight small, reproducible, **fictional** cash-application cases. Every case is
a different incoming payment against **one shared list of ten open invoices**
for a single customer, so the only thing that changes from case to case is the
payment (and whether a remittance is attached). Each case is built fresh on
every `reset`, so scenarios are reproducible and independent, and each carries a
hidden ground-truth resolution used only for evaluation (never exposed in
observations). All money is integer cents internally; shown here in dollars.

## Shared accounts receivable (AR)

Customer: **Northwind Trading Co.** (`CUST-100`). All ten invoices are open.
Every payment below is dated **2026-11-05**, so all ten invoices fall inside
the AR matcher's ±7-day window and are eligible for auto-matching.

| Invoice | Amount | Invoice date |
|---------|--------|--------------|
| INV-001 | $1,000 | 2026-11-03 |
| INV-002 | $2,000 | 2026-11-04 |
| INV-003 | $3,000 | 2026-11-02 |
| INV-004 | $1,500 | 2026-11-03 |
| INV-005 | $2,500 | 2026-11-02 |
| INV-006 | $4,000 | 2026-11-03 |
| INV-007 | $500   | 2026-11-05 |
| INV-008 | $1,000 | 2026-11-06 |
| INV-009 | $2,000 | 2026-11-07 |
| INV-000 | $4,500 | 2026-11-10 |

## CASE-1 — Exact Match
- Payment: PAY-001, **$3,000**
- Remittance: references INV-003, reliable, **visible**
- Expected: allocate $3,000 to INV-003 and close it.
- Ground truth: `INV-003 += $3,000`.

## CASE-2 — Partial Payment
- Payment: PAY-002, **$1,200**
- Remittance: references INV-002 ($2,000), reliable, **visible**
- Expected: allocate $1,200, leaving $800 outstanding.
- Ground truth: `INV-002 += $1,200` (balance $800).

## CASE-3 — Consolidated Payment
- Payment: PAY-003, **$3,500**
- Remittance: references INV-004 + INV-002, reliable, **visible**
- Expected: split across both invoices using remittance evidence.
- Ground truth: `INV-004 += $1,500`, `INV-002 += $2,000`.

## CASE-4 — Ambiguous Payment (default demo)
- Payment: PAY-004, **$2,500**
- Remittance: references INV-001 + INV-004, reliable, **HIDDEN**
- Candidates: (a) INV-001 ($1,000) + INV-004 ($1,500), (b) INV-005 ($2,500) —
  both total $2,500, so the total alone is ambiguous.
- Expected: **investigate** to reveal the remittance, then allocate the
  supported split.
- Ground truth: `INV-001 += $1,000`, `INV-004 += $1,500`.

This case is the default demo because it is the clearest illustration of why
the agent needs sequential decision-making: it cannot resolve the payment
correctly from the payment amount alone.

## CASE-5 — Missing AR
- Payment: PAY-005, **$2,750**
- Remittance: no references, **unreliable**, hidden.
- Every open invoice is a multiple of $500, so no invoice or combination
  reconciles to $2,750 — the paying invoice is **not in the AR**.
- Expected: investigate, search the AR, find nothing, and **escalate**
  (resolution category **Missing AR**).
- Ground truth: none (no determinable correct allocation); `expects_unapplied`.

## CASE-6 — AR Auto-Match (missing remittance, resolvable from AR)
- Payment: PAY-006, **$14,000**
- Remittance: none / unreliable, hidden.
- Exactly **one** combination in the open AR sums to $14,000:
  INV-003 ($3,000) + INV-005 ($2,500) + INV-006 ($4,000) + INV-000 ($4,500).
- Expected: investigate (no remittance), **search the AR**, find the single
  reconciling combination, and allocate it.
- Ground truth: `INV-003 += $3,000`, `INV-005 += $2,500`, `INV-006 += $4,000`,
  `INV-000 += $4,500`.

## CASE-7 — AR Ambiguous (missing remittance, not resolvable)
- Payment: PAY-007, **$1,000**
- Remittance: none / unreliable, hidden.
- **Two** open invoices are each exactly $1,000 (INV-001 and INV-008), so there
  are two equally valid matches.
- Expected: investigate, search the AR, see **more than one** valid match, and
  **escalate** rather than guess.
- Ground truth: none (ambiguous); `expects_unapplied`.

## CASE-8 — Missing AR (remittance present, AR not loaded)
- Payment: PAY-008, **$2,000**
- Remittance: references **INV-950**, reliable, hidden until investigated.
- INV-950 is **not in the loaded AR** — we know (from the remittance) which
  invoice the customer intended to pay, but it is not among the open invoices
  we have (the AR for it may simply not have been loaded). Two loaded invoices
  are also $2,000 (INV-002, INV-009), so an amount-only match would *misapply*
  the cash; the remittance tells us not to.
- Expected: investigate to read the remittance, find the named invoice is
  absent, and **escalate** rather than guess (resolution category
  **Missing AR**).
- Ground truth: none (cannot be determined from the loaded AR);
  `expects_unapplied`.

This is the second flavour of "Missing AR": CASE-5 has *no* usable remittance
and nothing in the AR reconciles, whereas CASE-8 *does* have a clear remittance
but the named invoice is not loaded.

### Resolution categories
Beyond the correct/incorrect outcome, each case is tagged with a resolution
category shown in the demos: `remittance_match` (CASE-1–4), `ar_auto_match`
(CASE-6), `ambiguous_ar` (CASE-7), `missing_ar` (CASE-5 and CASE-8),
`no_evidence`, or `misapplied`.
