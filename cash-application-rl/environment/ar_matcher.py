"""AR (Accounts Receivable) auto-matching for missing-remittance cases.

When a payment arrives without reliable remittance advice, a human analyst
does not give up immediately: they look at the customer's *open AR* and try
to find the invoice (or combination of invoices) that the payment most likely
pays off. This module encodes a deliberately simple, transparent version of
that heuristic for research use.

Matching logic (all amounts are integer cents):
  1. Filter the open AR to the *paying customer* only.
  2. Keep invoices dated within +/- ``window_days`` of the payment date.
     (Invoices with no date are ignored by the matcher.)
  3. Find every combination of up to ``max_combo`` invoices whose outstanding
     balances sum to the payment amount (within ``tolerance_cents``).
  4. Return those combinations, sorted by date proximity to the payment
     (closest first) as a tie-breaker.

IMPORTANT: amount reconciliation is the primary signal; customer and date are
filters, and proximity is only a tie-breaker. A single, unique combination is
a confident match. Zero or multiple combinations mean the evidence is
*ambiguous* -- the safe action is then to escalate / leave unapplied rather
than guess. The matcher only surfaces candidates; it does not decide.
"""

from __future__ import annotations

from itertools import combinations

from .models import Allocation, Invoice, Payment


def eligible_invoices(
    payment: Payment, invoices: list[Invoice], window_days: int = 7
) -> list[Invoice]:
    """AR invoices for this customer within the payment's date window."""
    if payment.payment_date is None:
        return []
    out = []
    for inv in invoices:
        if inv.customer_id != payment.customer_id:
            continue
        if inv.invoice_date is None:
            continue
        if abs((inv.invoice_date - payment.payment_date).days) <= window_days:
            out.append(inv)
    return out


def _proximity(payment: Payment, combo: tuple[Invoice, ...]) -> int:
    """Total absolute day distance of a combination from the payment date."""
    return sum(abs((inv.invoice_date - payment.payment_date).days) for inv in combo)


def find_ar_matches(
    payment: Payment,
    invoices: list[Invoice],
    *,
    window_days: int = 7,
    max_combo: int = 4,
    tolerance_cents: int = 0,
) -> list[list[Allocation]]:
    """Return candidate allocations whose amounts reconcile to the payment.

    Each candidate is a list of :class:`Allocation` objects (one per invoice in
    the combination), applying each invoice's full outstanding balance.
    """
    eligible = eligible_invoices(payment, invoices, window_days)
    matches: list[tuple[int, list[Allocation]]] = []
    seen: set[frozenset[str]] = set()

    for size in range(1, max_combo + 1):
        for combo in combinations(eligible, size):
            total = sum(inv.amount_cents for inv in combo)
            if abs(total - payment.amount_cents) <= tolerance_cents:
                key = frozenset(inv.invoice_id for inv in combo)
                if key in seen:
                    continue
                seen.add(key)
                allocation = [
                    Allocation(payment.payment_id, inv.invoice_id, inv.amount_cents)
                    for inv in combo
                ]
                matches.append((_proximity(payment, combo), allocation))

    # Closest to the payment date first; stable for determinism.
    matches.sort(key=lambda m: (m[0], len(m[1])))
    return [allocation for _, allocation in matches]
