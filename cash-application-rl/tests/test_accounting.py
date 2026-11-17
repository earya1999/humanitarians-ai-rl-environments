"""Accounting-correctness tests for the ledger and scenarios.

Covers (from the milestone test checklist):
  1. Exact matching
  2. Partial payment balance updates
  3. Consolidated allocations
  6. Payment conservation
  7. Prevention of invoice over-allocation
  8. Prevention of duplicate allocations
  9. Reproducible scenario initialization
"""

from __future__ import annotations

from environment.ar_matcher import eligible_invoices, find_ar_matches
from environment.ledger import Ledger
from environment.models import Allocation, dollars_to_cents
from environment.scenarios import SCENARIO_IDS, build_scenario


def _ledger_for(case_id: str):
    case = build_scenario(case_id)
    return case, Ledger(case.payment, case.invoices)


# --- 1. Exact matching ------------------------------------------------------
def test_exact_match_closes_invoice():
    case, ledger = _ledger_for("CASE-1")
    (alloc,) = case.ground_truth_allocations
    result = ledger.apply(alloc)
    assert result.accepted
    assert ledger.invoice_balance("INV-003") == 0
    assert ledger.is_invoice_closed("INV-003")
    assert ledger.unapplied_cents == 0


# --- 2. Partial payment balance updates -------------------------------------
def test_partial_payment_leaves_outstanding_balance():
    case, ledger = _ledger_for("CASE-2")
    (alloc,) = case.ground_truth_allocations
    assert ledger.apply(alloc).accepted
    # $1,200 paid against a $2,000 invoice leaves $800 open.
    assert ledger.invoice_balance("INV-002") == dollars_to_cents(800)
    assert not ledger.is_invoice_closed("INV-002")
    assert ledger.unapplied_cents == 0


# --- 3. Consolidated allocations --------------------------------------------
def test_consolidated_payment_covers_both_invoices():
    case, ledger = _ledger_for("CASE-3")
    result = ledger.apply_batch(case.ground_truth_allocations)
    assert result.accepted
    assert ledger.invoice_balance("INV-004") == 0
    assert ledger.invoice_balance("INV-002") == 0
    assert ledger.unapplied_cents == 0
    assert len(ledger.allocations) == 2


# --- 6. Payment conservation ------------------------------------------------
def test_payment_conservation_holds():
    case, ledger = _ledger_for("CASE-2")
    (alloc,) = case.ground_truth_allocations
    ledger.apply(alloc)
    # applied + unapplied must equal the original payment amount.
    assert ledger.applied_cents + ledger.unapplied_cents == case.payment.amount_cents
    assert ledger.check_conservation()
    assert ledger.accounting_ok()


def test_conservation_holds_before_any_allocation():
    case, ledger = _ledger_for("CASE-4")
    assert ledger.unapplied_cents == case.payment.amount_cents
    assert ledger.applied_cents == 0
    assert ledger.check_conservation()


# --- 7. Prevention of invoice over-allocation -------------------------------
def test_over_allocation_of_invoice_is_rejected():
    case, ledger = _ledger_for("CASE-1")  # $3,000 payment
    # INV-001 is $1,000; attempting $1,500 against it must be rejected.
    bad = Allocation("PAY-001", "INV-001", dollars_to_cents(1500))
    result = ledger.apply(bad)
    assert not result.accepted
    assert "exceeds" in result.reason
    # Ledger unchanged.
    assert ledger.invoice_balance("INV-001") == dollars_to_cents(1000)
    assert ledger.unapplied_cents == case.payment.amount_cents


def test_over_application_of_payment_is_rejected():
    case, ledger = _ledger_for("CASE-2")  # $1,200 payment
    # INV-002 is $2,000, but only $1,200 is available to apply.
    bad = Allocation("PAY-002", "INV-002", dollars_to_cents(1300))
    result = ledger.apply(bad)
    assert not result.accepted
    assert ledger.unapplied_cents == dollars_to_cents(1200)


# --- 8. Prevention of duplicate allocations ---------------------------------
def test_duplicate_allocation_is_rejected():
    case, ledger = _ledger_for("CASE-3")
    alloc = Allocation("PAY-003", "INV-004", dollars_to_cents(1500))
    assert ledger.apply(alloc).accepted
    # Applying the identical allocation again must be rejected as duplicate.
    result = ledger.apply(alloc)
    assert not result.accepted
    assert result.reason == "duplicate allocation"
    assert ledger.invoice_balance("INV-004") == 0


def test_batch_rollback_leaves_ledger_unchanged():
    case, ledger = _ledger_for("CASE-3")
    good = Allocation("PAY-003", "INV-004", dollars_to_cents(1500))
    bad = Allocation("PAY-003", "INV-002", dollars_to_cents(9999))  # too large
    result = ledger.apply_batch([good, bad])
    assert not result.accepted
    # Nothing should have been applied because the batch rolled back.
    assert ledger.invoice_balance("INV-004") == dollars_to_cents(1500)
    assert ledger.invoice_balance("INV-002") == dollars_to_cents(2000)
    assert ledger.unapplied_cents == case.payment.amount_cents


# --- 9. Reproducible scenario initialization --------------------------------
def test_scenarios_are_reproducible_and_independent():
    for case_id in SCENARIO_IDS:
        a = build_scenario(case_id)
        b = build_scenario(case_id)
        assert a.payment.amount_cents == b.payment.amount_cents
        assert [i.invoice_id for i in a.invoices] == [i.invoice_id for i in b.invoices]
        assert [i.amount_cents for i in a.invoices] == [
            i.amount_cents for i in b.invoices
        ]
    # Mutating one instance must not affect a freshly built instance.
    first = build_scenario("CASE-1")
    first.invoices[0].amount_cents = 0
    fresh = build_scenario("CASE-1")
    assert fresh.invoices[0].amount_cents == dollars_to_cents(1000)


def test_all_monetary_values_are_integer_cents():
    for case_id in SCENARIO_IDS:
        case = build_scenario(case_id)
        assert isinstance(case.payment.amount_cents, int)
        for inv in case.invoices:
            assert isinstance(inv.amount_cents, int)


# --- AR matcher correctness -------------------------------------------------
def test_ar_matcher_finds_unique_combination():
    case = build_scenario("CASE-6")
    matches = find_ar_matches(case.payment, case.invoices)
    assert len(matches) == 1
    ids = {a.invoice_id for a in matches[0]}
    assert ids == {"INV-003", "INV-005", "INV-006", "INV-000"}
    # The applied amounts must sum exactly to the payment.
    assert sum(a.amount_cents for a in matches[0]) == case.payment.amount_cents


def test_ar_matcher_reports_ambiguity_with_multiple_combinations():
    case = build_scenario("CASE-7")
    matches = find_ar_matches(case.payment, case.invoices)
    assert len(matches) >= 2
    for combo in matches:
        assert sum(a.amount_cents for a in combo) == case.payment.amount_cents


def test_ar_matcher_excludes_out_of_window_and_other_customers():
    from datetime import date

    from environment.models import Invoice, Payment

    pay = Payment("P1", "C1", dollars_to_cents(1000), date(2026, 11, 5))
    invoices = [
        Invoice("IN-WIN", "C1", dollars_to_cents(1000), date(2026, 11, 6)),
        Invoice("OUT-WIN", "C1", dollars_to_cents(1000), date(2026, 10, 1)),
        Invoice("OTHER-CUST", "C2", dollars_to_cents(1000), date(2026, 11, 5)),
    ]
    eligible = {inv.invoice_id for inv in eligible_invoices(pay, invoices)}
    assert eligible == {"IN-WIN"}
    # Only the in-window, same-customer invoice can be auto-matched.
    matches = find_ar_matches(pay, invoices)
    assert len(matches) == 1
    assert {a.invoice_id for a in matches[0]} == {"IN-WIN"}


def test_ar_matcher_returns_nothing_without_dates():
    from environment.models import Invoice, Payment

    pay = Payment("P1", "C1", dollars_to_cents(1000))  # no payment date
    invoices = [Invoice("I1", "C1", dollars_to_cents(1000))]  # no invoice date
    assert find_ar_matches(pay, invoices) == []


def test_case5_ar_has_no_reconciling_combination():
    # CASE-5 is the "Missing AR" case: the AR is dated and searchable, but no
    # invoice or combination reconciles to the payment amount.
    case = build_scenario("CASE-5")
    assert find_ar_matches(case.payment, case.invoices) == []
