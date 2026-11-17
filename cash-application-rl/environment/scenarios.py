"""Seven small, reproducible synthetic Cash Application scenarios.

All data is fictional. Every scenario is a different incoming payment against
one shared list of ten open invoices for a single customer (see ``_AR_INVOICES``).
Each case bundles the customer, open invoices, the
incoming payment, the remittance evidence (which may be initially hidden),
a short list of *candidate* allocations the agent can choose from, and a
hidden ground-truth resolution used only for evaluation.

The candidate allocations deliberately include both correct and incorrect
options so that an agent must reason about evidence rather than blindly pick
the first candidate.
"""

from __future__ import annotations

from datetime import date

from .models import (
    Allocation,
    Customer,
    Invoice,
    Payment,
    ReconciliationCase,
    Remittance,
    dollars_to_cents,
)


# ---------------------------------------------------------------------------
# Shared accounts-receivable (AR) ledger.
#
# Every scenario below is a different incoming payment against this ONE shared
# list of open invoices for a single customer. This keeps the demo easy to
# follow: the open AR never changes, only the payment (and whether a remittance
# is attached) changes from case to case.
#
# All payments are dated 2026-11-05, so every invoice below falls within the
# AR matcher's +/-7 day window and is therefore eligible for auto-matching.
# ---------------------------------------------------------------------------

_AR_CUSTOMER = Customer("CUST-100", "Northwind Trading Co.")

# (invoice_id, amount_dollars, invoice_date)
_AR_INVOICES: list[tuple[str, int, date]] = [
    ("INV-001", 1000, date(2026, 11, 3)),
    ("INV-002", 2000, date(2026, 11, 4)),
    ("INV-003", 3000, date(2026, 11, 2)),
    ("INV-004", 1500, date(2026, 11, 3)),
    ("INV-005", 2500, date(2026, 11, 2)),
    ("INV-006", 4000, date(2026, 11, 3)),
    ("INV-007", 500, date(2026, 11, 5)),
    ("INV-008", 1000, date(2026, 11, 6)),
    ("INV-009", 2000, date(2026, 11, 7)),
    ("INV-000", 4500, date(2026, 11, 10)),
]

_PAYMENT_DATE = date(2026, 11, 5)


def _shared_ar() -> list[Invoice]:
    """Fresh copies of the shared open invoices (one customer's AR)."""
    return [
        Invoice(iid, _AR_CUSTOMER.customer_id, dollars_to_cents(amt), d)
        for iid, amt, d in _AR_INVOICES
    ]


def _alloc(pay: Payment, inv_id: str, dollars: int | float) -> Allocation:
    return Allocation(pay.payment_id, inv_id, dollars_to_cents(dollars))


def _case_1_exact_match() -> ReconciliationCase:
    pay = Payment("PAY-001", _AR_CUSTOMER.customer_id, dollars_to_cents(3000), _PAYMENT_DATE)
    remit = Remittance(pay.payment_id, ["INV-003"], reliable=True, hidden=False)
    truth = [_alloc(pay, "INV-003", 3000)]
    return ReconciliationCase(
        case_id="CASE-1",
        title="Exact Match",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[truth],
        ground_truth_allocations=truth,
        expects_unapplied=False,
        description="Customer pays $3,000 with a remittance pointing at INV-003 ($3,000).",
    )


def _case_2_partial_payment() -> ReconciliationCase:
    pay = Payment("PAY-002", _AR_CUSTOMER.customer_id, dollars_to_cents(1200), _PAYMENT_DATE)
    remit = Remittance(pay.payment_id, ["INV-002"], reliable=True, hidden=False)
    truth = [_alloc(pay, "INV-002", 1200)]
    return ReconciliationCase(
        case_id="CASE-2",
        title="Partial Payment",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[truth],
        ground_truth_allocations=truth,
        expects_unapplied=False,
        description="Customer pays $1,200 against INV-002 ($2,000), leaving $800 open.",
    )


def _case_3_consolidated() -> ReconciliationCase:
    pay = Payment("PAY-003", _AR_CUSTOMER.customer_id, dollars_to_cents(3500), _PAYMENT_DATE)
    remit = Remittance(
        pay.payment_id, ["INV-004", "INV-002"], reliable=True, hidden=False
    )
    truth = [
        _alloc(pay, "INV-004", 1500),
        _alloc(pay, "INV-002", 2000),
    ]
    return ReconciliationCase(
        case_id="CASE-3",
        title="Consolidated Payment",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[truth],
        ground_truth_allocations=truth,
        expects_unapplied=False,
        description=(
            "Customer pays $3,500 covering INV-004 ($1,500) and INV-002 ($2,000) "
            "with supporting remittance."
        ),
    )


def _case_4_ambiguous() -> ReconciliationCase:
    pay = Payment("PAY-004", _AR_CUSTOMER.customer_id, dollars_to_cents(2500), _PAYMENT_DATE)
    # Remittance is initially hidden; investigation reveals INV-001 + INV-004.
    remit = Remittance(
        pay.payment_id, ["INV-001", "INV-004"], reliable=True, hidden=True
    )
    truth = [
        _alloc(pay, "INV-001", 1000),
        _alloc(pay, "INV-004", 1500),
    ]
    # Both candidates total $2,500, but only one matches the hidden remittance:
    # INV-001 ($1,000) + INV-004 ($1,500), versus INV-005 ($2,500) alone.
    candidate_decoy = [_alloc(pay, "INV-005", 2500)]
    return ReconciliationCase(
        case_id="CASE-4",
        title="Ambiguous Payment",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[truth, candidate_decoy],
        ground_truth_allocations=truth,
        expects_unapplied=False,
        description=(
            "Customer pays $2,500; it could be INV-001 + INV-004 or INV-005 alone. "
            "Remittance is hidden until investigated and reveals INV-001 + INV-004."
        ),
    )


def _case_5_missing_ar() -> ReconciliationCase:
    """Missing remittance AND the open AR has no reconciling match.

    The payment is $2,750. Every open invoice is a multiple of $500, so no
    invoice or combination sums to $2,750 -- the paying invoice is simply not
    in the open AR. Correct behaviour is to escalate / leave unapplied.
    """
    pay = Payment("PAY-005", _AR_CUSTOMER.customer_id, dollars_to_cents(2750), _PAYMENT_DATE)
    remit = Remittance(pay.payment_id, [], reliable=False, hidden=True)
    candidates = [
        # A tempting-but-unsupported guess (valid in the ledger, but wrong).
        [_alloc(pay, "INV-005", 2500)],
    ]
    return ReconciliationCase(
        case_id="CASE-5",
        title="Missing AR",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=candidates,
        ground_truth_allocations=None,  # No determinable correct allocation.
        expects_unapplied=True,
        description=(
            "Customer pays $2,750 with no remittance, and no invoice or "
            "combination in the open AR reconciles to it. Correct behaviour is "
            "to escalate / leave unapplied."
        ),
    )


def _case_6_ar_auto_match() -> ReconciliationCase:
    """Missing remittance, but the open AR uniquely reconciles to the payment.

    The payment is $14,000. Exactly one combination within the window sums to
    it: INV-003 ($3,000) + INV-005 ($2,500) + INV-006 ($4,000) + INV-000
    ($4,500). Expected behaviour: search AR, find the single match, allocate.
    """
    pay = Payment("PAY-006", _AR_CUSTOMER.customer_id, dollars_to_cents(14000), _PAYMENT_DATE)
    remit = Remittance(pay.payment_id, [], reliable=False, hidden=True)
    truth = [
        _alloc(pay, "INV-003", 3000),
        _alloc(pay, "INV-005", 2500),
        _alloc(pay, "INV-006", 4000),
        _alloc(pay, "INV-000", 4500),
    ]
    return ReconciliationCase(
        case_id="CASE-6",
        title="AR Auto-Match",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[],  # generated at runtime by the AR matcher
        ground_truth_allocations=truth,
        expects_unapplied=False,
        description=(
            "Customer pays $14,000 with no remittance. The open AR has one unique "
            "combination that reconciles (INV-003 + INV-005 + INV-006 + INV-000)."
        ),
    )


def _case_7_ar_ambiguous() -> ReconciliationCase:
    """Missing remittance and the open AR has MORE THAN ONE reconciling option.

    The payment is $1,000. Two open invoices are each exactly $1,000 (INV-001
    and INV-008), so there are two equally valid matches. Expected behaviour:
    escalate / leave unapplied rather than guess.
    """
    pay = Payment("PAY-007", _AR_CUSTOMER.customer_id, dollars_to_cents(1000), _PAYMENT_DATE)
    remit = Remittance(pay.payment_id, [], reliable=False, hidden=True)
    return ReconciliationCase(
        case_id="CASE-7",
        title="AR Ambiguous",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[],
        ground_truth_allocations=None,  # genuinely undeterminable
        expects_unapplied=True,
        description=(
            "Customer pays $1,000 with no remittance. Two open invoices are each "
            "$1,000 (INV-001 and INV-008), so the match is ambiguous."
        ),
    )


def _case_8_missing_ar_not_loaded() -> ReconciliationCase:
    """Remittance IS present, but the named invoice is not in the loaded AR.

    The customer pays $2,000 and the remittance clearly names INV-950 -- but
    INV-950 is not in our open AR (the AR for that invoice may simply not have
    been loaded yet). This is still a "Missing AR" situation: we know the
    intent but cannot apply the cash. Crucially, two loaded invoices happen to
    be $2,000 as well (INV-002, INV-009), so a naive amount-only match would
    misapply the payment; the remittance tells us not to. Correct behaviour is
    to escalate rather than guess.
    """
    pay = Payment("PAY-008", _AR_CUSTOMER.customer_id, dollars_to_cents(2000), _PAYMENT_DATE)
    # Reliable remittance, hidden until investigated, naming an unloaded invoice.
    remit = Remittance(pay.payment_id, ["INV-950"], reliable=True, hidden=True)
    return ReconciliationCase(
        case_id="CASE-8",
        title="Missing AR (not loaded)",
        customer=_AR_CUSTOMER,
        invoices=_shared_ar(),
        payment=pay,
        remittance=remit,
        candidate_allocations=[],  # the named invoice is not in the AR to apply
        ground_truth_allocations=None,  # cannot be determined from the loaded AR
        expects_unapplied=True,
        description=(
            "Customer pays $2,000 with a remittance naming INV-950, but INV-950 "
            "is not in the loaded AR. Even though two loaded invoices are also "
            "$2,000, the named invoice is absent, so the agent must escalate "
            "rather than misapply."
        ),
    )


# Ordered registry of scenario builders. Builders are used (rather than module
# level constants) so every ``reset`` produces fresh, independent objects.
_BUILDERS = {
    "CASE-1": _case_1_exact_match,
    "CASE-2": _case_2_partial_payment,
    "CASE-3": _case_3_consolidated,
    "CASE-4": _case_4_ambiguous,
    "CASE-5": _case_5_missing_ar,
    "CASE-6": _case_6_ar_auto_match,
    "CASE-7": _case_7_ar_ambiguous,
    "CASE-8": _case_8_missing_ar_not_loaded,
}

SCENARIO_IDS = list(_BUILDERS.keys())


def build_scenario(case_id: str) -> ReconciliationCase:
    """Build a fresh, independent instance of the given scenario."""
    if case_id not in _BUILDERS:
        raise KeyError(f"Unknown scenario {case_id!r}. Known: {SCENARIO_IDS}")
    return _BUILDERS[case_id]()


def all_scenarios() -> list[ReconciliationCase]:
    """Build fresh instances of all scenarios in order."""
    return [build_scenario(cid) for cid in SCENARIO_IDS]


def _greedy_allocation(
    payment: Payment, invoices: list[Invoice], ref_ids: list[str]
) -> list[Allocation]:
    """Greedily apply the payment across the referenced invoices in order.

    Each referenced invoice receives ``min(remaining payment, invoice total)``.
    This reproduces exact, partial, and consolidated behaviour from simple
    remittance references. Any leftover payment remains unapplied.
    """
    by_id = {inv.invoice_id: inv for inv in invoices}
    remaining = payment.amount_cents
    allocations: list[Allocation] = []
    for inv_id in ref_ids:
        if inv_id not in by_id or remaining <= 0:
            continue
        amount = min(remaining, by_id[inv_id].amount_cents)
        if amount > 0:
            allocations.append(Allocation(payment.payment_id, inv_id, amount))
            remaining -= amount
    return allocations


def build_manual_scenario(
    *,
    customer_name: str,
    payment_dollars: float | int | str,
    invoices: list[tuple],
    remittance_refs: list[str],
    reliable: bool = True,
    hidden: bool = True,
    payment_date: date | None = None,
    customer_id: str = "CUST-MANUAL",
    payment_id: str = "PAY-MANUAL",
    case_id: str = "CASE-MANUAL",
    title: str = "Manual Entry",
) -> ReconciliationCase:
    """Build a reconciliation case from values entered at runtime.

    ``invoices`` items are ``(invoice_id, amount)`` or
    ``(invoice_id, amount, invoice_date)``. Any missing invoice date defaults
    to the payment date (so it falls inside the AR date window); the payment
    date itself defaults to today.

    The "correct" resolution is derived from the evidence the user provides:
      * reliable remittance references -> allocate greedily per references;
      * no remittance, but the open AR uniquely reconciles -> that AR match;
      * otherwise -> undeterminable (leave unapplied / escalate).

    This keeps evaluation self-consistent for an ad-hoc live demo without any
    hidden, pre-authored ground truth.
    """
    from .ar_matcher import find_ar_matches

    cust = Customer(customer_id, customer_name)
    pay_date = payment_date or date.today()
    inv_objs = []
    for item in invoices:
        inv_id, amount = item[0], item[1]
        inv_date = item[2] if len(item) > 2 and item[2] is not None else pay_date
        inv_objs.append(
            Invoice(inv_id, cust.customer_id, dollars_to_cents(amount), inv_date)
        )
    pay = Payment(
        payment_id, cust.customer_id, dollars_to_cents(payment_dollars), pay_date
    )

    known_ids = {inv.invoice_id for inv in inv_objs}
    refs = [r for r in remittance_refs if r in known_ids]
    remit = Remittance(pay.payment_id, refs, reliable=reliable, hidden=hidden)

    if reliable and refs:
        # Remittance-driven: allocate per the referenced invoices.
        truth = _greedy_allocation(pay, inv_objs, refs)
        candidates = [truth] if truth else []
        expects_unapplied = False
    else:
        # Missing remittance: fall back to AR auto-matching. A single unique
        # reconciliation is the correct answer; zero or several is ambiguous.
        ar_matches = find_ar_matches(pay, inv_objs)
        if len(ar_matches) == 1:
            truth = ar_matches[0]
            candidates = []  # generated at runtime via SEARCH_AR
            expects_unapplied = False
        else:
            truth = None
            candidates = []
            expects_unapplied = True

    return ReconciliationCase(
        case_id=case_id,
        title=title,
        customer=cust,
        invoices=inv_objs,
        payment=pay,
        remittance=remit,
        candidate_allocations=candidates,
        ground_truth_allocations=truth,
        expects_unapplied=expects_unapplied,
        description="Manually entered at runtime.",
    )
