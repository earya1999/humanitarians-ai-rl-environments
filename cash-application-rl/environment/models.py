"""Core financial data models for the Cash Application environment.

All monetary values are stored internally as **integer cents** to avoid
floating-point currency errors. Helper functions convert to/from a human
readable dollar string for display only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


def dollars_to_cents(amount: float | int | str) -> int:
    """Convert a dollar amount to integer cents using Decimal rounding.

    Using Decimal here (rather than ``int(amount * 100)``) avoids binary
    float representation errors such as ``19.99 * 100 == 1998.9999...``.
    """
    from decimal import ROUND_HALF_UP, Decimal

    quantized = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(quantized * 100)


def cents_to_str(cents: int) -> str:
    """Render integer cents as a ``$1,234.56`` style string for display."""
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


@dataclass(frozen=True)
class Customer:
    """A customer / payer in the accounts-receivable system."""

    customer_id: str
    name: str


@dataclass
class Invoice:
    """An open receivable owed by a customer.

    ``amount_cents`` is the original invoice total. The *current* outstanding
    balance is tracked by the :class:`~environment.ledger.Ledger`, not here,
    so that a single invoice object can be reused across scenarios.
    """

    invoice_id: str
    customer_id: str
    amount_cents: int
    # Optional invoice date, used by AR auto-matching (date-proximity filter).
    invoice_date: date | None = None


@dataclass
class Payment:
    """An incoming customer payment that must be applied to invoices."""

    payment_id: str
    customer_id: str
    amount_cents: int
    # Optional payment date, used by AR auto-matching (date-proximity filter).
    payment_date: date | None = None


@dataclass
class Remittance:
    """Supporting evidence describing which invoices a payment intends to pay.

    In some scenarios the remittance is initially *hidden* and only becomes
    visible to the agent after an explicit ``investigate`` action.
    """

    payment_id: str
    referenced_invoice_ids: list[str] = field(default_factory=list)
    reliable: bool = True
    hidden: bool = False


@dataclass(frozen=True)
class Allocation:
    """A single application of payment funds to one invoice."""

    payment_id: str
    invoice_id: str
    amount_cents: int


@dataclass
class ReconciliationCase:
    """A complete synthetic scenario the agent must resolve.

    ``ground_truth_allocations`` is the hidden correct resolution used only
    for *evaluation*. It is never exposed through environment observations.
    ``expects_unapplied`` marks cases where the correct behaviour is to leave
    the cash unapplied / escalate rather than allocate.
    """

    case_id: str
    title: str
    customer: Customer
    invoices: list[Invoice]
    payment: Payment
    remittance: Remittance
    candidate_allocations: list[list[Allocation]]
    ground_truth_allocations: list[Allocation] | None = None
    expects_unapplied: bool = False
    description: str = ""
