"""A minimal double-entry-style ledger for cash application.

The ledger is the single source of truth for accounting correctness. It
tracks outstanding invoice balances and the remaining unapplied balance of a
payment, and it validates every allocation before accepting it.

Invariants the ledger guarantees:
  * An invoice can never be over-allocated (balance never goes negative).
  * A payment can never be over-applied (unapplied balance never negative).
  * The same allocation cannot be applied twice (duplicate detection).
  * Conservation: applied + unapplied == original payment amount.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Allocation, Invoice, Payment


class AllocationError(ValueError):
    """Raised when an allocation would violate an accounting invariant."""


@dataclass
class AllocationResult:
    accepted: bool
    reason: str = "ok"


class Ledger:
    """Tracks balances for one payment against a set of invoices."""

    def __init__(self, payment: Payment, invoices: list[Invoice]):
        self.payment = payment
        self.invoices = {inv.invoice_id: inv for inv in invoices}
        # Outstanding balance per invoice (integer cents).
        self._invoice_balance: dict[str, int] = {
            inv.invoice_id: inv.amount_cents for inv in invoices
        }
        self._unapplied_cents: int = payment.amount_cents
        self._allocations: list[Allocation] = []

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------
    @property
    def unapplied_cents(self) -> int:
        return self._unapplied_cents

    @property
    def applied_cents(self) -> int:
        return self.payment.amount_cents - self._unapplied_cents

    @property
    def allocations(self) -> list[Allocation]:
        return list(self._allocations)

    def invoice_balance(self, invoice_id: str) -> int:
        return self._invoice_balance[invoice_id]

    def all_invoice_balances(self) -> dict[str, int]:
        return dict(self._invoice_balance)

    def is_invoice_closed(self, invoice_id: str) -> bool:
        return self._invoice_balance[invoice_id] == 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, allocation: Allocation) -> AllocationResult:
        """Check an allocation without mutating ledger state."""
        if allocation.payment_id != self.payment.payment_id:
            return AllocationResult(False, "payment_id mismatch")
        if allocation.invoice_id not in self._invoice_balance:
            return AllocationResult(False, "unknown invoice")
        if allocation.amount_cents <= 0:
            return AllocationResult(False, "non-positive amount")
        if self._is_duplicate(allocation):
            return AllocationResult(False, "duplicate allocation")
        if allocation.amount_cents > self._unapplied_cents:
            return AllocationResult(False, "exceeds unapplied payment balance")
        if allocation.amount_cents > self._invoice_balance[allocation.invoice_id]:
            return AllocationResult(False, "exceeds invoice outstanding balance")
        return AllocationResult(True, "ok")

    def _is_duplicate(self, allocation: Allocation) -> bool:
        return any(
            a.payment_id == allocation.payment_id
            and a.invoice_id == allocation.invoice_id
            and a.amount_cents == allocation.amount_cents
            for a in self._allocations
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def apply(self, allocation: Allocation) -> AllocationResult:
        """Apply an allocation if valid; return the result.

        Invalid allocations are rejected and leave the ledger unchanged.
        """
        result = self.validate(allocation)
        if not result.accepted:
            return result
        self._invoice_balance[allocation.invoice_id] -= allocation.amount_cents
        self._unapplied_cents -= allocation.amount_cents
        self._allocations.append(allocation)
        return result

    def apply_batch(self, allocations: list[Allocation]) -> AllocationResult:
        """Apply several allocations atomically.

        If any allocation is invalid, none are applied and the first failing
        reason is returned.
        """
        snapshot_balances = dict(self._invoice_balance)
        snapshot_unapplied = self._unapplied_cents
        snapshot_allocs = list(self._allocations)
        for alloc in allocations:
            result = self.apply(alloc)
            if not result.accepted:
                # Roll back.
                self._invoice_balance = snapshot_balances
                self._unapplied_cents = snapshot_unapplied
                self._allocations = snapshot_allocs
                return result
        return AllocationResult(True, "ok")

    # ------------------------------------------------------------------
    # Accounting checks
    # ------------------------------------------------------------------
    def check_conservation(self) -> bool:
        """applied + unapplied must equal the original payment amount."""
        return self.applied_cents + self._unapplied_cents == self.payment.amount_cents

    def check_no_negative_balances(self) -> bool:
        return self._unapplied_cents >= 0 and all(
            bal >= 0 for bal in self._invoice_balance.values()
        )

    def accounting_ok(self) -> bool:
        return self.check_conservation() and self.check_no_negative_balances()
