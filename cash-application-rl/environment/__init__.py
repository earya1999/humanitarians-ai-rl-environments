"""Cash Application RL — research environment package.

A small, reproducible Gymnasium-compatible environment for studying
sequential financial decisions in Accounts Receivable cash application.
"""

from .ar_matcher import eligible_invoices, find_ar_matches
from .cash_application_env import (
    ACTION_ESCALATE,
    ACTION_INVESTIGATE,
    ACTION_LEAVE_UNAPPLIED,
    ACTION_NAMES,
    ACTION_SEARCH_AR,
    CashApplicationEnv,
)
from .ledger import AllocationResult, Ledger
from .models import (
    Allocation,
    Customer,
    Invoice,
    Payment,
    ReconciliationCase,
    Remittance,
    cents_to_str,
    dollars_to_cents,
)
from .rule_based_agent import RuleBasedAgent
from .scenarios import SCENARIO_IDS, all_scenarios, build_scenario

__all__ = [
    "CashApplicationEnv",
    "ACTION_INVESTIGATE",
    "ACTION_SEARCH_AR",
    "ACTION_LEAVE_UNAPPLIED",
    "ACTION_ESCALATE",
    "ACTION_NAMES",
    "find_ar_matches",
    "eligible_invoices",
    "Ledger",
    "AllocationResult",
    "Customer",
    "Invoice",
    "Payment",
    "Remittance",
    "Allocation",
    "ReconciliationCase",
    "dollars_to_cents",
    "cents_to_str",
    "RuleBasedAgent",
    "SCENARIO_IDS",
    "build_scenario",
    "all_scenarios",
]
