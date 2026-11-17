"""A minimal Gymnasium environment for Cash Application decision-making.

This is a *first prototype* for research. The agent observes a numeric,
fixed-size summary of an incoming payment and the open invoices for a single
reconciliation case, and chooses among a small discrete set of actions:

    0  INVESTIGATE        reveal hidden remittance evidence (small cost)
    1  ALLOCATE candidate 0   apply predefined candidate allocation #0
    2  ALLOCATE candidate 1   apply predefined candidate allocation #1
    3  LEAVE_UNAPPLIED     finish without applying the cash
    4  ESCALATE            hand the case to a human reviewer (small cost)

The observation never exposes the hidden ground-truth resolution. Hidden
remittance references are only revealed after an INVESTIGATE action.

Reward values below are **initial design choices**. They have NOT been tuned
or empirically validated; they exist so the environment produces a learning
signal for future experiments.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ar_matcher import eligible_invoices, find_ar_matches
from .ledger import Ledger
from .models import Allocation, ReconciliationCase
from .scenarios import SCENARIO_IDS, build_scenario

# --- Action constants -------------------------------------------------------
ACTION_INVESTIGATE = 0
ACTION_SEARCH_AR = 1  # search the open AR for an amount/date/customer match
ACTION_ALLOCATE_BASE = 2  # candidate index == action - ACTION_ALLOCATE_BASE
MAX_CANDIDATES = 4
ACTION_LEAVE_UNAPPLIED = ACTION_ALLOCATE_BASE + MAX_CANDIDATES  # 6
ACTION_ESCALATE = ACTION_LEAVE_UNAPPLIED + 1  # 7
N_ACTIONS = ACTION_ESCALATE + 1  # 8

ACTION_NAMES = {
    ACTION_INVESTIGATE: "INVESTIGATE",
    ACTION_SEARCH_AR: "SEARCH_AR",
    ACTION_ALLOCATE_BASE + 0: "ALLOCATE_CANDIDATE_0",
    ACTION_ALLOCATE_BASE + 1: "ALLOCATE_CANDIDATE_1",
    ACTION_ALLOCATE_BASE + 2: "ALLOCATE_CANDIDATE_2",
    ACTION_ALLOCATE_BASE + 3: "ALLOCATE_CANDIDATE_3",
    ACTION_LEAVE_UNAPPLIED: "LEAVE_UNAPPLIED",
    ACTION_ESCALATE: "ESCALATE",
}

# --- AR matching parameters -------------------------------------------------
AR_WINDOW_DAYS = 7
AR_MAX_COMBO = 4

# --- Observation layout -----------------------------------------------------
MAX_INVOICES = 3  # number of per-invoice balance slots shown in the observation
# [ payment$, unapplied$, num_open_invoices, total_outstanding$,
#   investigated, remittance_visible, remittance_num_refs, remittance_reliable,
#   num_candidates, inv_balance_0$, inv_balance_1$, inv_balance_2$, steps,
#   ar_open_count, ar_in_window_count, ar_searched, ar_num_matches ]
OBS_SIZE = 9 + MAX_INVOICES + 1 + 4

# --- Reward constants (INITIAL, UNTUNED design choices) ---------------------
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -1.0
COST_INVESTIGATE = -0.05
COST_SEARCH_AR = -0.05
COST_ESCALATE = -0.10
REWARD_UNSUPPORTED = 0.0  # unsupported / rejected allocation attempts


class CashApplicationEnv(gym.Env):
    """Single-case Cash Application environment following the Gymnasium API."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        case_id: str | None = None,
        max_steps: int = 8,
        case: ReconciliationCase | None = None,
    ):
        super().__init__()
        self._fixed_case_id = case_id
        # A caller may supply a fully built case (e.g. an interactive demo),
        # bypassing the predefined scenario registry.
        self._provided_case = case
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1e7, shape=(OBS_SIZE,), dtype=np.float32
        )

        # Episode state (populated in reset()).
        self.case: ReconciliationCase | None = None
        self.ledger: Ledger | None = None
        self._investigated: bool = False
        self._ar_searched: bool = False
        self._ar_candidates: list[list[Allocation]] = []
        self._steps: int = 0
        self._done: bool = False
        self._outcome: str = "pending"

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # A hand-built case may be supplied via the constructor or options.
        provided = None
        if options and "case" in options:
            provided = options["case"]
        elif self._provided_case is not None:
            provided = self._provided_case

        if provided is not None:
            # Deep-copy so each reset starts from fresh, independent state.
            self.case = copy.deepcopy(provided)
        else:
            case_id = self._fixed_case_id
            if options and "case_id" in options:
                case_id = options["case_id"]
            if case_id is None:
                # Reproducible random selection via the seeded RNG.
                idx = int(self.np_random.integers(len(SCENARIO_IDS)))
                case_id = SCENARIO_IDS[idx]
            self.case = build_scenario(case_id)

        self.ledger = Ledger(self.case.payment, self.case.invoices)
        self._investigated = False
        self._ar_searched = False
        self._ar_candidates = []
        self._steps = 0
        self._done = False
        self._outcome = "pending"

        return self._build_obs(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset().")

        action = int(action)
        self._steps += 1
        reward = 0.0
        terminated = False
        event = ""

        if action == ACTION_INVESTIGATE:
            reward += COST_INVESTIGATE
            if not self._investigated:
                self._investigated = True
                event = "investigated"
            else:
                event = "re-investigated (no new evidence)"

        elif action == ACTION_SEARCH_AR:
            reward += COST_SEARCH_AR
            if not self._ar_searched:
                self._ar_searched = True
                self._ar_candidates = find_ar_matches(
                    self.case.payment,
                    self.case.invoices,
                    window_days=AR_WINDOW_DAYS,
                    max_combo=AR_MAX_COMBO,
                )[:MAX_CANDIDATES]
                n = len(self._ar_candidates)
                event = f"searched accounts receivable — found {n} candidate match(es)"
            else:
                event = "re-searched AR (no new evidence)"

        elif ACTION_ALLOCATE_BASE <= action < ACTION_ALLOCATE_BASE + MAX_CANDIDATES:
            cand_idx = action - ACTION_ALLOCATE_BASE
            reward, terminated, event = self._try_allocate(cand_idx)

        elif action == ACTION_LEAVE_UNAPPLIED:
            terminated = True
            if self.case.expects_unapplied:
                self._outcome = "correct_unapplied"
                reward += REWARD_CORRECT
            else:
                self._outcome = "incorrect_unapplied"
                reward += REWARD_INCORRECT
            event = "left cash unapplied"

        elif action == ACTION_ESCALATE:
            reward += COST_ESCALATE
            terminated = True
            if self.case.expects_unapplied:
                self._outcome = "correct_escalated"
                reward += REWARD_CORRECT
            else:
                # Safe but inefficient fallback on a determinable case.
                self._outcome = "escalated"
            event = "escalated to human reviewer"

        # Enforce the step limit to prevent infinite loops.
        truncated = False
        if not terminated and self._steps >= self.max_steps:
            truncated = True
            if self._outcome == "pending":
                self._outcome = "truncated"

        self._done = terminated or truncated
        info = self._build_info()
        info["event"] = event
        return self._build_obs(), float(reward), terminated, truncated, info

    def render(self):  # pragma: no cover - simple text aid
        print(self._build_info())

    # ------------------------------------------------------------------
    # Allocation handling
    # ------------------------------------------------------------------
    def _current_candidates(self) -> list[list[Allocation]]:
        """Candidate allocations available now.

        Once the AR has been searched, candidates come from the AR matcher;
        otherwise they are the scenario's pre-authored candidates (empty for
        AR scenarios until a search is performed).
        """
        if self._ar_searched:
            return self._ar_candidates
        return self.case.candidate_allocations

    def _try_allocate(self, cand_idx: int) -> tuple[float, bool, str]:
        candidates = self._current_candidates()
        if cand_idx >= len(candidates):
            # Unsupported candidate for this case: no reward, no state change.
            return REWARD_UNSUPPORTED, False, "unsupported allocation (no such candidate)"

        allocation = candidates[cand_idx]
        result = self.ledger.apply_batch(allocation)
        if not result.accepted:
            # Rejected by the ledger (e.g. over-allocation): no reward.
            return REWARD_UNSUPPORTED, False, f"allocation rejected ({result.reason})"

        # A valid allocation was applied: the case is now resolved.
        if self.case.expects_unapplied:
            self._outcome = "incorrect_allocated"
            return REWARD_INCORRECT, True, "allocated (should have stayed unapplied)"

        if self._matches_ground_truth():
            self._outcome = "correct_allocated"
            return REWARD_CORRECT, True, "allocated correctly"

        self._outcome = "incorrect_allocated"
        return REWARD_INCORRECT, True, "allocated incorrectly"

    def _matches_ground_truth(self) -> bool:
        truth = self.case.ground_truth_allocations
        if truth is None:
            return False
        applied = Counter(
            (a.invoice_id, a.amount_cents) for a in self.ledger.allocations
        )
        expected = Counter((a.invoice_id, a.amount_cents) for a in truth)
        return applied == expected

    # ------------------------------------------------------------------
    # Observation / info builders
    # ------------------------------------------------------------------
    def _remittance_visible(self) -> bool:
        return (not self.case.remittance.hidden) or self._investigated

    def _remittance_targets_unloaded(self) -> bool:
        """True if a visible remittance names invoices, none of which are in the
        loaded AR.

        This models the "AR not loaded" flavour of a missing-AR case: we know
        (from the remittance) which invoices the customer intended to pay, but
        those invoices are absent from the open AR we have, so the cash cannot
        be applied even though the intent is clear.
        """
        if not self._remittance_visible():
            return False
        refs = self.case.remittance.referenced_invoice_ids
        if not refs:
            return False
        loaded = {inv.invoice_id for inv in self.case.invoices}
        return all(r not in loaded for r in refs)

    def _build_obs(self) -> np.ndarray:
        case = self.case
        ledger = self.ledger
        visible = self._remittance_visible()

        # Fixed number of per-invoice balance slots (first MAX_INVOICES shown;
        # larger ARs are summarised by the ar_* fields instead).
        balances = [
            ledger.invoice_balance(inv.invoice_id) / 100.0
            for inv in case.invoices[:MAX_INVOICES]
        ]
        balances += [0.0] * (MAX_INVOICES - len(balances))

        num_refs = len(case.remittance.referenced_invoice_ids) if visible else 0
        reliable = 1.0 if (visible and case.remittance.reliable) else 0.0
        total_outstanding = sum(ledger.all_invoice_balances().values()) / 100.0

        # AR context summary (does not expose the hidden correct answer).
        ar_customer_invoices = [
            inv for inv in case.invoices
            if inv.customer_id == case.payment.customer_id
        ]
        ar_in_window = eligible_invoices(
            case.payment, case.invoices, window_days=AR_WINDOW_DAYS
        )

        obs = [
            case.payment.amount_cents / 100.0,
            ledger.unapplied_cents / 100.0,
            float(len(case.invoices)),
            total_outstanding,
            1.0 if self._investigated else 0.0,
            1.0 if visible else 0.0,
            float(num_refs),
            reliable,
            float(len(self._current_candidates())),
            *balances,
            float(self._steps),
            float(len(ar_customer_invoices)),
            float(len(ar_in_window)),
            1.0 if self._ar_searched else 0.0,
            float(len(self._ar_candidates)),
        ]
        return np.asarray(obs, dtype=np.float32)

    def _build_info(self) -> dict[str, Any]:
        """Human-readable info. Deliberately excludes hidden ground truth."""
        case = self.case
        visible = self._remittance_visible()
        return {
            "case_id": case.case_id,
            "title": case.title,
            "customer": case.customer.name,
            "payment_cents": case.payment.amount_cents,
            "unapplied_cents": self.ledger.unapplied_cents,
            "invoice_balances": self.ledger.all_invoice_balances(),
            "investigated": self._investigated,
            "remittance_visible": visible,
            "remittance_references": (
                list(case.remittance.referenced_invoice_ids) if visible else []
            ),
            "remittance_reliable": (case.remittance.reliable if visible else None),
            "num_candidates": len(self._current_candidates()),
            "steps": self._steps,
            "outcome": self._outcome,
            "accounting_ok": self.ledger.accounting_ok(),
            "ar_searched": self._ar_searched,
            "ar_num_matches": len(self._ar_candidates),
            "resolution_category": self._resolution_category(),
        }

    def _resolution_category(self) -> str:
        """Classify *how* the case was (or is being) resolved.

        This is informational only (it does not affect reward) and is used by
        the demos to label the result, e.g. a "Missing AR" case is one where
        the agent searched the AR but found no reconciling invoice.
        """
        outcome = self._outcome
        if outcome == "correct_allocated":
            return "ar_auto_match" if self._ar_searched else "remittance_match"
        if outcome == "incorrect_allocated":
            return "misapplied"
        if outcome in (
            "correct_escalated", "escalated",
            "correct_unapplied", "incorrect_unapplied", "truncated",
        ):
            # Remittance names invoices we don't have loaded -> missing AR.
            if self._remittance_targets_unloaded():
                return "missing_ar"
            if self._ar_searched:
                n = len(self._ar_candidates)
                if n == 0:
                    return "missing_ar"
                if n >= 2:
                    return "ambiguous_ar"
            return "no_evidence"
        return "pending"

    # ------------------------------------------------------------------
    # Helpers for agents (non-ground-truth information only)
    # ------------------------------------------------------------------
    def visible_remittance(self):
        """Return the remittance if currently visible, else ``None``."""
        if self._remittance_visible():
            return self.case.remittance
        return None

    def candidate_allocations(self) -> list[list[Allocation]]:
        """Candidate allocations available to the agent (not ground truth).

        After an AR search these are the matcher's candidates; otherwise the
        scenario's pre-authored candidates.
        """
        return [list(c) for c in self._current_candidates()]

    def ar_searched(self) -> bool:
        return self._ar_searched
