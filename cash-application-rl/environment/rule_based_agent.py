"""A deterministic rule-based baseline for the Cash Application environment.

This baseline encodes the heuristics a careful human cash-application analyst
would follow. It uses only information the environment exposes to an agent:
the visible remittance (if any) and the predefined candidate allocations. It
never reads the hidden ground-truth resolution.

Policy (evaluated top to bottom each step):
  1. If remittance evidence is still hidden, INVESTIGATE to gather evidence.
  2. If the remittance is reliable with references, pick the first candidate
     allocation that is fully *supported* by those references and valid
     against the ledger, and ALLOCATE.
  3. If the remittance is missing/unreliable, SEARCH_AR to look for an
     amount/customer/date match in the open AR before giving up.
  4. After searching, if the AR produced exactly ONE valid candidate, ALLOCATE
     it (a confident auto-match).
  5. Otherwise (no match, or an ambiguous multiple-match), ESCALATE for human
     review rather than guess.
"""

from __future__ import annotations

from .cash_application_env import (
    ACTION_ALLOCATE_BASE,
    ACTION_ESCALATE,
    ACTION_INVESTIGATE,
    ACTION_SEARCH_AR,
    CashApplicationEnv,
)


class RuleBasedAgent:
    """Deterministic policy; no access to simulator ground truth."""

    name = "rule_based_baseline"

    def act(self, env: CashApplicationEnv) -> int:
        remittance = env.visible_remittance()

        # 1. Gather remittance evidence first if it is hidden.
        if remittance is None:
            return ACTION_INVESTIGATE

        # 2. Reliable remittance with references -> allocate the supported,
        #    valid candidate.
        if remittance.reliable and remittance.referenced_invoice_ids:
            refs = set(remittance.referenced_invoice_ids)
            for idx, candidate in enumerate(env.candidate_allocations()):
                invoice_ids = {alloc.invoice_id for alloc in candidate}
                supported = invoice_ids.issubset(refs) and len(invoice_ids) > 0
                valid = all(env.ledger.validate(a).accepted for a in candidate)
                if supported and valid:
                    return ACTION_ALLOCATE_BASE + idx
            # Remittance pointed somewhere we cannot support -> escalate.
            return ACTION_ESCALATE

        # 3. Missing / unreliable remittance: try the open AR before giving up.
        if not env.ar_searched():
            return ACTION_SEARCH_AR

        # 4. Auto-match only when the AR gives exactly ONE valid candidate.
        candidates = env.candidate_allocations()
        valid_candidates = [
            (idx, cand)
            for idx, cand in enumerate(candidates)
            if cand and all(env.ledger.validate(a).accepted for a in cand)
        ]
        if len(valid_candidates) == 1:
            return ACTION_ALLOCATE_BASE + valid_candidates[0][0]

        # 5. No match, or ambiguous multiple matches -> escalate.
        return ACTION_ESCALATE

    def run_episode(self, env: CashApplicationEnv, options: dict | None = None):
        """Run one full episode and return a trajectory summary."""
        obs, info = env.reset(options=options)
        trajectory = []
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = self.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append(
                {
                    "action": action,
                    "reward": reward,
                    "event": info.get("event", ""),
                    "outcome": info["outcome"],
                }
            )
        return {
            "total_reward": total_reward,
            "outcome": info["outcome"],
            "trajectory": trajectory,
            "final_info": info,
        }
