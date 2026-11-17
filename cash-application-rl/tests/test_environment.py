"""Environment-level tests for the Gymnasium interface and the baseline.

Covers (from the milestone test checklist):
  4. Evidence revelation during investigation
  5. Missing-remittance escalation
  10. Gymnasium environment interface validity
Plus: the rule-based baseline resolves every scenario correctly, and the
observation never leaks hidden ground truth.
"""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.cash_application_env import (
    ACTION_ALLOCATE_BASE,
    ACTION_ESCALATE,
    ACTION_INVESTIGATE,
    ACTION_SEARCH_AR,
    N_ACTIONS,
    OBS_SIZE,
    CashApplicationEnv,
)
from environment.rule_based_agent import RuleBasedAgent


# --- 10. Gymnasium environment interface validity ---------------------------
def test_gymnasium_check_env_passes():
    # Use a fixed case so the checker's resets are deterministic.
    env = CashApplicationEnv(case_id="CASE-1")
    # skip_render_check avoids requiring a rendering backend.
    check_env(env, skip_render_check=True)


def test_reset_returns_obs_and_info():
    env = CashApplicationEnv(case_id="CASE-4")
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert env.observation_space.contains(obs)
    assert info["case_id"] == "CASE-4"


def test_step_signature_and_action_space():
    env = CashApplicationEnv(case_id="CASE-1")
    env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.observation_space.contains(obs)


def test_seed_reproducibility_for_random_case():
    env1 = CashApplicationEnv()
    env2 = CashApplicationEnv()
    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    assert info1["case_id"] == info2["case_id"]
    assert np.allclose(obs1, obs2)


def test_episode_truncates_at_step_limit():
    env = CashApplicationEnv(case_id="CASE-1", max_steps=3)
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(ACTION_INVESTIGATE)
        if terminated:
            break
    assert truncated  # repeated investigate must hit the step limit


# --- 4. Evidence revelation during investigation ----------------------------
def test_investigation_reveals_hidden_remittance():
    env = CashApplicationEnv(case_id="CASE-4")
    _, info = env.reset(seed=0)
    # Before investigation the hidden remittance must NOT be visible.
    assert info["remittance_visible"] is False
    assert info["remittance_references"] == []
    assert env.visible_remittance() is None

    _, reward, _, _, info = env.step(ACTION_INVESTIGATE)
    # After investigation the evidence is revealed and a small cost applied.
    assert info["remittance_visible"] is True
    assert set(info["remittance_references"]) == {"INV-001", "INV-004"}
    assert reward < 0


def test_observation_does_not_leak_ground_truth():
    env = CashApplicationEnv(case_id="CASE-4")
    obs, info = env.reset(seed=0)
    # The hidden-remittance ref count slot must read 0 before investigation.
    # OBS layout index 6 == remittance_num_refs.
    assert obs[6] == 0.0
    # The full observation is finite and within the declared space.
    assert env.observation_space.contains(obs)
    assert np.all(np.isfinite(obs))


# --- 5. Missing-remittance escalation ---------------------------------------
def test_missing_remittance_escalation_is_correct():
    env = CashApplicationEnv(case_id="CASE-5")
    env.reset(seed=0)
    # Investigation reveals unreliable / empty evidence.
    _, _, _, _, info = env.step(ACTION_INVESTIGATE)
    assert info["remittance_reliable"] is False
    assert info["remittance_references"] == []
    # Escalation is the correct resolution for an undeterminable case.
    _, reward, terminated, _, info = env.step(ACTION_ESCALATE)
    assert terminated
    assert info["outcome"] == "correct_escalated"


def test_allocating_missing_remittance_case_is_incorrect():
    env = CashApplicationEnv(case_id="CASE-5")
    env.reset(seed=0)
    # Candidate 0 is a tempting but unsupported full allocation.
    _, reward, terminated, _, info = env.step(ACTION_ALLOCATE_BASE)
    assert terminated
    assert info["outcome"] == "incorrect_allocated"
    assert reward < 0


# --- Rule-based baseline resolves every scenario ----------------------------
def test_rule_based_agent_resolves_all_scenarios():
    agent = RuleBasedAgent()
    expected_outcomes = {
        "CASE-1": "correct_allocated",
        "CASE-2": "correct_allocated",
        "CASE-3": "correct_allocated",
        "CASE-4": "correct_allocated",
        "CASE-5": "correct_escalated",
        "CASE-6": "correct_allocated",
        "CASE-7": "correct_escalated",
        "CASE-8": "correct_escalated",
    }
    for case_id, expected in expected_outcomes.items():
        env = CashApplicationEnv(case_id=case_id)
        result = agent.run_episode(env)
        assert result["outcome"] == expected, (case_id, result["outcome"])
        assert result["final_info"]["accounting_ok"] is True


def test_rule_based_agent_investigates_before_allocating_ambiguous():
    agent = RuleBasedAgent()
    env = CashApplicationEnv(case_id="CASE-4")
    result = agent.run_episode(env)
    actions = [t["action"] for t in result["trajectory"]]
    # First action must be INVESTIGATE for the ambiguous case.
    assert actions[0] == ACTION_INVESTIGATE
    assert result["outcome"] == "correct_allocated"


# --- AR auto-matching (missing remittance) ----------------------------------
def test_ar_search_reveals_unique_match_and_allocates():
    env = CashApplicationEnv(case_id="CASE-6")
    env.reset(seed=0)
    # No candidates are available before searching the AR.
    _, info = env.reset(seed=0)
    assert info["num_candidates"] == 0
    assert info["ar_searched"] is False

    env.step(ACTION_INVESTIGATE)             # reveals no reliable remittance
    _, _, _, _, info = env.step(ACTION_SEARCH_AR)
    # AR search finds exactly one reconciling combination.
    assert info["ar_searched"] is True
    assert info["ar_num_matches"] == 1
    assert info["num_candidates"] == 1

    _, reward, terminated, _, info = env.step(ACTION_ALLOCATE_BASE)
    assert terminated
    assert info["outcome"] == "correct_allocated"
    for inv_id in ("INV-003", "INV-005", "INV-006", "INV-000"):
        assert info["invoice_balances"][inv_id] == 0
    assert info["accounting_ok"] is True


def test_ar_search_ambiguous_match_leads_to_escalation():
    env = CashApplicationEnv(case_id="CASE-7")
    env.reset(seed=0)
    env.step(ACTION_INVESTIGATE)
    _, _, _, _, info = env.step(ACTION_SEARCH_AR)
    # More than one combination reconciles -> ambiguous.
    assert info["ar_num_matches"] >= 2
    # Escalation is the correct resolution for an ambiguous AR match.
    _, _, terminated, _, info = env.step(ACTION_ESCALATE)
    assert terminated
    assert info["outcome"] == "correct_escalated"


def test_ar_search_produces_the_unique_reconciling_set():
    # CASE-6 ($14,000) has exactly one reconciling combination in the open AR.
    env = CashApplicationEnv(case_id="CASE-6")
    env.reset(seed=0)
    env.step(ACTION_INVESTIGATE)
    env.step(ACTION_SEARCH_AR)
    matched_ids = {
        a.invoice_id for cand in env.candidate_allocations() for a in cand
    }
    assert matched_ids == {"INV-003", "INV-005", "INV-006", "INV-000"}


def test_resolution_categories_are_classified():
    agent = RuleBasedAgent()
    expected = {
        "CASE-1": "remittance_match",
        "CASE-5": "missing_ar",
        "CASE-6": "ar_auto_match",
        "CASE-7": "ambiguous_ar",
        "CASE-8": "missing_ar",
    }
    for case_id, cat in expected.items():
        env = CashApplicationEnv(case_id=case_id)
        result = agent.run_episode(env)
        assert result["final_info"]["resolution_category"] == cat, case_id


def test_remittance_naming_unloaded_invoice_escalates_without_misapplying():
    # CASE-8: the remittance names INV-950, which is not in the loaded AR.
    # Two loaded invoices (INV-002, INV-009) are also $2,000, but the agent
    # must NOT amount-match to them; it should escalate as a missing-AR case.
    agent = RuleBasedAgent()
    env = CashApplicationEnv(case_id="CASE-8")
    result = agent.run_episode(env)
    info = result["final_info"]
    assert result["outcome"] == "correct_escalated"
    assert info["resolution_category"] == "missing_ar"
    # Nothing was applied: every open invoice keeps its full balance and the
    # payment stays fully unapplied.
    assert info["unapplied_cents"] == info["payment_cents"]
    assert all(bal > 0 for bal in info["invoice_balances"].values())
    assert info["accounting_ok"] is True


def test_ar_observation_fields_do_not_leak_ground_truth():
    env = CashApplicationEnv(case_id="CASE-6")
    obs, _ = env.reset(seed=0)
    # ar_searched (index 15) and ar_num_matches (index 16) start at 0.
    assert obs[15] == 0.0
    assert obs[16] == 0.0
    assert env.observation_space.contains(obs)
