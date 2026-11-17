"""Tests that the tabular Q-learning agent actually learns.

These run a short, seeded training run and assert measurable learning, so the
RL training path is guarded against regressions. Nothing here is fabricated:
the assertions are over the results of a real training run.
"""

from __future__ import annotations

from training.tabular_q_learning import (
    baseline_results,
    evaluate_policy,
    moving_average,
    train,
)


def test_training_improves_mean_reward():
    _, rewards = train(episodes=3000, seed=0)
    ma = moving_average(rewards, window=200)
    # The smoothed reward at the end must be clearly higher than at the start.
    assert ma[-1] > ma[0] + 0.5
    assert ma[-1] > 0.8  # converges near the optimal (~0.95) reward region


def test_trained_policy_resolves_every_scenario_correctly():
    agent, _ = train(episodes=3000, seed=0)
    for r in evaluate_policy(agent):
        # Every scenario is resolved with a "correct_*" outcome ...
        assert r["outcome"].startswith("correct_"), r
        # ... and a positive episode reward.
        assert r["reward"] > 0, r


def test_trained_policy_at_least_matches_baseline_total_reward():
    agent, _ = train(episodes=3000, seed=0)
    trained = {r["case_id"]: r["reward"] for r in evaluate_policy(agent)}
    base = {r["case_id"]: r["reward"] for r in baseline_results()}
    assert sum(trained.values()) >= sum(base.values())
