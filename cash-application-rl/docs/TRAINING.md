# Reinforcement-learning training

This document describes how an RL agent is trained on `CashApplicationEnv` and
records the results included in the project archive. These are exploratory
within-scenario checks, not independent held-out evaluation.

## Primary trainer: tabular Q-learning (dependency-free)

`training/tabular_q_learning.py` implements standard tabular Q-learning:

```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

- **State abstraction.** The raw observation is a 17-dim continuous vector, but
  the decision-relevant state is small and discrete. `state_key(obs)` maps each
  observation to a 7-tuple using only information the environment already
  exposes to the agent:
  `(investigated, remittance_visible, remittance_refs_bucket,
  remittance_reliable, num_candidates_bucket, ar_searched, ar_matches_bucket)`,
  where counts are bucketed as 0 / 1 / 2+.
- **Training mix.** Each episode samples one of the eight scenarios uniformly
  at random (seeded), so the agent must learn a single policy that handles all
  cases.
- **Exploration.** Epsilon-greedy with linear decay (1.0 → 0.02).
- **Defaults.** `alpha=0.5`, `gamma=0.95`, `episodes=4000`, `max_steps=8`.

Run it:

```bash
python -m training.tabular_q_learning --episodes 4000
```

Artifacts are written to `results/`: `tabular_rewards.csv` (per-episode reward)
and, if matplotlib is installed, `tabular_learning_curve.png`.

## Latest results (seed 0, 4,000 episodes)

- Smoothed (200-episode) mean reward: **~0.00 → ~0.95**.
- Distinct discrete states learned: **23**.
- Learning curve: `results/tabular_learning_curve.png`.

Trained greedy policy vs. the deterministic rule-based baseline:

| Case | Title | Baseline outcome | RL outcome | Baseline | RL |
|------|-------|------------------|------------|---------:|---:|
| CASE-1 | Exact Match | correct_allocated | correct_allocated | +1.00 | +1.00 |
| CASE-2 | Partial Payment | correct_allocated | correct_allocated | +1.00 | +1.00 |
| CASE-3 | Consolidated Payment | correct_allocated | correct_allocated | +1.00 | +1.00 |
| CASE-4 | Ambiguous Payment | correct_allocated | correct_allocated | +0.95 | +1.00 |
| CASE-5 | Missing AR | correct_escalated | correct_unapplied | +0.80 | +1.00 |
| CASE-6 | AR Auto-Match | correct_allocated | correct_allocated | +0.90 | +0.90 |
| CASE-7 | AR Ambiguous | correct_escalated | correct_unapplied | +0.80 | +0.90 |
| CASE-8 | Missing AR (not loaded) | correct_escalated | correct_unapplied | +0.85 | +0.95 |
| **TOTAL** | | | | **+7.30** | **+7.75** |

On the eight case types also used during training, the trained policy reaches a
`correct_*` outcome and matches or exceeds the baseline reward. This does not
establish performance on unseen customers or payment patterns.

### Honest findings (not bugs — reward-design signals)

The agent optimises the *given* reward, which surfaces two artifacts of the
current untuned reward function:

1. **It prefers "leave unapplied" over "escalate"** on undeterminable cases
   (CASE-5/7/8), because leaving unapplied scores +1.0 while escalating costs
   −0.10 (net +0.90). Both are counted as correct, but a real deployment would
   likely want *escalate* preferred so a human reviews the item. This is a
   reward-shaping decision, documented in `docs/NEXT_STEPS.md`.
2. **It can skip investigation on CASE-4**, allocating the correct candidate
   directly, because candidate allocations are exposed before investigation.
   Making investigation genuinely necessary (hiding candidates until evidence
   is gathered) is a future environment-design refinement.

These are exactly the kind of insights a learning environment is meant to
expose, and they motivate the planned reward/observation refinements.

## Optional trainer: PPO (stable-baselines3)

`training/train_ppo.py` trains a PPO `MlpPolicy` on the same env. It is optional
and requires deep-RL extras that are **not** in the core requirements:

```bash
pip install "stable-baselines3>=2.3" torch
python -m training.train_ppo --timesteps 100000
```

It evaluates the trained PPO policy against the baseline on every scenario and
saves the model to `results/ppo_cash_application.zip`.
