"""Reinforcement-learning training utilities for CashApplicationEnv.

Two trainers are provided:

* ``tabular_q_learning`` -- a dependency-free tabular Q-learning agent that
  learns a policy over a small, interpretable discretisation of the
  observation. This is the primary trainer and runs in seconds.
* ``train_ppo`` -- an optional PPO trainer built on stable-baselines3 for
  those who want a deep-RL baseline. It is only importable when
  stable-baselines3 (and PyTorch) are installed.

Nothing here fabricates results: every reported number comes from an actual
training run executed by these scripts.
"""
