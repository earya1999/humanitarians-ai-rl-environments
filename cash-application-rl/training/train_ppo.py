"""Optional deep-RL trainer: PPO on CashApplicationEnv via stable-baselines3.

This is an *optional* extra for those who want a deep-RL baseline alongside the
dependency-free tabular Q-learning trainer. It requires stable-baselines3 and
PyTorch, which are intentionally NOT part of the project's core requirements::

    pip install "stable-baselines3>=2.3" torch

Then run::

    python -m training.train_ppo --timesteps 100000

The script trains a PPO MlpPolicy on a mix of all scenarios, logs the mean
episode reward, evaluates the trained policy against the rule-based baseline on
every scenario, and saves the model to ``results/ppo_cash_application.zip``.

Reported results are printed only when training and evaluation are executed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _require_sb3():
    try:
        import stable_baselines3  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without sb3
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Install the optional deep-RL extras with:\n"
            '    pip install "stable-baselines3>=2.3" torch\n'
            f"(import error: {exc})"
        )


def make_env(max_steps: int = 8):
    """A fresh env that samples a random scenario on every reset."""
    from environment.cash_application_env import CashApplicationEnv

    return CashApplicationEnv(max_steps=max_steps)


def evaluate(model, max_steps: int = 8) -> list[dict]:
    from environment.cash_application_env import ACTION_NAMES, CashApplicationEnv
    from environment.scenarios import SCENARIO_IDS

    results = []
    for case_id in SCENARIO_IDS:
        env = CashApplicationEnv(case_id=case_id, max_steps=max_steps)
        obs, info = env.reset(seed=0)
        done = False
        total = 0.0
        actions = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            actions.append(ACTION_NAMES[action])
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        results.append(
            {
                "case_id": case_id,
                "title": env.case.title,
                "outcome": info["outcome"],
                "reward": total,
                "category": info["resolution_category"],
                "actions": actions,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    _require_sb3()
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(make_env())
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, n_steps=512,
                batch_size=64, gae_lambda=0.95, gamma=0.95)
    model.learn(total_timesteps=args.timesteps)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "ppo_cash_application.zip"
    model.save(model_path)
    print(f"\nSaved PPO model -> {model_path}")

    print("\nTrained PPO policy per scenario:")
    print("-" * 70)
    for r in evaluate(model):
        print(f"{r['case_id']:7} {r['title'][:22]:22} {r['outcome']:22} "
              f"reward={r['reward']:+.2f}")


if __name__ == "__main__":
    main()
