"""Dependency-free tabular Q-learning for CashApplicationEnv.

The environment's observation is a 17-dimensional continuous vector, but the
*decision-relevant* state is small and discrete: whether we have investigated,
whether a remittance is visible/reliable, how many candidates/AR matches are
available, and whether the AR has been searched. We map each observation to a
compact tuple (``state_key``) and learn a Q-value table over that abstraction.

The agent begins with zero-initialized Q-values, acts
epsilon-greedily, and updates Q-values from the reward signal via the standard
Q-learning rule

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

Run it directly::

    python -m training.tabular_q_learning --episodes 4000

It prints a learning curve and a table comparing the trained greedy policy
against the deterministic rule-based baseline across every scenario.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from environment.cash_application_env import (
    ACTION_NAMES,
    N_ACTIONS,
    CashApplicationEnv,
)
from environment.rule_based_agent import RuleBasedAgent
from environment.scenarios import SCENARIO_IDS

# --- Observation indices used to build the discrete state -------------------
# (see CashApplicationEnv OBS layout)
IDX_INVESTIGATED = 4
IDX_REMIT_VISIBLE = 5
IDX_REMIT_REFS = 6
IDX_REMIT_RELIABLE = 7
IDX_NUM_CANDIDATES = 8
IDX_AR_SEARCHED = 15
IDX_AR_MATCHES = 16

StateKey = tuple[int, int, int, int, int, int, int]


def state_key(obs: np.ndarray) -> StateKey:
    """Map a raw observation to a small, interpretable discrete state.

    Counts are bucketed as 0 / 1 / 2+ ("many") so the table stays compact.
    Only information the environment already exposes to the agent is used.
    """

    def bucket(x: float) -> int:
        return min(int(round(float(x))), 2)

    return (
        int(round(float(obs[IDX_INVESTIGATED]))),
        int(round(float(obs[IDX_REMIT_VISIBLE]))),
        bucket(obs[IDX_REMIT_REFS]),
        int(round(float(obs[IDX_REMIT_RELIABLE]))),
        bucket(obs[IDX_NUM_CANDIDATES]),
        int(round(float(obs[IDX_AR_SEARCHED]))),
        bucket(obs[IDX_AR_MATCHES]),
    )


class TabularQLearningAgent:
    """A simple tabular Q-learning agent over the discretised state."""

    name = "tabular_q_learning"

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        alpha: float = 0.5,
        gamma: float = 0.95,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self._rng = random.Random(seed)
        self.q: dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float64)
        )

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action. With epsilon=0 this is the greedy policy."""
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.n_actions)
        return self._greedy(obs)

    def _greedy(self, obs: np.ndarray) -> int:
        q = self.q[state_key(obs)]
        best = float(q.max())
        # Deterministic tie-break: lowest action index among the best.
        return int(np.flatnonzero(q == best)[0])

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        s, s2 = state_key(obs), state_key(next_obs)
        target = reward if done else reward + self.gamma * float(self.q[s2].max())
        self.q[s][action] += self.alpha * (target - self.q[s][action])


def train(
    episodes: int = 4000,
    *,
    seed: int = 0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    alpha: float = 0.5,
    gamma: float = 0.95,
    max_steps: int = 8,
) -> tuple[TabularQLearningAgent, list[float]]:
    """Train on a uniform mix of all scenarios. Returns (agent, reward_history)."""
    rng = random.Random(seed)
    env = CashApplicationEnv(max_steps=max_steps)
    agent = TabularQLearningAgent(alpha=alpha, gamma=gamma, seed=seed)

    rewards: list[float] = []
    for ep in range(episodes):
        # Linear epsilon decay from start to end.
        frac = ep / max(1, episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        case_id = rng.choice(SCENARIO_IDS)
        obs, _ = env.reset(seed=rng.randint(0, 2**31 - 1),
                           options={"case_id": case_id})
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total += reward
        rewards.append(total)
    return agent, rewards


# --- Evaluation -------------------------------------------------------------
def evaluate_policy(agent: TabularQLearningAgent, max_steps: int = 8) -> list[dict]:
    """Run the greedy trained policy once per scenario and record the result."""
    results = []
    for case_id in SCENARIO_IDS:
        env = CashApplicationEnv(case_id=case_id, max_steps=max_steps)
        obs, info = env.reset(seed=0)
        done = False
        total = 0.0
        actions: list[int] = []
        while not done:
            action = agent.act(obs, epsilon=0.0)
            actions.append(action)
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
                "actions": [ACTION_NAMES[a] for a in actions],
            }
        )
    return results


def baseline_results(max_steps: int = 8) -> list[dict]:
    """Run the deterministic rule-based baseline once per scenario."""
    agent = RuleBasedAgent()
    results = []
    for case_id in SCENARIO_IDS:
        env = CashApplicationEnv(case_id=case_id, max_steps=max_steps)
        res = agent.run_episode(env)
        info = res["final_info"]
        results.append(
            {
                "case_id": case_id,
                "title": info["title"],
                "outcome": res["outcome"],
                "reward": res["total_reward"],
                "category": info["resolution_category"],
            }
        )
    return results


# --- Reporting helpers ------------------------------------------------------
def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    acc = sum(values[:window])
    out.append(acc / window)
    for i in range(window, len(values)):
        acc += values[i] - values[i - window]
        out.append(acc / window)
    return out


def ascii_learning_curve(rewards: list[float], window: int = 200,
                         width: int = 60, height: int = 12) -> str:
    """Render a compact ASCII plot of the smoothed episode reward."""
    ma = moving_average(rewards, window)
    if not ma:
        return "(no data)"
    # Downsample to `width` columns.
    idxs = np.linspace(0, len(ma) - 1, num=min(width, len(ma))).astype(int)
    series = [ma[i] for i in idxs]
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1.0
    rows = []
    for r in range(height, 0, -1):
        threshold = lo + span * (r - 0.5) / height
        line = "".join("#" if v >= threshold else " " for v in series)
        rows.append(f"{threshold:+5.2f} |{line}")
    rows.append(f"      +{'-' * len(series)}")
    rows.append(f"       episode 0 .. {len(rewards)}  (reward, {window}-ep moving avg)")
    return "\n".join(rows)


def save_learning_curve_png(rewards: list[float], path: Path,
                            window: int = 200) -> bool:
    """Save a PNG learning curve if matplotlib is available; else return False."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    ma = moving_average(rewards, window)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(rewards)), rewards, color="#c9d6e5", linewidth=0.5,
            label="episode reward")
    offset = window - 1 if len(rewards) >= window else 0
    ax.plot(range(offset, offset + len(ma)), ma, color="#1f6feb", linewidth=2.0,
            label=f"{window}-episode moving avg")
    ax.set_xlabel("training episode")
    ax.set_ylabel("total episode reward")
    ax.set_title("Tabular Q-learning on CashApplicationEnv")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def save_rewards_csv(rewards: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i, f"{r:.4f}"])


def print_comparison(trained: list[dict], baseline: list[dict]) -> None:
    by_id = {r["case_id"]: r for r in baseline}
    print("\nTrained policy vs. rule-based baseline")
    print("-" * 94)
    print(f"{'Case':7} {'Title':24} {'Baseline outcome':22} {'RL outcome':22} "
          f"{'base':>5} {'RL':>5}")
    print("-" * 94)
    rl_total = base_total = 0.0
    for r in trained:
        b = by_id[r["case_id"]]
        rl_total += r["reward"]
        base_total += b["reward"]
        print(f"{r['case_id']:7} {r['title'][:24]:24} {b['outcome']:22} "
              f"{r['outcome']:22} {b['reward']:+5.2f} {r['reward']:+5.2f}")
    print("-" * 94)
    print(f"{'TOTAL':7} {'':24} {'':22} {'':22} {base_total:+5.2f} {rl_total:+5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    print(f"Training tabular Q-learning for {args.episodes} episodes "
          f"(seed={args.seed}, alpha={args.alpha}, gamma={args.gamma})...")
    agent, rewards = train(
        episodes=args.episodes, seed=args.seed, alpha=args.alpha, gamma=args.gamma
    )

    first = moving_average(rewards, args.window)[0] if rewards else 0.0
    last = moving_average(rewards, args.window)[-1] if rewards else 0.0
    print(f"Mean episode reward: start ~{first:+.3f}  ->  end ~{last:+.3f}")
    print(f"Distinct states learned: {len(agent.q)}")
    print()
    print(ascii_learning_curve(rewards, window=args.window))

    outdir = Path(args.outdir)
    save_rewards_csv(rewards, outdir / "tabular_rewards.csv")
    png_ok = save_learning_curve_png(rewards, outdir / "tabular_learning_curve.png",
                                     window=args.window)
    print(f"\nSaved rewards CSV -> {outdir / 'tabular_rewards.csv'}")
    if png_ok:
        print(f"Saved learning curve PNG -> {outdir / 'tabular_learning_curve.png'}")
    else:
        print("(matplotlib not installed; skipped PNG. `pip install matplotlib` "
              "to enable.)")

    trained = evaluate_policy(agent)
    baseline = baseline_results()
    print_comparison(trained, baseline)


if __name__ == "__main__":
    main()
