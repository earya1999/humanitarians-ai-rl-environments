# cash-application-rl

A small, reproducible **Gymnasium-compatible research environment** for
studying sequential financial decisions in Accounts Receivable (AR)
**cash application** — the process of deciding how an incoming customer
payment should be matched to open invoices.

This repository is an open-source research project for **Humanitarians AI**
under the **Learning Environments** research vertical. It is at the
**initial environment-development milestone**: a working simulator,
scenarios, a rule-based baseline, accounting safeguards, tests, a live demo,
and a first **reinforcement-learning training** loop (a dependency-free
tabular Q-learning agent, with an optional PPO trainer). The included training results are exploratory and use the same eight case types
as training. They are not evidence of performance on unseen financial workflows.

## What this models

An agent receives one incoming payment and a set of open invoices and must
decide how to reconcile it:

- **Investigate** hidden remittance evidence (at a small cost).
- **Search the open AR** for an amount/customer/date match when remittance is
  missing (at a small cost).
- **Allocate** the payment using a supported candidate allocation.
- **Leave the cash unapplied** and finish.
- **Escalate** the case to a human reviewer (at a small cost).

The challenge is that the correct action depends on *evidence that may be
hidden*, so the agent must sometimes gather information before acting —
making this a natural sequential-decision-under-uncertainty problem.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python demo.py                     # live terminal demo (ambiguous case)
python demo.py --interactive       # type in your own payment + remittance live
python demo.py --step              # pause after each decision (press Enter)
python demo.py --all               # run every scenario + summary table
python demo.py --no-color          # disable ANSI colors

python web_demo.py                 # browser demo (opens http://localhost:8000)
pytest -q                          # run the test suite

python -m training.tabular_q_learning --episodes 4000   # train an RL agent
```

## Repository layout

```
cash-application-rl/
  README.md                 # this file
  WORK_PROGRESS.md          # dated, factual progress log
  requirements.txt
  demo.py                   # live terminal demo
  environment/
    models.py               # Customer/Invoice/Payment/Remittance/Allocation/Case
    ledger.py               # integer-cents ledger + accounting invariants
    scenarios.py            # eight synthetic reconciliation cases
    ar_matcher.py           # AR amount/customer/date auto-matching
    cash_application_env.py # CashApplicationEnv(gymnasium.Env)
    rule_based_agent.py     # deterministic baseline policy
  training/
    tabular_q_learning.py   # primary RL trainer (dependency-free)
    train_ppo.py            # optional PPO trainer (stable-baselines3)
  tests/
    test_accounting.py      # ledger + scenario correctness
    test_environment.py     # Gymnasium interface + baseline behaviour
    test_training.py        # the RL agent actually learns
  docs/
    ENVIRONMENT_DESIGN.md   # observation/action/reward design
    SCENARIOS.md            # the eight scenarios in detail
    NEXT_STEPS.md           # planned research
    TRAINING.md             # how RL training works + latest results
  results/                  # training artifacts (learning curve, CSV)
```

## The eight scenarios

Every case is a different payment against **one shared list of ten open
invoices** for a single customer (Northwind Trading Co., `CUST-100`), all dated
within the AR matcher's ±7-day window.

| Case | Title | Payment | Expected behaviour |
|------|-------|---------|--------------------|
| CASE-1 | Exact Match | $3,000 | Allocate in full, close INV-003 |
| CASE-2 | Partial Payment | $1,200 | Allocate $1,200 to INV-002, leave $800 outstanding |
| CASE-3 | Consolidated Payment | $3,500 | Split across INV-004 + INV-002 |
| CASE-4 | Ambiguous Payment | $2,500 | Investigate, then allocate INV-001 + INV-004 |
| CASE-5 | Missing AR | $2,750 | No remittance; AR has no reconciling invoice → escalate |
| CASE-6 | AR Auto-Match | $14,000 | No remittance → search AR → unique 4-invoice match → allocate |
| CASE-7 | AR Ambiguous | $1,000 | No remittance → search AR → two $1,000 invoices → escalate |
| CASE-8 | Missing AR (not loaded) | $2,000 | Remittance names an invoice absent from the loaded AR → escalate (don't amount-match) |

All data is fictional and synthetic. See [`docs/SCENARIOS.md`](docs/SCENARIOS.md).

CASE-6 and CASE-7 model the realistic "no remittance" workflow: the agent
searches the customer's open AR for an invoice combination that reconciles to
the payment amount within a ±7-day window, and auto-matches only when the
answer is unambiguous. See [`environment/ar_matcher.py`](environment/ar_matcher.py).

## Reinforcement-learning training

The environment is Gymnasium-compliant and training-ready. The primary trainer
is a **dependency-free tabular Q-learning** agent that maps each observation to
a small, interpretable discrete state and learns a Q-table from reward:

```bash
python -m training.tabular_q_learning --episodes 4000
```

It prints a learning curve and a table comparing the trained greedy policy
against the rule-based baseline on every scenario, and saves artifacts to
`results/`. An **optional** PPO trainer (`training/train_ppo.py`) is provided
for deep RL and requires `stable-baselines3` + `torch`.

A recorded 4,000-episode run reports a smoothed episode reward rising from
approximately 0.00 to 0.95 on the eight defined case types. Those cases are
also represented during training, so the outcome is a smoke-test result, not a
held-out generalization benchmark. See [`docs/TRAINING.md`](docs/TRAINING.md)
and [`results/tabular_rewards.csv`](results/tabular_rewards.csv).

## Monetary arithmetic

All money is stored and computed as **integer cents**; dollar strings are
produced for display only. This avoids binary floating-point currency errors.

## Status

See [`WORK_PROGRESS.md`](WORK_PROGRESS.md) for the factual status summary.
Reward values in the environment are **initial, untuned design choices**.
A first RL training loop now exists and learns; note the trained agent
exploits the current (untuned) reward — e.g. it prefers *leave unapplied*
(reward 1.0) over *escalate* (reward 0.9) on undeterminable cases. Shaping the
reward so escalation is preferred for human-review cases is future work.
