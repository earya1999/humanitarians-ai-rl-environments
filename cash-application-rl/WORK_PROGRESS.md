# Work Progress

Dated notes describing implementation changes, checks, and reported experiments.

## 2026-11-11 — First RL training loop (tabular Q-learning + optional PPO)

Added an actual reinforcement-learning trainer. This reverses the earlier
milestone scope (which deferred training) at the user's explicit request. The training metrics below are from a recorded run on the eight defined scenarios.

### COMPLETED AND VERIFIED
- `training/tabular_q_learning.py`: dependency-free tabular Q-learning over a
  7-feature discretisation of the observation (`state_key`). Trains on a
  uniform mix of all eight scenarios with epsilon-greedy exploration and the
  standard Q-learning update. Prints an ASCII learning curve + a baseline
  comparison table; saves `results/tabular_rewards.csv` and (if matplotlib is
  present) `results/tabular_learning_curve.png`.
- `training/train_ppo.py`: optional PPO trainer (stable-baselines3 + torch),
  import-guarded so it is not a core dependency.
- Real result (seed 0, 4,000 episodes): smoothed reward ~0.00 → ~0.95; the
  trained greedy policy resolves every scenario with a `correct_*` outcome and
  reaches total reward +7.75 vs the baseline's +7.30.
- `tests/test_training.py`: asserts measurable learning (end >> start,
  converges > 0.8), all scenarios resolved correctly, and total reward ≥
  baseline. **Full suite now 36 tests, all passing.**
- Documented method + results in `docs/TRAINING.md`; updated `README.md`,
  `requirements.txt` (optional extras), and `docs/NEXT_STEPS.md`.

### HONEST FINDINGS (reward-design signals, not bugs)
- The agent prefers *leave unapplied* (+1.0) over *escalate* (+0.9) on
  undeterminable cases, and can skip *investigate* on CASE-4 (candidates are
  exposed before investigation). Both are logged as reward-shaping / env-design
  follow-ups in `docs/NEXT_STEPS.md`.

## 2026-11-11 — Second "Missing AR" flavour: remittance present, AR not loaded

Added CASE-8 to model the situation where the remittance **is** present and
names a specific invoice, but that invoice is not in the loaded AR (the AR may
not have been loaded). We know the intent but cannot apply the cash.

### COMPLETED AND VERIFIED
- `scenarios.py`: CASE-8 — $2,000 payment, reliable remittance naming INV-950
  (absent from the shared AR). Two loaded invoices are also $2,000, so an
  amount-only match would misapply; the remittance says not to.
- `cash_application_env.py`: `_remittance_targets_unloaded()` detects when a
  visible remittance names only unloaded invoices; `_resolution_category()`
  now classifies that outcome as `missing_ar` even without an AR search.
- Rule-based baseline already handles it correctly: it investigates, finds no
  supported/loadable candidate, and escalates (does not amount-match).
- Tests: added a dedicated CASE-8 test (escalates, `missing_ar`, nothing
  applied) and extended the all-scenarios / categories tests. **33 tests pass.**
- Updated `README.md`, `docs/SCENARIOS.md`, and the web demo's category help.

## 2026-11-11 — Rebuilt scenarios on one shared AR

Reworked all seven scenarios so they derive from a single shared list of ten
open invoices for one customer (Northwind Trading Co., `CUST-100`), instead of
each case having its own bespoke customer and invoices. This makes the demo
easier to follow: the open AR is constant and only the incoming payment (and
whether a remittance is attached) changes per case.

### COMPLETED AND VERIFIED
- `environment/scenarios.py`: added `_AR_INVOICES`/`_shared_ar()`; rebuilt
  CASE-1…CASE-7 as different payments against the shared AR. Payment amounts
  were chosen so the AR matcher provably yields the intended match count
  (verified empirically): $14,000 → exactly one 4-invoice combination
  (CASE-6), $1,000 → two $1,000 invoices (CASE-7), $2,750 → none (CASE-5).
- Updated `tests/test_accounting.py` and `tests/test_environment.py` to the new
  invoice IDs/amounts. **All 32 tests pass**, and `check_env` still passes.
- Rule-based baseline resolves every case correctly: CASE-1–4/6 allocate,
  CASE-5/7 escalate, with resolution categories unchanged.
- Updated `README.md` and `docs/SCENARIOS.md` to document the shared AR.

## 2026-11-11 — AR auto-matching extension

Added an Accounts-Receivable (AR) auto-matching capability for the
missing-remittance situation, grounded in real cash-application practice.

### COMPLETED AND VERIFIED
- `environment/ar_matcher.py`: filters the open AR to the paying customer
  within a ±7-day window and finds invoice combinations (up to four) whose
  balances reconcile to the payment amount; returns them sorted by date
  proximity. Amount is the primary signal; customer and date are filters.
- New `SEARCH_AR` action (action space is now `Discrete(8)`), new AR state,
  and four new observation fields (open AR count, in-window count, searched
  flag, number of matches). Observation size is now 17. Hidden ground truth is
  still never exposed.
- Invoice/Payment gained optional dates for date-proximity matching.
- Two new scenarios: **CASE-6 (AR Auto-Match)** — a 12-invoice AR with a single
  unique reconciliation — and **CASE-7 (AR Ambiguous)** — two valid
  reconciliations, so the correct action is to escalate.
- Rule-based baseline extended: on a missing/unreliable remittance it now
  searches the AR and auto-matches only when exactly one valid combination
  exists, otherwise escalates. It still never reads ground truth.
- Tests: 30 pass (was 22), including AR unique-match, AR ambiguity, date-window
  exclusion, and the AR observation fields. `check_env` passes for the new
  multi-invoice cases. Terminal and browser demos both narrate the AR flow.

### IN PROGRESS / PLANNED
- Tolerance for short-pays/deductions, cross-customer AR distractors, and
  confidence-scored matching remain future work (see docs/NEXT_STEPS.md).

## 2026-11-11 — Initial environment-development milestone

Set up the first functional milestone of `cash-application-rl`.

### COMPLETED AND VERIFIED
- Financial data models (`Customer`, `Invoice`, `Payment`, `Remittance`,
  `Allocation`, `ReconciliationCase`) using integer-cents money.
- Integer-cents `Ledger` tracking outstanding invoice balances and the
  remaining unapplied payment balance, with validation that rejects
  duplicate, over-invoice, and over-payment allocations, plus atomic batch
  application with rollback.
- Five reproducible synthetic scenarios (exact match, partial, consolidated,
  ambiguous with hidden remittance, and missing remittance), each with a
  hidden ground-truth resolution where determinable.
- `CashApplicationEnv` implementing the modern Gymnasium API
  (`reset(seed, options)`, `step` returning the 5-tuple) with a fixed-size
  13-element observation, a 5-action discrete space, a step limit, and a
  preliminary (untuned) reward function. Observations do not expose hidden
  ground truth.
- Deterministic rule-based baseline that investigates when evidence is
  hidden, allocates only evidence-supported candidates, and escalates when
  evidence is insufficient. It does not read simulator ground truth.
- pytest suite (22 tests) covering the required accounting and environment
  checks. **All 22 tests pass.** The Gymnasium `check_env` passes.
- `demo.py` live terminal demo; the baseline resolves all five scenarios
  with accounting invariants holding.

### IN PROGRESS
- Reward design refinement (current values are initial design choices,
  not tuned or empirically validated).
- Scenario extension (more cases, noisier and partially-reliable evidence).

### PLANNED
- RL training (e.g. PPO) once the reward and observation design stabilise.
- Stronger baselines and generalization experiments across scenario
  distributions.
- Harvey-style document/remittance understanding tasks.
- A proper research evaluation protocol and metrics.
