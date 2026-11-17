# Next Steps

Planned research directions, in rough priority order. None of this has been
implemented or run yet — this is a roadmap, not a results log.

## Near term — refine the environment
- **Reward refinement.** The current reward values are initial, untuned
  design choices. Run sensitivity analyses and align incentives with
  business cost (e.g. cost of misapplication vs. cost of escalation).
  Training already exposed two concrete signals (see `docs/TRAINING.md`): the
  agent prefers *leave unapplied* (+1.0) over *escalate* (+0.9) on
  undeterminable cases, and it can skip *investigate* on CASE-4 because
  candidates are exposed before investigation. Shape the reward so escalation
  is preferred for human-review cases, and hide candidates until the relevant
  evidence is gathered so investigation is genuinely necessary.
- **Scenario extension.** Add more cases: overpayments, short payments with
  deductions, currency/date noise, partially-reliable remittance, multiple
  payments per customer, and distractor invoices.
- **Richer evidence.** Model remittance quality on a spectrum rather than a
  single reliable/unreliable flag.

## AR auto-matching — implemented, with refinements to come
A first AR auto-matching capability exists (`SEARCH_AR` action + CASE-6/7 and
`environment/ar_matcher.py`): on a missing remittance the agent searches the
customer's open AR for amount-reconciling combinations within a date window,
and auto-matches only when the answer is unambiguous. Remaining work:
- **Tolerance** for short-pays and deductions (currently exact-amount only).
- **Cross-customer AR** distractors to exercise the customer filter (today all
  AR items in a scenario belong to the paying customer).
- **Confidence scoring** instead of a hard "exactly one match" rule, so the
  agent can weigh date proximity, amount closeness, and match count — turning
  the auto-match-vs-escalate boundary into a tunable precision/recall decision.
- **Outstanding-balance matching** against partially-paid invoices (the matcher
  currently assumes AR items are fully open).

## Mid term — learning and baselines
- **RL training — first pass done.** A dependency-free tabular Q-learning
  trainer now learns a policy that resolves all scenarios and matches/beats the
  baseline (`training/tabular_q_learning.py`, results in `docs/TRAINING.md`). An
  optional PPO trainer (`training/train_ppo.py`) is provided. Remaining: run
  PPO at scale, add seeds/variance reporting, and compare sample efficiency.
- **Stronger baselines.** Add a probabilistic/scoring baseline and a
  cost-sensitive heuristic to make RL comparisons meaningful.
- **Generalization experiments.** Train on one scenario distribution and
  evaluate on held-out distributions.

## Longer term — research program
- **Harvey-style document tasks.** Incorporate document/remittance
  understanding (parsing real-world-like remittance artifacts) as an
  upstream perception step feeding the decision environment.
- **Evaluation protocol.** Define metrics (resolution accuracy, unnecessary
  escalations, investigation efficiency, monetary error) and a reproducible
  benchmark suite.
- **Packaging.** Register the environment with Gymnasium and publish a
  versioned, citable research artifact.
