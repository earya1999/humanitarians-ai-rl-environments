"""Live terminal demo for the Cash Application RL environment.

Runs the deterministic rule-based baseline against a chosen scenario and
prints, step by step, the incoming payment, open invoices, available
remittance evidence, each agent action, newly revealed evidence, the final
invoice balances, the unapplied amount, the reconciliation outcome, and the
accounting checks.

Usage:
    python demo.py                 # default: the ambiguous payment case (CASE-4)
    python demo.py --interactive   # enter your own payment + remittance live
    python demo.py --step          # pause after each decision (press Enter)
    python demo.py --rl            # show raw Gymnasium transitions (obs/reward/done)
    python demo.py --case CASE-3
    python demo.py --all           # run every scenario + a summary table
    python demo.py --no-color      # disable ANSI colors

Flags combine, e.g.  python demo.py --interactive --step
"""

from __future__ import annotations

import argparse
import os
import sys

from environment.cash_application_env import ACTION_NAMES, CashApplicationEnv
from environment.models import cents_to_str
from environment.rule_based_agent import RuleBasedAgent
from environment.scenarios import SCENARIO_IDS, build_manual_scenario, build_scenario

# --- Color handling ---------------------------------------------------------
_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "blue": "\033[34m",
}
USE_COLOR = True


def c(text: str, *styles: str) -> str:
    if not USE_COLOR or not styles:
        return text
    prefix = "".join(_COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{_COLORS['reset']}"


BAR = "=" * 70
SUB = "-" * 70

CATEGORY_LABEL = {
    "remittance_match": "Matched from remittance",
    "ar_auto_match": "Auto-matched from AR",
    "missing_ar": "Missing AR",
    "ambiguous_ar": "Ambiguous AR",
    "no_evidence": "No usable evidence",
    "misapplied": "Misapplied",
    "pending": "Pending",
}

OUTCOME_LABEL = {
    "correct_allocated": "CORRECT — cash allocated per evidence",
    "correct_escalated": "CORRECT — escalated to human reviewer",
    "correct_unapplied": "CORRECT — cash left unapplied",
    "incorrect_allocated": "INCORRECT — allocation not supported by evidence",
    "incorrect_unapplied": "INCORRECT — cash left unapplied when it was applicable",
    "escalated": "ESCALATED — safe fallback on a determinable case",
    "truncated": "TRUNCATED — step limit reached without resolution",
}


def _outcome_styles(outcome: str) -> tuple[str, ...]:
    if outcome.startswith("correct"):
        return ("bold", "green")
    if outcome.startswith("incorrect"):
        return ("bold", "red")
    return ("bold", "yellow")  # escalated / truncated on a determinable case


def _fmt_refs(refs: list[str]) -> str:
    return ", ".join(refs) if refs else "(none)"


def _pause(step_mode: bool) -> None:
    """In step mode, wait for the user to press Enter before continuing."""
    if not step_mode:
        return
    try:
        input(c("        (press Enter to continue)", "dim"))
    except EOFError:
        # Non-interactive stdin (e.g. piped): just carry on.
        pass


def print_scenario_menu() -> None:
    print(c(BAR, "blue"))
    print(c("AVAILABLE SCENARIOS", "bold"))
    print(SUB)
    for cid in SCENARIO_IDS:
        case = build_scenario(cid)
        print(f"  {c(cid, 'bold')}  {case.title:<22}  {c(case.description, 'dim')}")
    print(c(BAR, "blue"))
    print()


# Short names for each slot of the fixed-size observation vector.
OBS_LEGEND = [
    "pay$", "unapplied$", "n_inv", "outstanding$", "investigated",
    "remit_visible", "remit_refs", "reliable", "n_cand",
    "inv0$", "inv1$", "inv2$", "steps",
    "ar_open", "ar_in_window", "ar_searched", "ar_matches",
]


def _fmt_obs(obs) -> str:
    return "[" + ", ".join(f"{v:g}" for v in obs.tolist()) + "]"


def demo_case(case_id: str, step_mode: bool = False, rl_mode: bool = False) -> dict:
    """Run one of the predefined registry scenarios."""
    return run_demo(
        CashApplicationEnv(case_id=case_id), step_mode=step_mode, rl_mode=rl_mode
    )


def run_demo(
    env: CashApplicationEnv, step_mode: bool = False, rl_mode: bool = False
) -> dict:
    agent = RuleBasedAgent()

    obs, info = env.reset(seed=0)
    case = env.case

    print(c(BAR, "blue"))
    print(c(f"SCENARIO {case.case_id}: {case.title}", "bold", "cyan"))
    print(SUB)
    print(f"  Customer         : {case.customer.name} ({case.customer.customer_id})")
    print(f"  Incoming payment : {case.payment.payment_id}  "
          f"{c(cents_to_str(case.payment.amount_cents), 'bold')}")
    print()
    print("  Open invoices:")
    for inv in case.invoices:
        print(f"    - {inv.invoice_id}: {cents_to_str(inv.amount_cents)} outstanding")
    print()
    print("  Remittance evidence available to agent:")
    if info["remittance_visible"]:
        print(f"    references: {_fmt_refs(info['remittance_references'])} "
              f"(reliable={info['remittance_reliable']})")
    else:
        print(f"    {c('HIDDEN — must investigate to reveal', 'yellow')}")

    if rl_mode:
        print(SUB)
        print(c("  Gymnasium environment (raw RL view):", "bold"))
        print(f"    action_space      : {env.action_space}")
        print(f"    observation_space : {env.observation_space}")
        print(f"    {c('obs legend', 'dim')}: "
              f"{c(', '.join(OBS_LEGEND), 'dim')}")
        print(f"    {c('reset obs', 'cyan')}  : {_fmt_obs(obs)}")
    print(SUB)

    # Drive the agent step by step so we can narrate each decision.
    print(c("  Agent decisions:", "bold"))
    _pause(step_mode)
    step_no = 0
    terminated = truncated = False
    while not (terminated or truncated):
        step_no += 1
        action = agent.act(env)
        prev_visible = info["remittance_visible"]
        obs, reward, terminated, truncated, info = env.step(action)
        name = ACTION_NAMES.get(action, f"ACTION_{action}")
        reward_str = c(f"reward={reward:+.2f}", "green" if reward >= 0 else "red")
        print(f"    [{step_no}] {c(name, 'bold'):<31} {reward_str}  "
              f"({info['event']})")

        # Narrate newly revealed evidence.
        if not prev_visible and info["remittance_visible"]:
            revealed = (f"        -> revealed remittance references: "
                        f"{_fmt_refs(info['remittance_references'])} "
                        f"(reliable={info['remittance_reliable']})")
            print(c(revealed, "cyan"))

        if rl_mode:
            transition = (
                f"        {c('step()', 'dim')} action={action} "
                f"reward={reward:+.2f} terminated={terminated} "
                f"truncated={truncated}"
            )
            print(transition)
            print(f"        {c('next obs', 'cyan')}: {_fmt_obs(obs)}")
        _pause(step_mode)

    print(SUB)
    print("  Final invoice balances:")
    for inv_id, bal in info["invoice_balances"].items():
        if bal == 0:
            status = c("CLOSED", "green")
        else:
            status = c("open", "yellow")
        print(f"    - {inv_id}: {cents_to_str(bal)} outstanding  [{status}]")
    print(f"  Unapplied payment amount : {cents_to_str(info['unapplied_cents'])}")
    print(SUB)
    outcome = info["outcome"]
    label = OUTCOME_LABEL.get(outcome, outcome)
    print(f"  Reconciliation outcome   : {c(label, *_outcome_styles(outcome))}")
    category = info.get("resolution_category", "pending")
    print(f"  Resolution category      : {c(CATEGORY_LABEL.get(category, category), 'bold')}")
    print("  Accounting checks:")
    conservation = env.ledger.check_conservation()
    no_negative = env.ledger.check_no_negative_balances()
    print(f"    - conservation (applied + unapplied == payment): "
          f"{c(str(conservation), 'green' if conservation else 'red')}")
    print(f"    - no negative balances                         : "
          f"{c(str(no_negative), 'green' if no_negative else 'red')}")
    print(c(BAR, "blue"))
    print()

    return {
        "case_id": case.case_id,
        "title": case.title,
        "outcome": outcome,
        "label": label,
        "accounting_ok": conservation and no_negative,
    }


def print_summary_table(results: list[dict]) -> None:
    print(c(BAR, "blue"))
    print(c("SUMMARY", "bold"))
    print(SUB)
    print(f"  {'Case':<8}{'Title':<22}{'Outcome':<40}{'Acct OK'}")
    for r in results:
        styles = _outcome_styles(r["outcome"])
        acct = c("yes", "green") if r["accounting_ok"] else c("NO", "red")
        print(f"  {r['case_id']:<8}{r['title']:<22}{c(r['label'], *styles):<40}{acct}")
    print(c(BAR, "blue"))
    print()


def _ask(prompt: str, default: str | None = None) -> str:
    """Prompt with an optional default shown in brackets."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(c(f"  {prompt}{suffix}: ", "bold")).strip()
    except EOFError:
        raw = ""
    return raw or (default or "")


def _ask_yes_no(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    ans = _ask(f"{prompt} (y/n)", d).lower()
    return ans.startswith("y")


def build_interactive_case():
    """Collect payment / invoices / remittance from the user at runtime."""
    print(c(BAR, "blue"))
    print(c("MANUAL CASE ENTRY", "bold"))
    print(c("  Enter the payment, open invoices, and remittance. The agent will "
            "then\n  investigate (if evidence is hidden) and resolve it.", "dim"))
    print(SUB)

    customer_name = _ask("Customer name", "Acme Corp")
    payment_dollars = _ask("Payment amount (dollars)", "2500")

    invoices: list[tuple[str, str]] = []
    try:
        n = int(_ask("How many open invoices", "2"))
    except ValueError:
        n = 2
    for i in range(1, max(n, 0) + 1):
        inv_id = _ask(f"  Invoice #{i} id", f"INV-{i:03d}")
        amount = _ask(f"  Invoice #{i} amount (dollars)", "1000")
        invoices.append((inv_id, amount))

    known_ids = [inv_id for inv_id, _ in invoices]
    print(c(f"  (known invoice ids: {', '.join(known_ids)})", "dim"))
    refs_raw = _ask("Remittance references the payment pays "
                    "(comma-separated, blank = none)", "")
    refs = [r.strip() for r in refs_raw.split(",") if r.strip()]

    reliable = _ask_yes_no("Is the remittance reliable?", bool(refs))
    hidden = _ask_yes_no("Start evidence hidden (so the agent must investigate)?",
                         True)

    case = build_manual_scenario(
        customer_name=customer_name,
        payment_dollars=payment_dollars,
        invoices=invoices,
        remittance_refs=refs,
        reliable=reliable,
        hidden=hidden,
        title="Manual Entry",
    )
    print(SUB)
    print()
    return case


def main() -> None:
    global USE_COLOR
    parser = argparse.ArgumentParser(description="Cash Application RL terminal demo")
    parser.add_argument("--case", default="CASE-4", help="scenario id (default CASE-4)")
    parser.add_argument("--all", action="store_true", help="run all scenarios")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="enter a payment and remittance by hand, then resolve it")
    parser.add_argument("--step", action="store_true",
                        help="pause after each agent decision (press Enter)")
    parser.add_argument("--rl", action="store_true",
                        help="show raw Gymnasium transitions (obs, action, reward, "
                             "terminated/truncated)")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = parser.parse_args()

    # Auto-disable color when not a TTY, when requested, or when NO_COLOR is set.
    USE_COLOR = (
        not args.no_color
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
    )

    print_scenario_menu()

    if args.interactive:
        case = build_interactive_case()
        run_demo(CashApplicationEnv(case=case), step_mode=args.step, rl_mode=args.rl)
    elif args.all:
        results = [
            demo_case(cid, step_mode=args.step, rl_mode=args.rl)
            for cid in SCENARIO_IDS
        ]
        print_summary_table(results)
    else:
        if args.case not in SCENARIO_IDS:
            raise SystemExit(f"Unknown case {args.case!r}. Choose from {SCENARIO_IDS}.")
        print("Default demonstration: the AMBIGUOUS payment case best shows why the\n"
              "agent must make sequential decisions (investigate, then allocate).\n")
        demo_case(args.case, step_mode=args.step, rl_mode=args.rl)


if __name__ == "__main__":
    main()
