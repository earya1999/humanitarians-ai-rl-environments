"""Local browser demo for the Cash Application RL environment.

A lightweight presentation aid (NOT part of the core research milestone). It
serves a small web page from Python's standard library only -- no Flask,
Streamlit, or other dependencies -- so it stays minimal and reproducible.

You fill in a payment, open invoices, and remittance in the browser; the page
then shows the rule-based agent investigate (if evidence is hidden), allocate
or escalate, and reports the final balances and accounting checks. The five
predefined scenarios are available as one-click presets.

Usage:
    python web_demo.py              # serve on http://localhost:8000 and open it
    python web_demo.py --port 9000  # choose a port
    python web_demo.py --no-open    # do not auto-open the browser

The browser used is whatever your OS opens for http URLs; on macOS this is
typically Safari if it is your default browser.
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from environment.cash_application_env import (
    ACTION_ALLOCATE_BASE,
    ACTION_ESCALATE,
    ACTION_INVESTIGATE,
    ACTION_LEAVE_UNAPPLIED,
    ACTION_NAMES,
    ACTION_SEARCH_AR,
    CashApplicationEnv,
)
from environment.models import cents_to_str
from environment.rule_based_agent import RuleBasedAgent
from environment.scenarios import (
    SCENARIO_IDS,
    build_manual_scenario,
    build_scenario,
)

OBS_LEGEND = [
    "pay$", "unapplied$", "n_inv", "outstanding$", "investigated",
    "remit_visible", "remit_refs", "reliable", "n_cand",
    "inv0$", "inv1$", "inv2$", "steps",
    "ar_open", "ar_in_window", "ar_searched", "ar_matches",
]


def _fmt_obs(obs) -> str:
    return "[" + ", ".join(f"{v:g}" for v in obs.tolist()) + "]"


def _fmt_date(d) -> str:
    return d.strftime("%b %d, %Y") if d is not None else "—"


OUTCOME_LABEL = {
    "correct_allocated": "Correct",
    "correct_escalated": "Correct",
    "correct_unapplied": "Correct",
    "incorrect_allocated": "Incorrect",
    "incorrect_unapplied": "Incorrect",
    "escalated": "Escalated",
    "truncated": "Unresolved",
}

# How the case was resolved (informational category).
CATEGORY_LABEL = {
    "remittance_match": "Matched from remittance",
    "ar_auto_match": "Auto-matched from AR",
    "missing_ar": "Missing AR",
    "ambiguous_ar": "Ambiguous AR",
    "no_evidence": "No usable evidence",
    "misapplied": "Misapplied",
    "pending": "Pending",
}
CATEGORY_HELP = {
    "remittance_match": "The remittance named the invoices, so the agent applied the payment directly.",
    "ar_auto_match": "No remittance, but the open AR had exactly one reconciling combination.",
    "missing_ar": "The paying invoice is not in the loaded AR — either nothing reconciles, or the remittance names an invoice we do not have loaded — so the agent escalated instead of guessing.",
    "ambiguous_ar": "The open AR had more than one reconciling combination, so the agent could not safely choose.",
    "no_evidence": "No reliable evidence was available to identify the invoices.",
    "misapplied": "The payment was applied without supporting evidence.",
    "pending": "",
}

# Plain-English explanation of why each outcome is right or wrong.
OUTCOME_PLAIN = {
    "correct_allocated": "The agent applied the payment to the invoices the "
                         "evidence supported. That is the right answer.",
    "correct_escalated": "There was no reliable evidence, so the agent sent the "
                         "case to a person instead of guessing. That is the safe, "
                         "correct choice here.",
    "correct_unapplied": "There was no reliable evidence, so the agent left the "
                         "cash unapplied instead of guessing. That is correct.",
    "incorrect_allocated": "The agent applied the payment without evidence to "
                           "support it — a wrong guess.",
    "incorrect_unapplied": "The evidence supported applying the payment, but the "
                           "agent left it unapplied.",
    "escalated": "The agent escalated to a human. Safe, but this case could have "
                 "been resolved automatically.",
    "truncated": "The agent ran out of steps before resolving the case.",
}

PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; background: #f5f6f8; color: #1d1d1f; }
.wrap { max-width: 920px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #6b7280; margin: 0 0 20px; font-size: 14px; }
.card { background: #fff; border: 1px solid #e3e5e8; border-radius: 12px;
        padding: 20px; margin-bottom: 20px; }
label { display: block; font-size: 13px; font-weight: 600; margin: 10px 0 4px; }
input[type=text], input[type=number] { width: 100%; padding: 8px 10px;
        border: 1px solid #c9ccd1; border-radius: 8px; font-size: 14px; }
.row { display: flex; gap: 12px; }
.row > div { flex: 1; }
.inv-row { display: flex; gap: 12px; align-items: end; margin-bottom: 8px; }
.inv-row > div { flex: 1; }
.check { display: inline-flex; align-items: center; gap: 8px; margin: 14px 18px 0 0;
         font-size: 14px; font-weight: 600; }
button { background: #0b63f6; color: #fff; border: 0; border-radius: 8px;
         padding: 10px 18px; font-size: 15px; font-weight: 600; cursor: pointer;
         margin-top: 16px; }
button.secondary { background: #e7eaef; color: #1d1d1f; }
.presets a { display: inline-block; margin: 4px 8px 4px 0; padding: 6px 12px;
         background: #eef1f5; border-radius: 999px; text-decoration: none;
         color: #0b63f6; font-size: 13px; }
.hint { color: #6b7280; font-size: 12px; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eef0f2; }
.step { font-family: ui-monospace, Menlo, monospace; font-size: 13px;
        padding: 6px 0; border-bottom: 1px solid #f0f1f3; }
.reveal { color: #0a7ea4; }
.lead { font-size: 16px; line-height: 1.5; background: #eef4ff;
        border: 1px solid #d6e4ff; border-radius: 10px; padding: 14px 16px;
        margin-bottom: 20px; }
h3 { margin: 0 0 12px; font-size: 16px; }
h4 { margin: 14px 0 6px; font-size: 14px; }
.storystep { display: flex; gap: 12px; align-items: flex-start;
        padding: 12px 0; border-bottom: 1px solid #f0f1f3; }
.storystep:last-child { border-bottom: 0; }
.storystep .sub { margin-top: 3px; line-height: 1.45; }
.stepnum { flex: 0 0 26px; width: 26px; height: 26px; border-radius: 50%;
        background: #0b63f6; color: #fff; font-weight: 700; font-size: 13px;
        display: flex; align-items: center; justify-content: center; }
details summary { cursor: pointer; font-size: 14px; }
.badge { display: inline-block; padding: 6px 12px; border-radius: 8px;
         font-weight: 700; font-size: 14px; }
.ok { background: #e6f6ec; color: #137a3b; }
.bad { background: #fdeaea; color: #b4231f; }
.warn { background: #fdf3e1; color: #9a6400; }
.mono { font-family: ui-monospace, Menlo, monospace; }
"""


def _page(body: str) -> bytes:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Cash Application RL — Demo</title><style>{PAGE_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    ).encode("utf-8")


# How many invoice rows the form shows (shared AR has 10).
MAX_FORM_INVOICES = 12


def _cents_to_input(cents: int) -> str:
    """Plain number for an HTML number input (no $ / commas)."""
    if cents % 100 == 0:
        return str(cents // 100)
    return f"{cents / 100:.2f}"


def _iso_date(d) -> str:
    return d.isoformat() if d is not None else ""


def _checked(flag: bool) -> str:
    return " checked" if flag else ""


def _invoice_row(
    i: int, inv_id: str = "", amount: str = "", inv_date: str = ""
) -> str:
    return (
        f"<div class='inv-row'>"
        f"<div><label>Invoice #{i} id</label>"
        f"<input type='text' name='inv_id_{i}' value='{html.escape(inv_id)}' "
        f"placeholder='INV-{i:03d}'></div>"
        f"<div><label>Amount ($)</label>"
        f"<input type='number' step='0.01' name='inv_amt_{i}' "
        f"value='{html.escape(amount)}' placeholder='1000'></div>"
        f"<div><label>Invoice date</label>"
        f"<input type='date' name='inv_date_{i}' value='{html.escape(inv_date)}'></div>"
        f"</div>"
    )


def form_page(
    message: str = "",
    *,
    case=None,
    show_rl: bool = True,
) -> bytes:
    """Render the entry form, optionally pre-filled from a scenario ``case``."""
    # Presets stay on this page and fill the fields; the user then clicks Run.
    presets = " ".join(
        f"<a href='/?preset={cid}&rl=1'>{cid}: {build_scenario(cid).title}</a>"
        for cid in SCENARIO_IDS
    )

    if case is not None:
        customer = case.customer.name
        payment = _cents_to_input(case.payment.amount_cents)
        pay_date = _iso_date(case.payment.payment_date)
        refs = ", ".join(case.remittance.referenced_invoice_ids)
        reliable = case.remittance.reliable
        hidden = case.remittance.hidden
        inv_values = [
            (
                inv.invoice_id,
                _cents_to_input(inv.amount_cents),
                _iso_date(inv.invoice_date),
            )
            for inv in case.invoices
        ]
    else:
        customer = "Acme Corp"
        payment = "2500"
        pay_date = ""
        refs = ""
        reliable = True
        hidden = True
        inv_values = []

    n_rows = max(MAX_FORM_INVOICES, len(inv_values))
    invoice_rows = []
    for i in range(1, n_rows + 1):
        if i <= len(inv_values):
            inv_id, amount, inv_date = inv_values[i - 1]
            invoice_rows.append(_invoice_row(i, inv_id, amount, inv_date))
        else:
            invoice_rows.append(_invoice_row(i))
    invoice_rows_html = "".join(invoice_rows)

    body = f"""
    <h1>Cash Application — live demo</h1>
    <p class='lead'>A customer sends money. The job is to figure out which
    invoices that money pays off. Sometimes the paperwork (the "remittance")
    that says which invoices to pay is missing, so the agent has to investigate
    first, and if there is no reliable evidence it should ask a human rather than
    guess. Pick a preset to fill the form, review or edit the fields, then run
    the agent.</p>
    {message}
    <div class='card'>
      <div class='presets'><strong style='font-size:13px'>Try a preset
        (fills the form below):</strong><br>{presets}</div>
    </div>
    <form method='post' action='/run'>
      <div class='card'>
        <div class='row'>
          <div><label>Customer</label>
            <input type='text' name='customer' value='{html.escape(customer)}'></div>
          <div><label>Payment amount ($)</label>
            <input type='number' step='0.01' name='payment' value='{html.escape(payment)}'></div>
          <div><label>Payment date</label>
            <input type='date' name='pay_date' value='{html.escape(pay_date)}'></div>
        </div>
        <label style='margin-top:18px'>Open invoices</label>
        {invoice_rows_html}
        <div class='hint'>Leave rows blank to skip them. Dates are optional — a
        blank invoice date defaults to the payment date (so it is inside the
        &plusmn;7-day AR window); a blank payment date defaults to today.</div>
        <label style='margin-top:18px'>Remittance references (which invoices the payment pays)</label>
        <input type='text' name='refs' value='{html.escape(refs)}'
               placeholder='e.g. INV-001, INV-002'>
        <div class='hint'>Comma-separated invoice ids. Leave blank for "no reliable remittance".</div>
        <div>
          <span class='check'><input type='checkbox' name='reliable'{_checked(reliable)}> Remittance is reliable</span>
          <span class='check'><input type='checkbox' name='hidden'{_checked(hidden)}> Start evidence hidden (agent must investigate)</span>
          <span class='check'><input type='checkbox' name='rl'{_checked(show_rl)}> Show RL transitions (obs / reward / done)</span>
        </div>
        <button type='submit'>Run the agent</button>
      </div>
    </form>
    """
    return _page(body)


def _run_and_render(case, heading: str, show_rl: bool = False) -> bytes:
    env = CashApplicationEnv(case=case)
    agent = RuleBasedAgent()

    # Drive the episode manually so we can capture per-step observations.
    obs, info = env.reset(seed=0)
    reset_obs = obs
    transitions = []
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(env)
        prev_visible = info["remittance_visible"]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        revealed = None
        if not prev_visible and info["remittance_visible"]:
            refs = info["remittance_references"]
            revealed = (", ".join(refs) if refs else "(none)",
                        info["remittance_reliable"])
        allocs = [(a.invoice_id, a.amount_cents) for a in env.ledger.allocations]
        transitions.append({
            "action": action,
            "reward": reward,
            "event": info["event"],
            "terminated": terminated,
            "truncated": truncated,
            "obs": obs,
            "revealed": revealed,
            "allocations": allocs,
            "ar_num_matches": info.get("ar_num_matches", 0),
        })
    outcome = info["outcome"]

    # ---- Plain-English step narration -------------------------------------
    def _plain_step(num: int, t: dict) -> str:
        action = t["action"]
        if action == ACTION_INVESTIGATE:
            title = "Investigated the payment"
            detail = ("The remittance note was missing, so the agent looked it up "
                      "to see which invoices this payment is meant for.")
            if t["revealed"] is not None:
                refs_txt, reliable = t["revealed"]
                if refs_txt == "(none)" or not reliable:
                    detail += (" <b>Nothing reliable was found</b> — there is no "
                               "trustworthy note of which invoices to pay.")
                else:
                    detail += f" <b>Found it:</b> the payment is for {refs_txt}."
        elif action == ACTION_SEARCH_AR:
            title = "Searched the open accounts receivable (AR)"
            n = t["ar_num_matches"]
            detail = ("With no remittance, the agent searched this customer's open "
                      "invoices dated within &plusmn;7 days of the payment for a "
                      "combination whose amounts add up to the payment. ")
            if n == 0:
                detail += "<b>No combination reconciled</b> to the payment amount."
            elif n == 1:
                detail += "<b>Found exactly one match</b> — a confident auto-match."
            else:
                detail += (f"<b>Found {n} possible matches</b> — the evidence is "
                           "ambiguous, so applying any one would be a guess.")
        elif ACTION_ALLOCATE_BASE <= action < ACTION_LEAVE_UNAPPLIED:
            title = "Applied the payment to invoices"
            parts = [f"{inv} ({cents_to_str(amt)})" for inv, amt in t["allocations"]]
            joined = " and ".join(parts) if parts else "the matching invoices"
            detail = f"The agent matched the payment to {joined}, as the evidence supported."
        elif action == ACTION_LEAVE_UNAPPLIED:
            title = "Left the cash unapplied"
            detail = ("The agent chose not to apply the payment to any invoice.")
        elif action == ACTION_ESCALATE:
            title = "Escalated to a human reviewer"
            detail = ("Without reliable evidence, the agent handed the case to a "
                      "person rather than risk a wrong match.")
        else:
            title = ACTION_NAMES.get(action, f"Action {action}")
            detail = t["event"]
        return (f"<div class='storystep'><div class='stepnum'>{num}</div>"
                f"<div><b>{title}</b><div class='sub'>{detail}</div></div></div>")

    steps_block = "".join(_plain_step(i, t) for i, t in enumerate(transitions, 1))

    # ---- The situation at the start ---------------------------------------
    pay_date = case.payment.payment_date
    show_window = pay_date is not None
    win_hdr = "<th>Within &plusmn;7 days</th>" if show_window else ""
    inv_rows = []
    for inv in case.invoices:
        win_cell = ""
        if show_window:
            within = (inv.invoice_date is not None
                      and abs((inv.invoice_date - pay_date).days) <= 7)
            win_cell = ("<td><span class='badge ok'>yes</span></td>" if within
                        else "<td><span class='sub'>no</span></td>")
        inv_rows.append(
            f"<tr><td class='mono'>{html.escape(inv.invoice_id)}</td>"
            f"<td class='mono'>{_fmt_date(inv.invoice_date)}</td>"
            f"<td class='mono'>{cents_to_str(inv.amount_cents)}</td>{win_cell}</tr>"
        )
    inv_start = "".join(inv_rows)
    if case.remittance.hidden:
        evidence_start = ("<span class='badge warn'>Hidden</span> "
                          "The note saying which invoices to pay is missing — "
                          "the agent must investigate to find it.")
    elif case.remittance.referenced_invoice_ids and case.remittance.reliable:
        refs = ", ".join(case.remittance.referenced_invoice_ids)
        evidence_start = (f"<span class='badge ok'>Available</span> "
                          f"The payment is for {refs}.")
    else:
        evidence_start = ("<span class='badge warn'>None / unreliable</span> "
                          "There is no trustworthy note of which invoices to pay.")

    # ---- The result -------------------------------------------------------
    rows = []
    for inv_id, bal in info["invoice_balances"].items():
        status = ("<span class='badge ok'>Paid off</span>" if bal == 0
                  else "<span class='badge warn'>Still owed</span>")
        rows.append(
            f"<tr><td class='mono'>{html.escape(inv_id)}</td>"
            f"<td class='mono'>{cents_to_str(bal)}</td><td>{status}</td></tr>"
        )
    balances_table = "".join(rows)

    if outcome.startswith("correct"):
        badge_cls = "ok"
    elif outcome.startswith("incorrect"):
        badge_cls = "bad"
    else:
        badge_cls = "warn"
    label = OUTCOME_LABEL.get(outcome, outcome)
    outcome_plain = OUTCOME_PLAIN.get(outcome, "")

    category = info.get("resolution_category", "pending")
    cat_label = CATEGORY_LABEL.get(category, category)
    cat_help = CATEGORY_HELP.get(category, "")
    if category == "ar_auto_match":
        cat_cls = "ok"
    elif category in ("missing_ar", "ambiguous_ar"):
        cat_cls = "warn"
    elif category == "misapplied":
        cat_cls = "bad"
    else:
        cat_cls = "ok" if category == "remittance_match" else "warn"

    conservation = env.ledger.check_conservation()
    no_negative = env.ledger.check_no_negative_balances()
    books_ok = conservation and no_negative
    books_badge = ("<span class='badge ok'>Books balance</span>" if books_ok
                   else "<span class='badge bad'>Problem</span>")

    # ---- Top one-line summary ---------------------------------------------
    pay_str = cents_to_str(case.payment.amount_cents)
    n_inv = len(case.invoices)
    summary = (f"{html.escape(case.customer.name)} sent a payment of "
               f"<b>{pay_str}</b>. There {'is' if n_inv == 1 else 'are'} "
               f"<b>{n_inv}</b> open invoice{'s' if n_inv != 1 else ''}. "
               f"The agent needs to decide how to apply it.")

    # ---- Optional technical (RL) details ----------------------------------
    rl_block = ""
    if show_rl:
        obs_rows = "".join(
            f"<tr><td class='mono'>{html.escape(name)}</td>"
            f"<td class='mono'>{v:g}</td></tr>"
            for name, v in zip(OBS_LEGEND, reset_obs.tolist())
        )
        step_rows = "".join(
            f"<tr><td>{i}</td><td class='mono'>{t['action']} "
            f"({html.escape(ACTION_NAMES.get(t['action'], ''))})</td>"
            f"<td class='mono'>{t['reward']:+.2f}</td>"
            f"<td class='mono'>{t['terminated']}</td>"
            f"<td class='mono'>{html.escape(_fmt_obs(t['obs']))}</td></tr>"
            for i, t in enumerate(transitions, 1)
        )
        rl_block = f"""
        <div class='card'>
          <details>
            <summary><b>Technical details</b> — the Gymnasium reinforcement-learning view
              (for the research audience)</summary>
            <p class='sub' style='margin-top:10px'>This environment follows the
              standard Gymnasium API (<span class='mono'>action_space = Discrete(8)</span>,
              <span class='mono'>observation_space = Box(17,)</span>). The agent only
              ever sees the numbers below — never the hidden correct answer.</p>
            <h4>What the agent sees at the start (its observation)</h4>
            <table><tr><th>Field</th><th>Value</th></tr>{obs_rows}</table>
            <h4 style='margin-top:14px'>Each environment step</h4>
            <table><tr><th>#</th><th>action</th><th>reward</th><th>terminated</th>
              <th>next observation</th></tr>{step_rows}</table>
            <p class='sub' style='margin-top:10px'>Total reward:
              <span class='mono'>{total_reward:+.2f}</span>. Reward values are
              initial, untuned design choices.</p>
          </details>
        </div>"""

    body = f"""
    <h1>{html.escape(heading)}</h1>
    <p class='lead'>{summary}</p>

    <div class='card'>
      <h3>1. The situation</h3>
      <p><b>Customer:</b> {html.escape(case.customer.name)}
         (<span class='mono'>{html.escape(case.customer.customer_id)}</span>)</p>
      <p><b>Payment received:</b> <span class='mono'>{pay_str}</span>
         &nbsp;dated <span class='mono'>{_fmt_date(case.payment.payment_date)}</span></p>
      <p style='margin-bottom:6px'><b>Open invoices:</b></p>
      <table><tr><th>Invoice</th><th>Invoice date</th><th>Amount owed</th>{win_hdr}</tr>{inv_start}</table>
      <p style='margin-top:12px'><b>Evidence (remittance):</b> {evidence_start}</p>
    </div>

    <div class='card'>
      <h3>2. What the agent did</h3>
      {steps_block}
    </div>

    <div class='card'>
      <h3>3. The result</h3>
      <table><tr><th>Invoice</th><th>Still owed</th><th>Status</th></tr>
      {balances_table}</table>
      <p class='sub' style='margin-top:12px'>Money not applied to any invoice:
        <span class='mono'>{cents_to_str(info['unapplied_cents'])}</span></p>
      <p style='margin-top:14px'>
        <span class='badge {badge_cls}'>{html.escape(label)}</span>
        &nbsp;{html.escape(outcome_plain)}</p>
      <p style='margin-top:8px'>
        <span class='badge {cat_cls}'>{html.escape(cat_label)}</span>
        &nbsp;<span class='sub'>{html.escape(cat_help)}</span></p>
      <p style='margin-top:8px'>{books_badge}
        &nbsp;<span class='sub'>Every cent is accounted for and no invoice was
        over-paid.</span></p>
    </div>

    {rl_block}

    <a href='/'><button class='secondary'>&larr; Try another case</button></a>
    """
    return _page(body)


def _parse_amount(s: str) -> str:
    return (s or "").strip()


def _parse_date(s: str) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` date from an HTML date input, or None."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, content: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):  # keep the console quiet
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        def _fill_preset(cid: str) -> bytes:
            if cid not in SCENARIO_IDS:
                return form_page(
                    "<div class='card'><span class='badge bad'>Unknown "
                    f"preset: {html.escape(cid)}</span></div>"
                ), 404
            show_rl = qs.get("rl", ["1"])[0] in ("1", "true", "on")
            case = build_scenario(cid)
            msg = (
                f"<div class='card'><span class='badge ok'>Loaded {html.escape(cid)}: "
                f"{html.escape(case.title)}</span> &nbsp;Review or edit the fields "
                f"below, then click <b>Run the agent</b>.</div>"
            )
            return form_page(msg, case=case, show_rl=show_rl), 200

        if parsed.path == "/":
            cid = (qs.get("preset", [""])[0]).strip()
            if cid:
                content, status = _fill_preset(cid)
                self._send(content, status)
            else:
                self._send(form_page())
        elif parsed.path == "/preset":
            # Back-compat: old /preset?id=... links also fill the form now.
            cid = (qs.get("id", qs.get("preset", [""]))[0]).strip()
            content, status = _fill_preset(cid)
            self._send(content, status)
        else:
            self._send(form_page("<div class='card'>Not found.</div>"), 404)

    def do_POST(self):
        if urlparse(self.path).path != "/run":
            self._send(form_page(), 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw, keep_blank_values=True)

        def field(name, default=""):
            return form.get(name, [default])[0].strip()

        try:
            customer = field("customer", "Acme Corp") or "Acme Corp"
            payment = _parse_amount(field("payment", "0")) or "0"
            pay_date = _parse_date(field("pay_date"))
            invoices = []
            for i in range(1, MAX_FORM_INVOICES + 1):
                inv_id = field(f"inv_id_{i}")
                amt = _parse_amount(field(f"inv_amt_{i}"))
                inv_date = _parse_date(field(f"inv_date_{i}"))
                if inv_id and amt:
                    invoices.append((inv_id, amt, inv_date))
            refs = [r.strip() for r in field("refs").split(",") if r.strip()]
            reliable = "reliable" in form
            hidden = "hidden" in form
            show_rl = "rl" in form

            if not invoices:
                self._send(form_page(
                    "<div class='card'><span class='badge bad'>Please enter at "
                    "least one invoice.</span></div>"), 400)
                return

            case = build_manual_scenario(
                customer_name=customer,
                payment_dollars=payment,
                invoices=invoices,
                remittance_refs=refs,
                reliable=reliable,
                hidden=hidden,
                payment_date=pay_date,
                title="Manual Entry",
            )
            self._send(_run_and_render(case, "Manual case result", show_rl=show_rl))
        except Exception as exc:  # pragma: no cover - defensive for live demo
            msg = html.escape(str(exc))
            self._send(form_page(
                f"<div class='card'><span class='badge bad'>Error: {msg}</span></div>"),
                400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cash Application RL browser demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true",
                        help="do not auto-open the browser")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Cash Application RL browser demo running at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open and sys.platform == "darwin":
        try:
            subprocess.Popen(["open", url])
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.shutdown()


if __name__ == "__main__":
    main()
