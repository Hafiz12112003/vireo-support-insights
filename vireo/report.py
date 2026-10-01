"""Static HTML dashboard (charts embedded, no server, no internet needed)."""
import base64, io, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_BAD, C_OK, C_GREY = "#c0392b", "#2c6fbb", "#888"


def _img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def fmt_ht(m, tier):
    if pd.isna(m): return "-"
    return f"{m/1440:.1f} d" if (tier == 2 or m > 600) else f"{m:.0f} min"


def chart_csat(t, flagged):
    d = t.dropna(subset=["csat_score"])
    m_all = d.groupby("month").csat_score.mean()
    m_ex = d[~d.lot_base.isin(flagged)].groupby("month").csat_score.mean()
    vol = t.groupby("month").size()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(vol.index, vol.values, color="#e8edf3", label="tickets (right axis is CSAT)")
    ax.set_yticks([]); ax2 = ax.twinx()
    ax2.plot(m_all.index, m_all.values, color=C_BAD, marker="o", label="CSAT, all tickets")
    ax2.plot(m_ex.index, m_ex.values, color=C_OK, marker="o", label="CSAT, excluding the 3 bad lots")
    ax2.set_ylim(2.6, 3.9); ax2.legend(loc="lower left", fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.set_title("CSAT slid because of one product batch, not a general decline", fontsize=10, loc="left")
    return _img(fig)


def chart_alert(t, o, sku, alert_week, flagged):
    ts = t[(t.product_sku == sku)]
    w = ts.groupby("week").replaced.sum()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(w.index, w.values, width=5, color=C_OK)
    if alert_week is not None:
        ax.axvline(alert_week, color=C_BAD, ls="--")
        ax.text(alert_week, ax.get_ylim()[1] * .92, "  early-warning fires", color=C_BAD, fontsize=8)
    ax.set_title(f"{sku}: replacements issued per week", fontsize=10, loc="left")
    ax.tick_params(axis="x", labelsize=8)
    return _img(fig)


def chart_agents(sc):
    s = sc[sc.responses >= 1].sort_values("gap")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    y = np.arange(len(s))
    col = [C_BAD if z <= -2.5 else (C_GREY if t2 == 2 else C_OK) for z, t2 in zip(s.z, s.tier)]
    ax.errorbar(s.gap, y, xerr=1.96 * s.gap_se, fmt="none", ecolor="#bbb", lw=1)
    ax.scatter(s.gap, y, c=col, zorder=3, s=22)
    ax.set_yticks(y); ax.set_yticklabels([f"{n} ({a})" for n, a in zip(s["name"], s.agent_id)], fontsize=7)
    ax.axvline(0, color="#444", lw=.8)
    ax.set_xlabel("CSAT vs. what you'd expect for the tickets this agent actually got (95% interval)")
    ax.set_title("Adjusted agent gap. Red = persists after adjustment; grey = Tier 2 (not comparable)", fontsize=10, loc="left")
    return _img(fig)


def table(df, cols, heads, fmts=None):
    fmts = fmts or {}
    h = "<table><tr>" + "".join(f"<th>{c}</th>" for c in heads) + "</tr>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td>{html.escape(str(fmts.get(c, lambda v: v)(r[c])))}</td>" for c in cols) + "</tr>"
    return h + "</table>"


def build(path, ctx):
    t, o, sc, lt, cost, al, log, mm = (ctx[k] for k in ["t", "o", "sc", "lt", "cost", "alerts", "log", "text_metrics"])
    flagged, sku = ctx["flagged"], ctx["sku"]
    alert_week = al[al.product_sku == sku].week.min() if len(al[al.product_sku == sku]) else None
    s = sc.sort_values("csat_raw").copy()
    s["Handle (median)"] = [fmt_ht(m, ti) for m, ti in zip(s.handle_median_min, s.tier)]
    s["gap_txt"] = [f"{g:+.2f} ± {1.96*se:.2f}" for g, se in zip(s.gap, s.gap_se)]
    rows = s[["agent_id", "name", "team", "tier", "responses", "csat_raw", "expected", "gap_txt", "Handle (median)",
              "sla_breach_rate", "share_flagged_lot", "verdict"]]
    agent_html = table(rows, list(rows.columns),
                       ["Agent", "Name", "Team", "Tier", "Survey n", "CSAT raw", "CSAT expected", "Gap (95%)",
                        "Handle (median)", "SLA breach", "Bad-lot share", "What to do"],
                       {"csat_raw": lambda v: f"{v:.2f}", "expected": lambda v: f"{v:.2f}",
                        "sla_breach_rate": lambda v: f"{v:.0%}", "share_flagged_lot": lambda v: f"{v:.0%}"})
    lot_html = table(lt.reset_index(), ["lot_base", "orders", "replacements", "repl_per_100_orders", "lift_vs_median", "quarantine_candidate"],
                     ["Lot", "Orders", "Replacements", "Repl per 100 orders", "x median lot", "Quarantine?"],
                     {"repl_per_100_orders": lambda v: f"{v:.1f}", "lift_vs_median": lambda v: f"{v:.1f}x"})
    css = ("body{font-family:system-ui,Segoe UI,Arial;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;font-size:12px;width:100%}th,td{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left}"
           "th{background:#f3f5f8}.k{display:flex;gap:12px;flex-wrap:wrap}.k div{background:#f3f5f8;padding:10px 14px;border-radius:8px}"
           ".k b{font-size:20px;display:block}h2{margin-top:34px}.note{color:#555;font-size:13px}img{max-width:100%}")
    first = (f"{alert_week.date()}" if alert_week is not None else "n/a")
    doc = f"""<!doctype html><meta charset=utf-8><title>Vireo support insights</title><style>{css}</style>
<h1>Vireo Audio: support dashboard</h1>
<p class=note>Jan 2025 to Jun 2026, {len(t):,} tickets. CSAT blanks are excluded (not zero). Handle time = first response to resolution, legacy timestamps corrected to IST.</p>
<div class=k><div><b>{len(flagged)} lots</b>{', '.join(flagged)} (Pulse 2)</div>
<div><b>Rs {cost['total_rs']/1e5:.1f} L</b>excess cost (replacements + extra contacts)</div>
<div><b>{cost['excess_replacements']:.0f}</b>excess replacements at Rs {cost['rep_cost_each']:,.0f} each</div>
<div><b>{first}</b>first week the early warning would have fired</div></div>
<h2>1. Why CSAT fell</h2>{chart_csat(t, flagged)}
<h2>2. Early warning and the lots behind it</h2>{chart_alert(t, o, sku, alert_week, flagged)}{lot_html}
<p class=note>Symptom text model (Pulse 2, customer's first message only, 5-fold CV): AUC {mm['auc']:.2f}, precision {mm['precision']:.0%}, recall {mm['recall']:.0%}. It is a secondary signal, not a decision-maker.</p>
<h2>3. Agents (what was asked for, with the context it needs)</h2>
<p class=note>Raw CSAT and handle time are shown as requested. "Expected" is the CSAT for the mix of tickets that agent got (category, channel, priority, product, month, team, bad lot). Tier 2 handle time is in days by design.</p>
{chart_agents(sc)}{agent_html}
<h2>4. Data fixes applied</h2><ul>{''.join(f'<li>{html.escape(l)}</li>' for l in log)}</ul>"""
    open(path, "w", encoding="utf-8").write(doc)
