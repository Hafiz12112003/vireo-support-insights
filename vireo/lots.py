"""Bad-lot early warning: (1) per-product weekly anomaly, (2) which manufacturing lots carry it, (3) what it costs."""
import numpy as np
import pandas as pd
from scipy.stats import poisson, binomtest

REPLACEMENT_LOGISTICS = 340   # Rs, policy s5
CONTACT_COST = 290            # Rs blended, policy s4


def weekly_alerts(t, o, event_col, window_days=120, trail=10, skip=2, min_events=6,
                  min_ratio=1.6, p_thr=1e-3, min_orders=20):
    """Flag a product-week when events (per recent order) are far above its own trailing baseline.
    Uses only data up to that week, so it can be backtested honestly."""
    weeks = pd.date_range(t.week.min() + pd.Timedelta(weeks=skip + trail), t.week.max(), freq="W-MON")
    rows = []
    for sku, ts in t.groupby("product_sku"):
        os_ = o[o.sku == sku]
        by_week = ts.groupby("week")[event_col].sum()

        def cnt(wk):
            exp = ((os_.order_date < wk + pd.Timedelta(days=7)) &
                   (os_.order_date >= wk - pd.Timedelta(days=window_days))).sum()
            return by_week.get(wk, 0), exp

        for w in weeks:
            ev0, exp0 = cnt(w)
            base = [cnt(w - pd.Timedelta(weeks=k)) for k in range(skip, skip + trail)]
            be, bx = sum(b[0] for b in base), sum(b[1] for b in base)
            if bx == 0 or exp0 < min_orders:
                continue
            lam = max(be / bx, 0.5 / bx) * exp0
            pv = poisson.sf(ev0 - 1, lam)
            if ev0 >= min_events and ev0 / lam >= min_ratio and pv < p_thr:
                rows.append(dict(product_sku=sku, week=w, events=int(ev0), expected=round(lam, 1), p_value=pv))
    return pd.DataFrame(rows, columns=["product_sku", "week", "events", "expected", "p_value"])


def lot_table(t, o, sku, as_of=None, min_orders=50):
    """Replacements per order by production lot, vs the product's median lot."""
    tt = t[(t.product_sku == sku) & t.lot_base.notna()]
    oo = o[o.sku == sku].copy()
    oo["lot_base"] = oo.lot_code.str.rsplit("-", n=1).str[0]
    if as_of is not None:
        tt = tt[tt.created_at < as_of]
        oo = oo[oo.order_date < as_of]
    g = oo.groupby("lot_base").size().rename("orders").to_frame()
    g["tickets"] = tt.groupby("lot_base").size()
    g["replacements"] = tt.groupby("lot_base").replaced.sum()
    g[["tickets", "replacements"]] = g[["tickets", "replacements"]].fillna(0).astype(int)
    g = g[g.orders >= min_orders].copy()
    g["repl_per_100_orders"] = 100 * g.replacements / g.orders
    base = g.repl_per_100_orders.median()
    g["lift_vs_median"] = g.repl_per_100_orders / max(base, 1e-9)
    # one-sided binomial test against the median lot's rate
    p0 = base / 100
    g["p_value"] = [binomtest(int(r), int(n), p0, alternative="greater").pvalue
                    for r, n in zip(g.replacements, g.orders)]
    g["quarantine_candidate"] = (g.lift_vs_median >= 2) & (g.p_value < 1e-4)
    return g.sort_index(), base


def cost_of_flagged_lots(t, o, p, sku, flagged, base_rate_per_100):
    """Excess replacements + excess contacts on flagged lots versus the product's normal lots."""
    unit = p.set_index("sku").unit_cost_inr[sku]
    rep_cost = unit + REPLACEMENT_LOGISTICS
    tt = t[(t.product_sku == sku) & t.lot_base.notna()]
    oo = o[o.sku == sku].copy(); oo["lot_base"] = oo.lot_code.str.rsplit("-", n=1).str[0]
    f_orders = oo[oo.lot_base.isin(flagged)]
    f_t = tt[tt.lot_base.isin(flagged)]
    n_orders = len(f_orders)
    excess_rep = max(f_t.replaced.sum() - n_orders * base_rate_per_100 / 100, 0)
    # tickets per order on normal lots (same product, lots with >=50 orders)
    normal = oo[~oo.lot_base.isin(flagged)]
    normal_lots = normal.groupby("lot_base").size(); normal_lots = normal_lots[normal_lots >= 50].index
    tpo_norm = tt[tt.lot_base.isin(normal_lots)].shape[0] / max(normal[normal.lot_base.isin(normal_lots)].shape[0], 1)
    excess_tix = max(len(f_t) - n_orders * tpo_norm, 0)
    return dict(orders=n_orders, rep_cost_each=rep_cost, excess_replacements=excess_rep,
                excess_tickets=excess_tix, replacement_rs=excess_rep * rep_cost,
                contact_rs=excess_tix * CONTACT_COST,
                total_rs=excess_rep * rep_cost + excess_tix * CONTACT_COST,
                order_value_rs=f_orders.order_value_inr.sum(), tickets_per_order_normal=tpo_norm)
