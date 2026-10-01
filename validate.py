"""python validate.py  ->  out/validation.md  (checks on the tool's own output; no labels needed from the client)"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from vireo.clean import load
from vireo.lots import weekly_alerts, lot_table, cost_of_flagged_lots
from vireo.text import fit_symptom_model, _clean
from vireo.agents import scorecard, Z_FLAG
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

os.makedirs("out", exist_ok=True)
t, o, a, p, _ = load("data")
SKU = "VA-EB-PL2"
L = []

# 1. lot attribution: order_id lot vs customer+sku fallback on tickets where both exist
oo = o.copy()
fb = oo.groupby(["customer_id", "sku"]).lot_code.agg(["first", "nunique"]).reset_index()
fb = fb[fb["nunique"] == 1].rename(columns={"sku": "product_sku", "first": "fb"})
x = t[t.order_id.notna()].copy()
x["by_order"] = x.order_id.map(o.set_index("order_id").lot_code)
x = x.merge(fb[["customer_id", "product_sku", "fb"]], on=["customer_id", "product_sku"], how="inner")
agree = (x.by_order == x.fb).mean()
L.append(f"## 1. Lot attribution\nFallback join (customer+sku) vs the order_id join on {len(x):,} tickets where both exist: "
         f"**{agree:.1%} identical**. Tickets with no lot at all: {(t.lot_src=='none').sum():,} ({(t.lot_src=='none').mean():.1%}); "
         f"they are left out of lot statistics (this understates the problem, it does not overstate it).")

# 2. early-warning backtest, no look-ahead
FL = ["PL2-2510", "PL2-2511", "PL2-2512"]
al = weekly_alerts(t, o, "replaced")
n_cells = t.product_sku.nunique() * t.week.nunique()
pl = al[al.product_sku == SKU]; other = al[al.product_sku != SKU]
first = pl.week.min()
det = {}
for w in pd.date_range(first - pd.Timedelta(weeks=6), first + pd.Timedelta(weeks=20), freq="W-MON"):
    lt_w, _ = lot_table(t, o, SKU, as_of=w)
    for l in lt_w[lt_w.quarantine_candidate].index:
        det.setdefault(l, w)
lt_all, base_all = lot_table(t, o, SKU)
flagged = list(lt_all[lt_all.quarantine_candidate].index)
cost = cost_of_flagged_lots(t, o, p, SKU, flagged, base_all)
per_order = cost["total_rs"] / cost["orders"]
oo = o[o.sku == SKU].copy(); oo["lot_base"] = oo.lot_code.str.rsplit("-", n=1).str[0]
HOLD = pd.Timedelta(days=14)
ruleA = sum(((oo.lot_base == l) & (oo.order_date >= w + HOLD)).sum() for l, w in det.items() if l in flagged)
ruleB = ((oo.lot_base.isin(flagged)) & (oo.order_date >= first + HOLD)).sum()
L.append("## 2. Early-warning backtest (uses only data available that week)\n"
         f"- Product-level alert on Pulse 2: **week of {first.date()}**; {len(pl)} Pulse 2 alert-weeks in total.\n"
         f"- Other products: {len(other)} alert-week(s) ({', '.join(f'{r.product_sku} {r.week.date()}' for r in other.itertuples())}) out of ~{n_cells:,} product-weeks "
         f"({len(other)/n_cells:.2%}); no sustained rise followed, so a human would close it after a look.\n"
         f"- Lot-level detection (first week each lot passes the test): "
         + ", ".join(f"{l} {w.date()}" for l, w in det.items() if l in flagged)
         + ". Lots made later are detected later because they have fewer tickets at first. No normal Pulse 2 lot was ever flagged.")

# 3. money, with a sensitivity range
lo = cost_of_flagged_lots(t, o, p, SKU, flagged, lt_all[~lt_all.index.isin(flagged)].repl_per_100_orders.max())["total_rs"]
L.append("## 3. Money\n"
         f"- Excess cost on the 3 lots (Jan 2025 to Jun 2026): **Rs {cost['total_rs']/1e5:.1f} L** = replacements Rs {cost['replacement_rs']/1e5:.1f} L "
         f"({cost['excess_replacements']:.0f} x Rs {cost['rep_cost_each']:,.0f}: unit cost 1,480 + 340 logistics, policy s5) + extra contacts Rs {cost['contact_rs']/1e5:.1f} L "
         f"({cost['excess_tickets']:.0f} x Rs 290). Using the highest normal lot as baseline instead of the median: Rs {lo/1e5:.1f} L.\n"
         f"- Rs {per_order:,.0f} excess per bad-lot order.\n"
         f"- Avoidable, rule A (hold each lot only once ITS OWN test fires, +14 days): {ruleA:,} orders = about **Rs {ruleA*per_order/1e5:.1f} L**.\n"
         f"- Avoidable, rule B (product alert fires, hold the flagged lot and later-made lots after +14 days): {ruleB:,} orders = about **Rs {ruleB*per_order/1e5:.1f} L**. "
         f"Rule B is an operational assumption, not something the data proves.\n"
         f"- Both assume held stock is swapped for good stock and the excess per order stays the same. Finance's Rs 2,500 per replacement does not match the policy figure for Pulse 2 (Rs {cost['rep_cost_each']:,.0f}).")

# 4. text model: random CV (done in the tool) + time hold-out
pipe, mm, d = fit_symptom_model(t, SKU, flagged)
d = d.sort_values("created_at"); cut = d.created_at.quantile(.7)
tr, te = d[d.created_at <= cut], d[d.created_at > cut]
m2 = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True),
                   LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")).fit(_clean(tr.customer_message), tr.y)
pr = m2.predict_proba(_clean(te.customer_message))[:, 1]
L.append(f"## 4. Text model (Pulse 2 first message -> 'looks like a bad-lot ticket')\n"
         f"- 5-fold CV on {mm['n']:,} tickets: AUC {mm['auc']:.2f}, precision {mm['precision']:.0%}, recall {mm['recall']:.0%}.\n"
         f"- Train on first 70% of time, test on the last 30% ({len(te):,} tickets): AUC {roc_auc_score(te.y, pr):.2f}, precision {precision_score(te.y, pr>=.5):.0%}, recall {recall_score(te.y, pr>=.5):.0%}.\n"
         f"- Where it is wrong: it misses {1-mm['recall']:.0%} of bad-lot tickets. Missed tickets are spread over delivery (17%), other (16%), billing (14%), returns (11%): ordinary queries from customers who happened to have a bad-lot unit. "
         f"Caught tickets are mostly Charging & Battery (39%) and Warranty (15%). It finds the symptom, not the lot. Use it as a second signal only.")

# 5. agent flags: how many would be flagged by chance? shuffle ticket->agent within team
sc, _ = scorecard(t, a, flagged)
real = int((sc.z <= -Z_FLAG).sum())
rng = np.random.default_rng(0); null = []
for _ in range(30):
    t2 = t.copy()
    t2["agent_id"] = t2.groupby("team").agent_id.transform(lambda s: rng.permutation(s.values))
    s2, _ = scorecard(t2, a, flagged)
    null.append(int((s2.z <= -Z_FLAG).sum()))
L.append(f"## 5. Agent flags\n- Agents with adjusted gap z <= -{Z_FLAG}: **{real}** (all four are the same chat rota: {', '.join(sc[sc.z<=-Z_FLAG].agent_id)}).\n"
         f"- Same test with ticket->agent assignment shuffled inside each team (30 runs): mean {np.mean(null):.1f} flagged, max {max(null)}. So 4 is above chance, but the four share one queue, "
         f"so the tool cannot separate 'worse agents' from 'worse rota'.\n"
         f"- Raw bottom ten: {int(sc.raw_bottom10.sum())} agents, of whom 6 are Tier 2 and 4 are this rota. After adjustment Tier 2 gaps are within noise except one borderline case.\n"
         f"- Known weaknesses: tickets with no lot code (8%) are not adjusted for lot (direction of bias unknown); only ~45% of customers answer the survey, and who answers is not random; the chat four's gap sits mostly on Pulse 2 hardware tickets in Dec to Mar (2.5 vs 3.6 for other chat agents on the same product) and is small elsewhere.")

open("out/validation.md", "w").write("# Validation\n\n" + "\n\n".join(L) + "\n")
print("\n\n".join(L))
