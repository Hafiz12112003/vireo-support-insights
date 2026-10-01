"""python run.py [--data data] [--out out]   ->  out/dashboard.html + CSVs"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from vireo.clean import load
from vireo.lots import weekly_alerts, lot_table, cost_of_flagged_lots
from vireo.text import fit_symptom_model, score_all
from vireo.agents import scorecard
from vireo import report

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

t, o, a, p, log = load(args.data)
alerts = weekly_alerts(t, o, "replaced")
alerts.to_csv(f"{args.out}/early_warning_alerts.csv", index=False)
print("Early-warning alerts by product:\n", alerts.groupby("product_sku").week.agg(["min", "count"]).to_string())

# products with at least 3 alerts are treated as incidents (one-off alerts are logged, not acted on)
cnt = alerts.groupby("product_sku").size()
incident_skus = list(cnt[cnt >= 3].index)
if not incident_skus:
    raise SystemExit("No product incident found.")
sku = incident_skus[0]
lt, base = lot_table(t, o, sku)
flagged = list(lt[lt.quarantine_candidate].index)
cost = cost_of_flagged_lots(t, o, p, sku, flagged, base)
lt.to_csv(f"{args.out}/lot_table_{sku}.csv")
print(f"{sku} flagged lots: {flagged}; excess cost Rs {cost['total_rs']:,.0f}")

pipe, mm, _ = fit_symptom_model(t, sku, flagged)
t["symptom_prob"] = score_all(pipe, t, sku)
sc, _ = scorecard(t, a, flagged)
sc.to_csv(f"{args.out}/agent_scorecard.csv", index=False)
report.build(f"{args.out}/dashboard.html", dict(t=t, o=o, sc=sc, lt=lt, cost=cost, alerts=alerts, log=log,
                                                   text_metrics=mm, flagged=flagged, sku=sku))
print("Wrote", f"{args.out}/dashboard.html")
