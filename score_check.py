"""python score_check.py  -> compares your labels with the model's flags"""
import pandas as pd
s = pd.read_csv("check_sheet.csv"); k = pd.read_csv("check_key.csv")
d = s.merge(k, on=["row", "ticket_id"])
d["y"] = d.your_label.astype(str).str.strip().str.upper().str[0].map({"Y": 1, "N": 0})
if d.y.isna().any():
    raise SystemExit(f"{int(d.y.isna().sum())} rows have no Y/N label yet (rows: {list(d[d.y.isna()].row)})")
f = d[d.pred == "flagged"]; n = d[d.pred == "not_flagged"]
print(f"Model flagged 25 tickets: you said Y on {int(f.y.sum())} -> precision {f.y.mean():.0%}")
print(f"Model did not flag 25 tickets: you said Y on {int(n.y.sum())} -> missed {n.y.mean():.0%}")
print(f"Overall agreement: {((d.pred == 'flagged') == (d.y == 1)).mean():.0%} of {len(d)}")
print(d[(d.pred == 'flagged') != (d.y == 1)][["row", "pred", "your_label", "customer_message"]].to_string(index=False))
