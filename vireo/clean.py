"""Load the Vireo exports and fix the known data problems. Every fix is logged."""
import re
import numpy as np
import pandas as pd

SLA_TARGET_MIN = {"chat": 15, "voice": 120, "social": 240, "email": 480}  # policy s3
JUNK_PAT = r"\[(?:inaudible|crosstalk|line dropped)\]|^\W*(?:test|call me)\W*$|9XXXXXXXXX"


def load(data_dir="data"):
    log = []
    t = pd.read_csv(f"{data_dir}/tickets.csv",
                    parse_dates=["created_at", "first_response_at", "resolved_at"])
    o = pd.read_csv(f"{data_dir}/orders.csv", parse_dates=["order_date"])
    a = pd.read_csv(f"{data_dir}/agents.csv")
    p = pd.read_csv(f"{data_dir}/products.csv")

    # 1. legacy resolved_at was rebuilt from a UTC event log; helpdesk shows IST (policy s9)
    leg = t.source_system == "legacy_fd"
    t["resolved_ist"] = t.resolved_at.where(~leg, t.resolved_at + pd.Timedelta(minutes=330))
    before = ((t.resolved_at - t.first_response_at).dt.total_seconds() < 0).sum()
    t["handle_min"] = (t.resolved_ist - t.first_response_at).dt.total_seconds() / 60
    t["frt_min"] = (t.first_response_at - t.created_at).dt.total_seconds() / 60
    log.append(f"Legacy resolved_at shifted +5:30 (UTC->IST): {before} negative handle times -> "
               f"{int((t.handle_min < 0).sum())}")

    # 2. lot code: order_id when quoted, else customer+sku if that pair has exactly one lot
    t["lot"] = t.order_id.map(o.set_index("order_id").lot_code)
    fb = o.groupby(["customer_id", "sku"]).lot_code.agg(["first", "nunique"]).reset_index()
    fb = fb[fb["nunique"] == 1].rename(columns={"sku": "product_sku", "first": "lot_fb"})
    t = t.merge(fb[["customer_id", "product_sku", "lot_fb"]], on=["customer_id", "product_sku"], how="left")
    t["lot_src"] = np.where(t.lot.notna(), "order_id", np.where(t.lot_fb.notna(), "customer+sku", "none"))
    t["lot"] = t.lot.fillna(t.lot_fb)
    t["lot_base"] = t.lot.str.rsplit("-", n=1).str[0]  # production month, e.g. PL2-2511
    log.append("Lot attached to " + f"{t.lot.notna().mean():.1%} of tickets "
               f"({(t.lot_src=='order_id').mean():.1%} via order_id, {(t.lot_src=='customer+sku').mean():.1%} via customer+sku)")

    # 3. agents: join on agent_id only (two agents are both called Kavya Pandey)
    ag = a.set_index("agent_id")
    t["team"] = t.agent_id.map(ag["team"])
    t["tier"] = t.agent_id.map(ag["tier"])
    t["shift"] = t.agent_id.map(ag["shift"])
    t["family"] = t.product_sku.map(p.set_index("sku").family)

    # 4. flags
    t["breach"] = (t.frt_min > t.channel.map(SLA_TARGET_MIN)).astype(int)
    t["replaced"] = (t.replacement_issued == "Y").astype(int)
    t["month"] = t.created_at.dt.to_period("M").astype(str)
    t["week"] = t.created_at.dt.to_period("W-SUN").dt.start_time
    t["junk_text"] = t.customer_message.str.contains(JUNK_PAT, case=False, regex=True)
    log.append(f"{int(t.junk_text.sum())} tickets with unusable message text (IVR failures etc.) kept for "
               "CSAT/handle time, excluded from the text model")
    ob = t.order_id.map(o.set_index("order_id").order_date)
    log.append(f"{int((t.created_at.dt.normalize() < ob).sum())} tickets dated before the order they cite "
               "(left as is, flagged only)")
    return t, o, a, p, log
