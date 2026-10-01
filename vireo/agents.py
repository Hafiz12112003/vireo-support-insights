"""Per-agent scorecard: raw CSAT / handle time (what was asked) + CSAT adjusted for the tickets each agent actually got."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MIN_RESPONSES = 80      # below this an agent mean is mostly noise
Z_FLAG = 2.5            # 44 agents compared -> a plain 2.0 would flag ~2 by luck alone


def scorecard(t, a, flagged_lots):
    d = t.dropna(subset=["csat_score"]).copy()
    d["in_flagged_lot"] = d.lot_base.isin(flagged_lots).astype(int)
    # lot unknown on a Pulse 2 ticket in the incident window: use the share of known tickets that month
    # (kept simple on purpose; see README limitations)
    # Controls are ticket-mix only. Breach/transfers are partly caused by the agent, so they are NOT controlled for.
    f = "csat_score ~ in_flagged_lot + C(category) + C(channel) + C(priority) + C(family) + C(month) + C(team)"
    m = smf.ols(f, d).fit()
    d["expected"] = m.predict(d)
    d["gap"] = d.csat_score - d.expected
    g = d.groupby("agent_id").agg(responses=("csat_score", "count"), csat_raw=("csat_score", "mean"),
                                   expected=("expected", "mean"), gap=("gap", "mean"), gap_sd=("gap", "std"))
    g["gap_se"] = g.gap_sd / np.sqrt(g.responses)
    g["z"] = g.gap / g.gap_se
    g["share_flagged_lot"] = d.groupby("agent_id").in_flagged_lot.mean()

    allt = t.groupby("agent_id").agg(tickets=("ticket_id", "size"),
                                      handle_median_min=("handle_min", "median"),
                                      handle_p90_min=("handle_min", lambda s: s.quantile(.9)),
                                      sla_breach_rate=("breach", "mean"),
                                      high_priority_share=("priority", lambda s: (s == "High").mean()))
    out = a.set_index("agent_id")[["name", "site", "team", "shift", "tier"]].join(allt).join(g)
    out["handle_vs_team_median"] = out.handle_median_min / out.groupby("team").handle_median_min.transform("median")
    out["rank_raw"] = out.csat_raw.rank(method="min")                 # 1 = lowest
    out["rank_adjusted"] = out.gap.rank(method="min")
    out["raw_bottom10"] = out.rank_raw <= 10
    out["adj_bottom10"] = out.rank_adjusted <= 10

    def verdict(r):
        if r.responses < MIN_RESPONSES:
            return "too few survey responses"
        if r.tier == 2:
            return "Tier 2: do not rank on CSAT/volume (policy s6)" if r.z > -Z_FLAG else "Tier 2: review queue first"
        if r.z <= -Z_FLAG:
            return "gap persists after adjusting: look at queue/rota, then coach"
        if r.raw_bottom10:
            return "raw bottom ten, but explained by ticket mix"
        return "no action"
    out["verdict"] = out.apply(verdict, axis=1)
    return out.reset_index(), m
