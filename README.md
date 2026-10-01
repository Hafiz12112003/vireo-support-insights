# Vireo Audio: support insights (CSAT, handle time, bad-lot early warning)

Answers Priya Raman's request (CSAT and handle time per agent, bottom ten) and adds the thing the data says
matters more: **three Pulse 2 production lots (PL2-2510, -2511, -2512) cost about Rs 16 L and explain most of the CSAT slide.**
Everything runs locally. **Zero paid API calls.**

## Run it (clean machine, Python 3.10+)

```bash
git clone <this repo> && cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the five CSVs from the data pack in `data/` (`tickets.csv orders.csv agents.csv products.csv customers.csv`).
They are not in the repo because they are client data.

```bash
python run.py          # about 10 s -> out/dashboard.html (open in a browser) + CSVs
python validate.py     # about 15 s -> out/validation.md (backtest, error rates, agent-flag null test)
```

| Output | What it is |
|---|---|
| `out/dashboard.html` | CSAT trend, early-warning chart, lot table, per-agent table (raw and adjusted), data fixes |
| `out/agent_scorecard.csv` | one row per agent: raw CSAT, expected CSAT, gap and interval, handle time, SLA breach, verdict |
| `out/lot_table_VA-EB-PL2.csv` | replacements per 100 orders by production lot |
| `out/early_warning_alerts.csv` | product-weeks that tripped the alarm |
| `out/validation.md` | how I know it works and where it does not |

## What it does

1. **Cleans** the exports (`vireo/clean.py`): legacy timestamps UTC to IST, joins on `agent_id` only, lot code from `order_id` or customer+SKU.
2. **Early warning** (`vireo/lots.py`): weekly replacements per product against its own trailing baseline (Poisson test, no look-ahead). Then ranks production lots by replacements per 100 orders and names quarantine candidates.
3. **Text signal** (`vireo/text.py`): TF-IDF + logistic regression on the customer's first message. Local, free, second opinion only.
4. **Agent scorecard** (`vireo/agents.py`): raw numbers as asked, plus CSAT *expected for the tickets that agent got*, with intervals.

## Decisions I made (nobody to ask)

- **Pushed back on "bottom ten".** Raw bottom ten = 6 Tier 2 warranty agents + 4 chat hardware-rota agents. Policy s6 says Tier 2 is not comparable to Tier 1; Neha's email says the rota gets the angriest customers. I still output the raw ranking, but the verdict column says what it can and cannot support.
- **Flag at z <= -2.5, minimum 80 survey responses.** 44 agents compared; at 2.0 about two would be flagged by luck.
- **Controls are ticket mix only** (category, channel, priority, product, month, team, bad lot). Breach and transfers are not controlled for because agents partly cause them.
- **Handle time** = first response to resolution (policy s10). Tier 1 shown in minutes, Tier 2 and Logistics/Returns in days.
- **Lots are grouped by production month** (PL2-2511-1..4 are pooled). All four sub-lots look the same, and pooling gives enough orders.
- **Open and pending tickets** have no resolution time; they are in CSAT counts only if surveyed.
- **Money:** replacement = unit cost + Rs 340 (policy s5), contact = Rs 290 blended, not Finance's Rs 2,500.

## Known limits (details in `out/validation.md`)

- The tool finds one incident (Pulse 2). The text model was trained on Pulse 2 only and needs retraining for another product.
- Lot-level detection is late for lots made later (2511 flagged 19 Jan, 2512 on 2 Mar). The product-level alert fired 1 Dec.
- 8% of tickets have no lot code and are left out of lot maths.
- The chat four's gap cannot be separated from the rota they work. The tool cannot tell if they are worse agents or have worse queues.
- Export is about 150 tickets a week; the brief says about 650 a week. Rs figures are for this export, not scaled.
- CSAT: about 45% respond; who responds is not random.
- No duplicate tickets were found despite policy s9 warning about re-imports; I checked on customer, time, product, message and agent.
- 175 tickets are dated before the order they cite. Left as is.

## If you pick this up on Monday

1. `python run.py`, open `out/dashboard.html`. The Pulse 2 table and chart are the main story.
2. Money numbers come from `vireo/lots.py::cost_of_flagged_lots`; the policy figures (Rs 340, Rs 290, Rs 350 credit) are constants at the top of that file.
3. Do not read the agent table as a ranking for retraining or bonuses without checking queue assignment first.
