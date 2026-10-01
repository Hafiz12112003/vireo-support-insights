# Validation

## 1. Lot attribution
Fallback join (customer+sku) vs the order_id join on 6,125 tickets where both exist: **100.0% identical**. Tickets with no lot at all: 917 (7.8%); they are left out of lot statistics (this understates the problem, it does not overstate it).

## 2. Early-warning backtest (uses only data available that week)
- Product-level alert on Pulse 2: **week of 2025-12-01**; 6 Pulse 2 alert-weeks in total.
- Other products: 1 alert-week(s) (VA-SW-NX2 2025-09-08) out of ~1,106 product-weeks (0.09%); no sustained rise followed, so a human would close it after a look.
- Lot-level detection (first week each lot passes the test): PL2-2510 2025-12-08, PL2-2511 2026-01-19, PL2-2512 2026-03-02. Lots made later are detected later because they have fewer tickets at first. No normal Pulse 2 lot was ever flagged.

## 3. Money
- Excess cost on the 3 lots (Jan 2025 to Jun 2026): **Rs 16.0 L** = replacements Rs 12.9 L (711 x Rs 1,820: unit cost 1,480 + 340 logistics, policy s5) + extra contacts Rs 3.0 L (1044 x Rs 290). Using the highest normal lot as baseline instead of the median: Rs 15.6 L.
- Rs 807 excess per bad-lot order.
- Avoidable, rule A (hold each lot only once ITS OWN test fires, +14 days): 623 orders = about **Rs 5.0 L**.
- Avoidable, rule B (product alert fires, hold the flagged lot and later-made lots after +14 days): 1,566 orders = about **Rs 12.6 L**. Rule B is an operational assumption, not something the data proves.
- Both assume held stock is swapped for good stock and the excess per order stays the same. Finance's Rs 2,500 per replacement does not match the policy figure for Pulse 2 (Rs 1,820).

## 4. Text model (Pulse 2 first message -> 'looks like a bad-lot ticket')
- 5-fold CV on 3,787 tickets: AUC 0.77, precision 81%, recall 62%.
- Train on first 70% of time, test on the last 30% (1,136 tickets): AUC 0.74, precision 78%, recall 62%.
- Where it is wrong: it misses 38% of bad-lot tickets. Missed tickets are spread over delivery (17%), other (16%), billing (14%), returns (11%): ordinary queries from customers who happened to have a bad-lot unit. Caught tickets are mostly Charging & Battery (39%) and Warranty (15%). It finds the symptom, not the lot. Use it as a second signal only.

## 5. Agent flags
- Agents with adjusted gap z <= -2.5: **4** (all four are the same chat rota: A3004, A3005, A3006, A3007).
- Same test with ticket->agent assignment shuffled inside each team (30 runs): mean 0.2 flagged, max 1. So 4 is above chance, but the four share one queue, so the tool cannot separate 'worse agents' from 'worse rota'.
- Raw bottom ten: 10 agents, of whom 6 are Tier 2 and 4 are this rota. After adjustment Tier 2 gaps are within noise except one borderline case.
- Known weaknesses: tickets with no lot code (8%) are not adjusted for lot (direction of bias unknown); only ~45% of customers answer the survey, and who answers is not random; the chat four's gap sits mostly on Pulse 2 hardware tickets in Dec to Mar (2.5 vs 3.6 for other chat agents on the same product) and is small elsewhere.
