#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Monte-Carlo latent-score forecaster, Python 3.13 standard library only, no
network, fully deterministic (fixed seed).

Model
-----
Each game has a latent score s = -ln(rank). The chart orders games by s.
Between reads, s moves as a random walk with a per-game drift and a shared
volatility, both estimated from the history itself:

* drift: each game's recent change in s per hour (exponentially weighted,
  18 h half-life, last 18 h of history), shrunk 25% toward 0 and capped;
  rank momentum persists (Spearman ~0.25 in the archive), while raw
  playing-count growth does not predict the next move at all (~-0.01).
* volatility: pooled per-hour standard deviation of changes in s,
  scaled by sqrt(dt) for the gaps between reads.
* exits: games leave the chart at a low hourly hazard (the archive loses
  ~5% of games per ~6 h read, concentrated in low ranks).
* entries: new games join at ~1.6/hour with scores drawn from the
  empirical entry-rank distribution; 5% of entrants are "rockets" that
  draw a top-30 score (in the archive 32 of 207 first-seen games reached
  the top 10 within 72 h, four of them from entry ranks 55-69 to #1).

Simulation: 6000 paths x 72 hours. P(game at place k) is the fraction of
(path, hour) slots where the game holds place k. Every place keeps a
minimum unnamed share (25% at place 1, tapering to 8% at places 6-10):
the unnamed share is divided, at each scored read, over the games on that
read's chart the forecast did not name, so it is the only way to cover
games that enter the chart during the window -- in the practice window
the eventual #1 and #2 ("Do Not Stop.", "Bloody Ragdoll Physics") were
games that were not in the history at all.

Backtest (practice window 2026-11-07T23:00Z -> 2026-11-10T23:00Z, the 10
archive reads, scored with the exact log-score rule): -35.17, vs -43.51
for the house baseline, -37.2 for the house model, and -34.6 for the
best entry on the board at submission time. Rolling backtests at four
earlier cutoffs average -38.1 vs -42.3 for the same model without the
unnamed taper.
"""

import json
import math
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

SEED = 20261111
N_SIM = 6000

# Tuned constants (see module docstring for the backtest evidence).
DRIFT_SHRINK = 0.75        # shrink per-game drift toward 0
DRIFT_CAP = 0.15           # |drift| cap, log-score points per hour
DRIFT_WINDOW_H = 18.0      # only this much recent history feeds the drift
VEL_HALF_LIFE_H = 18.0     # exponential weight half-life for drift
VOL_SCALE = 2.0            # multiplier on the pooled per-hour volatility
EXIT_PER_HOUR = 0.003      # uniform hourly exit hazard
ENTRY_PER_HOUR = 1.6       # new games per hour (archive: ~5 per ~6 h read)
ROCKET_PROB = 0.05         # share of entrants drawing a top-30 score
MIN_UNNAMED = (0.25, 0.20, 0.15, 0.12, 0.10,
               0.08, 0.08, 0.08, 0.08, 0.08)  # per place 1..10
MIN_P = 0.001              # drop probability rows below this
# Age-out: the chart lists only games published in the last 28 days.
# When details.created is available, ramp the exit hazard from 26 days
# to a certain exit at 28.5 days. (Archive reads carry no details, so
# this only bites on the final history, which has them.)
AGE_RAMP_START_D = 26.0
AGE_RAMP_END_D = 28.5


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_history(input_dir):
    rows = []
    path = Path(input_dir) / "history.ndjson"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["at"])
    return rows


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task = json.loads((Path(input_dir) / "task.json").read_text(encoding="utf-8"))
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {})
    hours = int(window.get("slots", 72) * window.get("slot_minutes", 60) // 60) or 72

    rows = load_history(input_dir)
    if not rows:
        raise SystemExit("empty history")

    rng = random.Random(SEED)
    times = [parse_time(r["at"]) for r in rows]
    t_end = times[-1]

    # Per-game rank series and creation dates (details.created when given).
    series = {}
    created = {}
    for r in rows:
        t = parse_time(r["at"])
        for g in r.get("up_and_coming") or []:
            series.setdefault(g["universe_id"], []).append((t, g["rank"]))
        details = r.get("details") or {}
        if isinstance(details, dict):
            items = details.items()
        elif isinstance(details, list):
            items = [(d.get("universe_id"), d) for d in details
                     if isinstance(d, dict)]
        else:
            items = []
        for key, d in items:
            if isinstance(d, dict) and d.get("created"):
                try:
                    created[int(key)] = parse_time(d["created"])
                except (ValueError, TypeError):
                    pass

    last = {g["universe_id"]: g["rank"]
            for g in rows[-1].get("up_and_coming") or []}
    if not last:
        raise SystemExit("last snapshot has no chart")

    # Pooled per-hour volatility of d(-ln rank), and entry-score pool.
    dvals = []
    entry_scores = []
    prev = None
    prev_t = None
    for r in rows:
        cur = {g["universe_id"]: g["rank"]
               for g in r.get("up_and_coming") or []}
        t = parse_time(r["at"])
        if prev is not None:
            dt = max((t - prev_t).total_seconds() / 3600.0, 0.25)
            for u, rk in cur.items():
                if u in prev and rk > 0 and prev[u] > 0:
                    dvals.append(((-math.log(rk)) - (-math.log(prev[u])))
                                 / math.sqrt(dt))
                elif u not in prev and rk > 0:
                    entry_scores.append(-math.log(rk))
        prev, prev_t = cur, t
    sigma = (statistics.pstdev(dvals) * VOL_SCALE) if len(dvals) > 10 else 0.30
    if not entry_scores:
        entry_scores = [-math.log(90.0)]

    # Per-game drift: exponentially weighted recent change in s per hour.
    drift = {}
    for u, pts in series.items():
        pts2 = [(t, rk) for (t, rk) in pts
                if (t_end - t).total_seconds() / 3600.0 <= DRIFT_WINDOW_H]
        if len(pts2) < 2:
            drift[u] = 0.0
            continue
        num = 0.0
        den = 0.0
        for (t1, r1), (t2, r2) in zip(pts2, pts2[1:]):
            dt = (t2 - t1).total_seconds() / 3600.0
            if dt <= 0 or r1 <= 0 or r2 <= 0:
                continue
            w = 0.5 ** ((t_end - t2).total_seconds() / 3600.0 / VEL_HALF_LIFE_H)
            num += w * (((-math.log(r2)) - (-math.log(r1))) / dt)
            den += w
        d = DRIFT_SHRINK * (num / den if den else 0.0)
        drift[u] = max(-DRIFT_CAP, min(DRIFT_CAP, d))

    # Age (days) of each game at the end of history, from details.created.
    age_days = {}
    for u, c in created.items():
        if u in last:
            age_days[u] = (t_end - c).total_seconds() / 86400.0

    def age_hazard(u, hour):
        age = age_days.get(u)
        if age is None:
            return 0.0
        a = age + hour / 24.0
        if a <= AGE_RAMP_START_D:
            return 0.0
        if a >= AGE_RAMP_END_D:
            return 1.0
        return 0.5 * (a - AGE_RAMP_START_D) / (AGE_RAMP_END_D - AGE_RAMP_START_D)

    s0 = {u: -math.log(rk) for u, rk in last.items() if rk > 0}
    counts = {}
    entrant_hits = [0] * (places + 1)
    total_slots = N_SIM * hours
    for _ in range(N_SIM):
        s = dict(s0)
        alive = set(s0)
        nxt = -1  # simulated entrants get negative ids
        for h in range(hours):
            for u in list(alive):
                if rng.random() < EXIT_PER_HOUR + age_hazard(u, h):
                    alive.discard(u)
                    s.pop(u, None)
            if rng.random() < ENTRY_PER_HOUR:
                if rng.random() < ROCKET_PROB:
                    es = -math.log(rng.uniform(1, 30))
                else:
                    es = rng.choice(entry_scores)
                s[nxt] = es
                alive.add(nxt)
                nxt -= 1
            for u in list(alive):
                s[u] = s[u] + drift.get(u, 0.0) + rng.gauss(0.0, sigma)
            order = sorted(alive, key=lambda u: (-s[u], u))
            for k, u in enumerate(order[:places], start=1):
                if u >= 0:
                    counts[(u, k)] = counts.get((u, k), 0) + 1
                else:
                    entrant_hits[k] += 1

    # Unnamed share per place: simulated entrants' share, at least the floor.
    unnamed = [0.0] * (places + 1)
    for k in range(1, places + 1):
        floor = MIN_UNNAMED[k - 1] if k - 1 < len(MIN_UNNAMED) else 0.08
        unnamed[k] = max(entrant_hits[k] / total_slots, floor)

    probabilities = []
    for (u, k), c in counts.items():
        probabilities.append({"universe_id": u, "place": k,
                              "p": c / total_slots})
    for k in range(1, places + 1):
        ks = [r for r in probabilities if r["place"] == k]
        tot = sum(r["p"] for r in ks)
        target = max(0.0, 1.0 - unnamed[k])
        if tot > 0:
            for r in ks:
                r["p"] = r["p"] * target / tot
    probabilities = [r for r in probabilities if r["p"] >= MIN_P]
    for r in probabilities:
        r["p"] = round(r["p"], 6)
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: rank games by expected place-score sum_k P(g at k) * (11 - k).
    top_score = {}
    for r in probabilities:
        top_score[r["universe_id"]] = (top_score.get(r["universe_id"], 0.0)
                                       + r["p"] * (places + 1 - r["place"]))
    top10 = sorted(top_score, key=lambda u: -top_score[u])[:places]
    if len(top10) < places:
        for u in sorted(last, key=lambda u: last[u]):
            if u not in top10:
                top10.append(u)
            if len(top10) == places:
                break

    out = {"probabilities": probabilities, "top10": top10}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / output_name, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
