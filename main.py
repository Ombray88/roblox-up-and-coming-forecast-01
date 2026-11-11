#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Jump-diffusion Monte-Carlo latent-score forecaster, Python 3.13 standard
library only, no network, fully deterministic (fixed seed).

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

Two structural changes vs the v1 Monte-Carlo forecaster (which scored
-35.17 on the practice window):

1. HEAVY-TAIL JUMPS. A Gaussian walk cannot produce the rockets the chart
   actually shows: a rank-29 game reaching #1 needs +3.4 in s, ~4 sigma
   under the pooled volatility, yet the archive has rank 21-30 games in
   the top 3 within 24 h about 0.8% of the time. Every game-hour now also
   carries a 1% chance of an extra N(0, 1.0) jump, giving the forecast
   the heavy tails the real chart has.
2. PLAYING-SURGE JUMPS. A game's `playing` count sometimes explodes while
   its rank barely moves -- the rank lags the surge by 12-36 h. In the
   pooled archive, games whose playing grew >e^1 over the last two reads
   reached the top 10 within 24 h 58% of the time (base rate 11%), and
   with a high playing level as well, 62% reached the top 10 and 43%
   the top 3. Such surge games (log playing growth over the last two
   reads above 1.0) get a jump-diffusion kick of their own: with
   probability 0.5 their latent score jumps +U(1.5, 3.0) in the first
   24 h. The practice window's rank-29 climber (playing 6.3k -> 45k
   while its rank sat at 29, then #1/#2 for most of the window) fits
   this profile exactly; brand-new rockets appear in no chart at all,
   so only the unnamed share covers them.

Simulation: 6000 paths x 72 hours. P(game at place k) is the fraction of
(path, hour) slots where the game holds place k. Every place keeps a
minimum unnamed share (25% at place 1, tapering to 14% at places 6-10):
the unnamed share is divided, at each scored read, over the games on that
read's chart the forecast did not name, so it is the only way to cover
games that enter the chart during the window. The floors are higher at
places 4-10 than v1's (which tapered to 8%): pooled across four
validation windows, 11-47% of scored cells at each place are occupied by
a game the forecast did not name at that place, so an 8% floor starves
the unnamed pool; the raised floors gain about +0.3 on the rolling mean
on every base model tested.

Backtest (6000 simulations, practice window 2026-11-07T23:00Z ->
2026-11-10T23:00Z, the 10 archive reads, scored with the exact log-score
rule): -34.89, vs -35.17 for v1; rolling backtests at three earlier
cutoffs (2026-11-04 / 11-05 / 11-06) score -39.04 / -42.63 / -36.06
(mean -39.24) vs -40.50 / -44.42 / -36.96 (mean -40.63) for v1 -- better
than v1 on the practice window and on every rolling window (four-window
mean -38.16 vs -39.26). Two runs on the kit input are byte-identical;
runtime ~75 s per run.
"""

import json
import math
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

SEED = 20261111
import os as _os
def _envf(n, d):
    try: return float(_os.environ.get(n, d))
    except ValueError: return d
N_SIM = int(_envf("EXP_N_SIM", 6000))

# Tuned constants (see module docstring for the backtest evidence).
DRIFT_SHRINK = _envf("EXP_SHRINK", 0.75)        # shrink per-game drift toward 0
DRIFT_CAP = _envf("EXP_CAP", 0.15)           # |drift| cap, log-score points per hour
DRIFT_WINDOW_H = _envf("EXP_WIN_H", 18.0)      # only this much recent history feeds the drift
VEL_HALF_LIFE_H = _envf("EXP_HALF", 18.0)     # exponential weight half-life for drift
VOL_SCALE = _envf("EXP_VOL", 2.0)            # multiplier on the pooled per-hour volatility
EXIT_PER_HOUR = _envf("EXP_EXIT", 0.003)      # uniform hourly exit hazard
ENTRY_PER_HOUR = _envf("EXP_ENTRY", 1.6)       # new games per hour (archive: ~5 per ~6 h read)
ROCKET_PROB = _envf("EXP_ROCKET", 0.05)         # share of entrants drawing a top-30 score
_mu = _os.environ.get("EXP_UNNAMED", "0.25,0.22,0.20,0.18,0.16,0.14,0.14,0.14,0.14,0.14")
MIN_UNNAMED = tuple(float(x) for x in _mu.split(","))
MIN_P = 0.001              # drop probability rows below this
BANDS = [(1, 3), (4, 10), (11, 20), (21, 30), (31, 50), (51, 97)]
SIGMA_BAND = False  # heteroscedastic sigma tested: no gain (see notes)
SIGMA_SHRINK_K = _envf("V3_SIGMA_K", 100.0)   # pseudo-count shrinking band var to pooled
EXIT_BAND = False  # rank-dependent exit tested: no gain (see notes)
SURGE_ON = True
SURGE_THRESH = 1.0  # log playing growth over last 2 reads
SURGE_PJUMP = 0.5
SURGE_JLO = 1.5
SURGE_JHI = 3.0
JUMP_P = 0.01  # per game-hour prob of a heavy-tail jump
JUMP_SD = 1.0  # sd of the jump (added to the normal step)
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

    # Per-game rank series, playing series, creation dates.
    series = {}
    playing_series = {}
    created = {}
    for r in rows:
        t = parse_time(r["at"])
        for g in r.get("up_and_coming") or []:
            series.setdefault(g["universe_id"], []).append((t, g["rank"]))
            if g.get("playing"):
                playing_series.setdefault(g["universe_id"], []).append(
                    (t, float(g["playing"])))
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
    # Also per-band dvals and exit counts for the heteroscedastic model.
    def band_of_rank(rk):
        for i, (lo, hi) in enumerate(BANDS):
            if lo <= rk <= hi:
                return i
        return len(BANDS) - 1

    dvals = []
    band_dvals = {i: [] for i in range(len(BANDS))}
    band_exposure_h = {i: 0.0 for i in range(len(BANDS))}
    band_exits = {i: 0 for i in range(len(BANDS))}
    entry_scores = []
    prev = None
    prev_t = None
    for r in rows:
        cur = {g["universe_id"]: g["rank"]
               for g in r.get("up_and_coming") or []}
        t = parse_time(r["at"])
        if prev is not None:
            dt = max((t - prev_t).total_seconds() / 3600.0, 0.25)
            for u, rk in prev.items():
                b = band_of_rank(rk)
                band_exposure_h[b] += dt
                if u not in cur:
                    band_exits[b] += 1
            for u, rk in cur.items():
                if u in prev and rk > 0 and prev[u] > 0:
                    dv = (((-math.log(rk)) - (-math.log(prev[u])))
                          / math.sqrt(dt))
                    dvals.append(dv)
                    band_dvals[band_of_rank(prev[u])].append(dv)
                elif u not in prev and rk > 0:
                    entry_scores.append(-math.log(rk))
        prev, prev_t = cur, t
    sigma = (statistics.pstdev(dvals) * VOL_SCALE) if len(dvals) > 10 else 0.30
    if not entry_scores:
        entry_scores = [-math.log(90.0)]
    # Per-band sigma: variance shrunk toward pooled variance.
    sigma_band = {}
    if SIGMA_BAND and len(dvals) > 10:
        var_pool = statistics.pvariance(dvals)
        for i, vs in band_dvals.items():
            if len(vs) > 5:
                var_b = statistics.pvariance(vs)
                var = ((len(vs) * var_b + SIGMA_SHRINK_K * var_pool)
                       / (len(vs) + SIGMA_SHRINK_K))
                sigma_band[i] = math.sqrt(var) * VOL_SCALE
            else:
                sigma_band[i] = sigma
    else:
        sigma_band = {i: sigma for i in range(len(BANDS))}
    # Per-band hourly exit hazard, shrunk toward the flat rate.
    exit_band = {}
    if EXIT_BAND:
        for i in range(len(BANDS)):
            hz = band_exits[i] / band_exposure_h[i] if band_exposure_h[i] else EXIT_PER_HOUR
            w = band_exposure_h[i] / (band_exposure_h[i] + 500.0)
            exit_band[i] = max(0.0002, min(0.03, w * hz + (1 - w) * EXIT_PER_HOUR))
    else:
        exit_band = {i: EXIT_PER_HOUR for i in range(len(BANDS))}
    # Playing-surge games: log playing growth over the last two reads.
    surge = {}  # universe_id -> g12
    if SURGE_ON:
        for u, pts in playing_series.items():
            if u in last and len(pts) >= 3:
                p_last = pts[-1][1]
                p_2ago = pts[-3][1]
                if p_last > 0 and p_2ago > 0:
                    g12 = math.log(p_last / p_2ago)
                    if g12 > SURGE_THRESH:
                        surge[u] = g12

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
    # Static per-game band (from closing rank) for sigma / exit hazard.
    game_band = {u: band_of_rank(rk) for u, rk in last.items() if rk > 0}
    game_sigma = {u: sigma_band[game_band[u]] for u in s0}
    game_exit = {u: exit_band[game_band[u]] for u in s0}
    _kappa = _envf("EXP_KAPPA", 0.0)
    _smean = {}
    if _kappa:
        for u, pts in series.items():
            _smean[u] = sum(-math.log(rk) for (_, rk) in pts) / len(pts)
    counts = {}
    entrant_hits = [0] * (places + 1)
    total_slots = N_SIM * hours
    for _ in range(N_SIM):
        s = dict(s0)
        alive = set(s0)
        nxt = -1  # simulated entrants get negative ids
        # Surge jumps scheduled for this path: {game: (hour, size)}.
        jumps = {}
        if SURGE_ON:
            for u in surge:
                if rng.random() < SURGE_PJUMP:
                    jumps[u] = (rng.randrange(0, 24),
                                rng.uniform(SURGE_JLO, SURGE_JHI))
        for h in range(hours):
            for u in list(alive):
                if rng.random() < game_exit.get(u, EXIT_PER_HOUR) + age_hazard(u, h):
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
            _decay = math.exp(-h / _envf("EXP_DECAY_H", 1e9)) if h else 1.0
            for u in list(alive):
                _step = drift.get(u, 0.0) * _decay + rng.gauss(0.0, game_sigma.get(u, sigma))
                if JUMP_P > 0 and rng.random() < JUMP_P:
                    _step += rng.gauss(0.0, JUMP_SD)
                if u in jumps and jumps[u][0] == h:
                    _step += jumps[u][1]
                if _kappa and u in _smean:
                    _step += -_kappa * (s[u] - _smean[u])
                s[u] = s[u] + _step
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
    if _os.environ.get("EXP_TOP10", "mc") == "closing":
        top10 = sorted(last, key=lambda u: last[u])[:places]
    else:
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
