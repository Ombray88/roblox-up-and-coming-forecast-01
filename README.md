# roblox-up-and-coming-forecast-01

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Ombray88/roblox-up-and-coming-forecast-01/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Ombray88/roblox-up-and-coming-forecast-01/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A program that forecasts Roblox's Up-and-Coming top 10 over the 72 hours
after entries close, place by place, with honest probabilities.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads
`/input/history.ndjson` and `/input/task.json` (or `$MAKEFUN_INPUT_DIR`)
and writes `/output/forecast.json` (or `$MAKEFUN_OUTPUT_DIR`): one
`probabilities` list of `{"universe_id", "place", "p"}` rows and a
`top10` guess in order. Two runs on the same history write identical
bytes (fixed seed).

## Method

Monte-Carlo latent-score forecaster with jump diffusion (6,000 paths x
72 hours). Each game has a latent score `s = -ln(rank)`; the chart orders
games by `s`. Between reads, `s` moves as a random walk with a per-game
drift and a shared volatility, both estimated from the history itself:

- **Drift:** each game's recent change in `s` per hour (exponentially
  weighted, 18 h half-life, last 18 h), shrunk 25% toward 0 and capped.
  Rank momentum persists (Spearman ~0.25 in the archive); raw
  playing-count growth does not predict the next move at all (~-0.01).
- **Volatility:** pooled per-hour standard deviation of changes in `s`,
  scaled by `sqrt(dt)` for the gaps between reads.
- **Exits:** games leave at a low hourly hazard; games older than 26 days
  (from `details.created` when the history carries it) ramp to a certain
  exit at 28.5 days, since the chart lists only games published in the
  last 28 days.
- **Entries:** new games join at ~1.6/hour with scores drawn from the
  empirical entry-rank distribution; 5% of entrants are "rockets" drawing
  a top-30 score (in the archive, 32 of 207 first-seen games reached the
  top 10 within 72 h, four of them from entry ranks 55-69 to #1).
- **Heavy-tail jumps:** a Gaussian walk cannot produce the rockets the
  chart actually shows (a rank-29 game reaching #1 needs +3.4 in `s`,
  ~4 sigma under the pooled volatility). Every game-hour also carries a
  1% chance of an extra N(0, 1.0) jump, giving the forecast the heavy
  tails the real chart has.
- **Playing-surge jumps:** a game's `playing` count sometimes explodes
  while its rank barely moves — the rank lags the surge by 12-36 h. In
  the pooled archive, games whose playing grew by more than e^1 over the
  last two reads reached the top 10 within 24 h 58% of the time (base
  rate 11%). Such surge games get a jump of +U(1.5, 3.0) in the first
  24 h with probability 0.5.
- **Unnamed share:** every place keeps a minimum unnamed share (25% at
  place 1, tapering to 14% at places 6-10). The unnamed share is divided
  over the games on a scored read's chart that the forecast did not name,
  so it is the only way to cover games that enter the chart during the
  window — in the practice window the eventual #1 and #2 were games that
  were not in the history at all.

## Backtest

Scored with the exact log-score rule on the practice window
(2026-11-07T23:00Z to 2026-11-10T23:00Z, the 10 public archive reads
from [Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots)):

| forecast | log score |
|---|---|
| this entry (v3) | **-34.89** |
| previous version (v1) | -35.17 |
| house baseline | -43.51 |
| house model | -37.2 |
| best entry on the board at submission | -34.6 |

Rolling backtests at three earlier cutoffs (2026-11-04 / 11-05 / 11-06)
score -39.04 / -42.63 / -36.06 (mean -39.24), vs -40.50 / -44.42 /
-36.96 (mean -40.63) for v1 — better than v1 on the practice window and
on every rolling window (four-window mean -38.16 vs -39.26). A drift-decay
variant scored better on practice (-33.56) but worse on one rolling
window and was not used; neither was an empirical-transition ensemble
(mean gain +0.10 only, for much more complexity).

---

<sub>Entered on [make.fun](https://make.fun/?ref=9w5z28cm&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
