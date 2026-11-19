# gta6-trophy-rarity-forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jinny0784/gta6-trophy-rarity-forecaster/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jinny0784/gta6-trophy-rarity-forecaster/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's **"Forecast how many GTA VI players earn each trophy
in week one"** (gta6-trophy-rarity-forecast-01). `main.py` (Python 3.13,
standard library only) reads each case file in the input directory — every
read the house made of Exophase's Grand Theft Auto VI PS5 trophy page before
the case's checkpoint — and writes five quantiles (5%, 25%, 50%, 75%, 95%)
of each trophy's earn rate at the case's target.

## Run it

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

With the variables unset it reads `/input` and writes `/output`, as the
grader runs it. It is deterministic: two runs on the same case write the
same bytes.

## How it forecasts

Scoring is pinball loss on the logit scale, so the model lives there too.

- **Log-time growth curves.** With x = ln(1 + hours since release), a
  trophy's logit earn rate is fitted as a straight line by weighted least
  squares over the distinct snapshots of the page (Exophase refreshes in
  steps; repeated identical reads are collapsed and weighted by how long
  each state held). A line in log-time, pushed back through the logistic
  function, is a saturating growth curve: fast early, flattening later.
- **Partial pooling.** Each trophy's slope is shrunk toward the mean slope
  of its tier (platinum/gold/silver/bronze), or of all trophies for small
  tiers, with strength from the slope's standard error and the spread of
  slopes across trophies. Sparse, flat or single-read series inherit the
  pooled slope anchored on their last observed rate instead of a wild fit.
- **Saturation guard.** A log-time line aims a rate at 100% given forever,
  but real trophies level off at the share of players who ever do the
  thing. The slope carried past the last read therefore blends the
  whole-history slope with the recent stretch's slope — a curve bending
  toward its ceiling shows it there first — and the increment is damped
  slightly toward the anchor.
- **Honest spread.** The predictive variance adds the fit's posterior
  variance, the pooled residual variance and a random-walk term growing
  with the square root of the horizon in days, so the 90% band is wide
  where the data is thin or the horizon long.
- **The stale-page case.** If a case's history is one unchanging page, no
  curve can be fitted; the forecast then prices the refresh the page is
  due as a two-branch mixture — most weight on Exophase refreshing before
  the target and diluting every rate the newly tracked players have not
  yet earned (dilution scaled by 1 − rate, so a trophy everyone holds is
  left alone), the rest on the page staying as it is.

Rates are fitted inside the scoring band (0.01–99.99) so logits stay
finite; a trophy never observed falls back to the cross-section of the
last read. Output rows are `{"id": n, "q": [q05, q25, q50, q75, q95]}`,
non-decreasing, rounded to 4 decimals.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
