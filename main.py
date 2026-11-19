"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (<id>.in) in the input directory and writes, for each
trophy, five quantiles of its Exophase earn rate at the case's target time.

Model, per trophy, on the logit scale (the scale the contest scores on):

  * Time runs as x = ln(1 + hours since release midnight). On that axis a
    trophy's logit earn rate is close to a straight line while the player
    base is young: holders accumulate fast, then the curve bends over and
    saturates, which is exactly what a line in log-time does once it is
    pushed back through the logistic function.
  * Each trophy's line is fitted by weighted least squares over the distinct
    snapshots in the case history (Exophase refreshes in steps, so repeated
    identical reads are collapsed and weighted by how long they held).
  * Slopes are partially pooled: a trophy's slope is shrunk toward the mean
    slope of its tier (or of all trophies, for small tiers) with strength set
    by the slope's own standard error and the spread of slopes across
    trophies. A trophy with a flat, single or missing series inherits the
    pooled slope, anchored on its last observed rate.
  * The predictive spread combines the fit's posterior variance, the pooled
    residual variance and a random-walk term that grows with the square root
    of the horizon in days.
  * If the whole history is one unchanging page (the launch-night practice
    shape), no line can be fitted at all. The forecast then prices in the
    refresh the page is due: a mixture that mostly expects Exophase to
    refresh before the target, multiplying the tracked players and diluting
    every rate the newcomers have not yet earned, with the dilution scaled
    by (1 - rate) so a trophy every player holds is left alone.

Deterministic, standard library only, no network. Runs in well under a
second per case.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

NORM = NormalDist()
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# The scoring clamps rates into this band before the logit; fit inside it too.
RATE_LO = 0.01
RATE_HI = 99.99

# Random-walk spread of a logit rate, per square-root day of horizon. The
# house persistence baseline uses 0.5 here; backtests on synthetic
# launch-week histories (log-time curves seen through step refreshes, with
# and without saturation ceilings and idiosyncratic jumps) put the
# pinball-optimal value nearer 0.3, the fit's own variance term carrying
# the rest of the uncertainty.
RW_PER_SQRT_DAY = 0.3

# A log-time line aims a trophy's rate at 100% given forever; real trophies
# saturate at the share of players who ever do the thing, often far below
# that. Two guards against the overshoot: the slope carried past the last
# read blends the whole-history slope with the recent slope (a curve that
# is bending over shows it there first), and the blended increment is
# damped a little toward the anchor.
GROWTH_DAMP = 0.9
RECENT_SLOPE_W = 0.8

# Floors and caps that keep pooled quantities sane on thin data.
SIGMA2_FLOOR = 0.02 ** 2
SIGMA2_CAP = 1.5 ** 2
TAU_FLOOR = 0.15
DEFAULT_SLOPE = 0.9

# The due-refresh mixture used when a case's history never changes: how far
# a refresh dilutes a small rate on the logit scale, how likely at least one
# refresh is per hour of staleness-plus-horizon (refreshes land hours apart),
# and the spreads of the refreshed / unrefreshed branches.
REFRESH_DROP = 1.5
REFRESH_GAP_HOURS = 4.0
REFRESH_SD = 0.55
STEADY_SD = 0.30


def parse_time(text):
    """ISO 8601 UTC text -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, rate))


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, whole numbers as ints (JSON-compact)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


class Snapshot:
    """One distinct state of the page: the reads that repeat it collapse."""

    __slots__ = ("hours", "players", "rates")

    def __init__(self, hours, players, rates):
        self.hours = hours
        self.players = players
        self.rates = rates


def snapshots_of(case, epoch):
    """Collapse the case history into distinct page states, oldest first."""
    snaps = []
    for read in case["history"]:
        when = parse_time(read.get("slot") or read["at"])
        hours = max(0.0, hours_between(epoch, when))
        rates = tuple(read["rates"])
        if snaps and snaps[-1].players == read["players_tracked"] and snaps[-1].rates == rates:
            continue
        snaps.append(Snapshot(hours, read["players_tracked"], rates))
    return snaps


def weights_of(snaps, checkpoint_hours):
    """One weight per snapshot: how long it held, gently discounted by age.

    Scaled to average 1, so the weights count like observations: the slope
    prior's precision and the fits' standard errors then share one scale,
    and partial pooling shrinks by the amount the data actually supports.
    """
    weights = []
    for i, snap in enumerate(snaps):
        end = snaps[i + 1].hours if i + 1 < len(snaps) else checkpoint_hours
        held = max(0.05, end - snap.hours)
        age = max(0.0, checkpoint_hours - snap.hours)
        weights.append((0.3 + min(held, 12.0) / 4.0) * 0.5 ** (age / 96.0))
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights]


def fit_line(xs, ys, ws, sigma2, prior_slope=None, tau=TAU_FLOOR):
    """Weighted least squares y ~ a + b*x, the slope shrunk to a prior.

    The prior enters as a ridge on the slope alone, scaled by the residual
    variance (ridge = sigma2 / tau^2) so it counts as what it is: one
    pseudo-observation of the slope, weighed against fits whose own noise
    is sigma2. The intercept stays free. Returns (a, b, minv) with minv the
    inverse of the 2x2 system matrix, so the prediction variance at x is
    sigma2 * (1 + [1, x] minv [1, x]^T).
    """
    sw = sum(ws)
    sx = sum(w * x for x, w in zip(xs, ws))
    sy = sum(w * y for y, w in zip(ys, ws))
    sxx = sum(w * x * x for x, w in zip(xs, ws))
    sxy = sum(w * x * y for x, y, w in zip(xs, ys, ws))
    ridge = sigma2 / (tau * tau)
    m00, m01, m11 = sw, sx, sxx + ridge
    r0, r1 = sy, sxy + (prior_slope * ridge if prior_slope is not None else 0.0)
    det = m00 * m11 - m01 * m01
    if det <= 1e-12:
        return None
    a = (r0 * m11 - r1 * m01) / det
    b = (m00 * r1 - sx * r0) / det
    minv = ((m11 / det, -m01 / det), (-m01 / det, m00 / det))
    return a, b, minv


def ols_slope(xs, ys, ws):
    """Plain weighted slope and its standard error, or None if unidentified."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-9:
        return None
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    slope = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(1.0, sum(1 for w in ws if w > 0) - 2)
    var = sum(w * r * r for r, w in zip(resid, ws)) / dof
    se = math.sqrt(max(var, 1e-6) / sxx)
    return slope, se, var, dof


def pool_prior(fits):
    """Precision-weighted mean slope and between-trophy spread (DL-style)."""
    if not fits:
        return None, TAU_FLOOR
    tau2 = 0.25
    mean = 0.0
    for _ in range(3):
        ws = [1.0 / (se * se + tau2) for _, se, _, _ in fits]
        mean = sum(w * b for w, (b, _, _, _) in zip(ws, fits)) / sum(ws)
        q = sum((b - mean) ** 2 / (se * se) for b, se, _, _ in fits)
        denom = sum(1.0 / (se * se) for _, se, _, _ in fits)
        tau2 = max(TAU_FLOOR ** 2, (q - (len(fits) - 1)) / max(denom, 1e-9))
    return mean, math.sqrt(tau2)


def mixture_quantiles(base, drop, p_refresh, levels):
    """Quantiles of a two-branch logit mixture: refreshed vs still stale.

    With probability p_refresh the page refreshes before the target and the
    rate's logit falls by `drop` (spread REFRESH_SD); otherwise it drifts
    only a little (spread STEADY_SD). Quantiles are read off the mixture CDF
    by bisection, so the asymmetry of the bet is kept.
    """
    def cdf(v):
        return (p_refresh * NORM.cdf((v - (base - drop)) / REFRESH_SD)
                + (1.0 - p_refresh) * NORM.cdf((v - base) / STEADY_SD))

    out = []
    for level in levels:
        lo, hi = base - drop - 6.0, base + 6.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < level:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    levels = case.get("levels") or LEVELS
    zs = [NORM.inv_cdf(level) for level in levels]

    release = case["game"].get("release_date", "2026-11-19")
    epoch = parse_time(release + "T00:00:00Z")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    checkpoint_hours = max(0.0, hours_between(epoch, checkpoint))
    target_hours = max(checkpoint_hours, hours_between(epoch, target))
    x_last = math.log1p(checkpoint_hours)
    x_target = math.log1p(target_hours)
    horizon_days = case["horizon_minutes"] / 1440.0
    rw_var = (RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2

    snaps = snapshots_of(case, epoch)
    if not snaps:
        # No history at all: a wide, low prior for every trophy.
        rows = []
        for trophy in trophies:
            mu = logit_pct(1.0)
            qs = [inv_logit_pct(mu + z * 1.5) for z in zs]
            rows.append({"id": trophy["id"], "q": [round4(q) for q in qs]})
        return {"trophies": rows}

    weights = weights_of(snaps, checkpoint_hours)
    snap_x = [math.log1p(s.hours) for s in snaps]

    # Per-trophy observation series on the logit scale.
    series = []
    for i in range(n):
        pts = [(snap_x[k], logit_pct(snaps[k].rates[i]), weights[k])
               for k in range(len(snaps)) if snaps[k].rates[i] is not None]
        series.append(pts)

    # First pass: plain fits wherever a slope is identified at all.
    raw_fits = []
    for pts in series:
        fit = None
        if len(pts) >= 3:
            fit = ols_slope([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        raw_fits.append(fit)
    fittable = [f for f in raw_fits if f is not None]

    # Pooled residual variance across every fittable trophy.
    if fittable:
        num = sum(f[2] * f[3] for f in fittable)
        den = sum(f[3] for f in fittable)
        sigma2 = min(SIGMA2_CAP, max(SIGMA2_FLOOR, num / max(den, 1e-9)))
    else:
        sigma2 = 0.25

    # Slope priors: one per tier with enough fittable trophies, else global.
    global_prior, global_tau = pool_prior(fittable)
    tier_fits = {}
    for trophy, fit in zip(trophies, raw_fits):
        if fit is not None:
            tier_fits.setdefault(trophy["tier"], []).append(fit)
    tier_priors = {}
    for tier, fits in tier_fits.items():
        if len(fits) >= 4:
            tier_priors[tier] = pool_prior(fits)

    def prior_for(trophy):
        if trophy["tier"] in tier_priors:
            return tier_priors[trophy["tier"]]
        return global_prior, global_tau

    flat_history = len(snaps) <= 1
    # How long the page has shown exactly what the last snapshot shows.
    stale_hours = checkpoint_hours - (snaps[-1].hours if snaps else checkpoint_hours)
    p_refresh = 1.0 - math.exp(-(stale_hours + case["horizon_minutes"] / 60.0) / REFRESH_GAP_HOURS)

    rows = []
    for i, trophy in enumerate(trophies):
        pts = series[i]
        last_rate = None
        for snap in reversed(snaps):
            if snap.rates[i] is not None:
                last_rate = snap.rates[i]
                break

        if flat_history:
            base = logit_pct(last_rate if last_rate is not None else 1.0)
            share = (last_rate if last_rate is not None else 1.0) / 100.0
            drop = REFRESH_DROP * (1.0 - share)
            qs = [inv_logit_pct(v) for v in mixture_quantiles(base, drop, p_refresh, levels)]
            rows.append({"id": trophy["id"], "q": [round4(q) for q in qs]})
            continue

        prior_slope, tau = prior_for(trophy)
        if prior_slope is None:
            prior_slope = DEFAULT_SLOPE
        mu = None
        var = None
        if len(pts) >= 2:
            fitted = fit_line([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts],
                              sigma2, prior_slope, max(tau, TAU_FLOOR))
            if fitted is not None:
                a, b, minv = fitted
                # The slope the forecast rides on: the fitted slope blended
                # with the slope of the recent stretch of the series, where
                # a curve bending toward its ceiling shows up first.
                forward = b
                xs_pts = [p[0] for p in pts]
                split = min(xs_pts) + 0.6 * (max(xs_pts) - min(xs_pts))
                recent = [p for p in pts if p[0] >= split]
                if len(recent) >= 3:
                    rfit = ols_slope([p[0] for p in recent], [p[1] for p in recent],
                                     [p[2] for p in recent])
                    if rfit is not None:
                        forward = (1.0 - RECENT_SLOPE_W) * b + RECENT_SLOPE_W * rfit[0]
                mu = a + b * x_last + GROWTH_DAMP * forward * (x_target - x_last)
                lever = (minv[0][0] + 2.0 * minv[0][1] * x_target
                         + minv[1][1] * x_target * x_target)
                var = sigma2 * (1.0 + max(0.0, lever))
        if mu is None:
            # Too little of a series to fit: pooled slope off the last rate.
            anchor = logit_pct(last_rate) if last_rate is not None else None
            if anchor is None:
                peers = [logit_pct(s.rates[j]) for s in snaps[-1:]
                         for j in range(n) if s.rates[j] is not None]
                anchor = sum(peers) / len(peers) if peers else logit_pct(1.0)
            mu = anchor + GROWTH_DAMP * prior_slope * (x_target - x_last)
            var = sigma2 + tau * tau * (x_target - x_last) ** 2 + 0.25

        mu = min(9.5, max(-9.5, mu))
        sd = math.sqrt(max(var, 0.04 ** 2) + rw_var)
        qs = [inv_logit_pct(mu + z * sd) for z in zs]
        # Rounding can only flatten the order; enforce it outright.
        qs = [max(qs[k], qs[k - 1]) if k else qs[k] for k in range(len(qs))]
        rows.append({"id": trophy["id"], "q": [round4(q) for q in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
