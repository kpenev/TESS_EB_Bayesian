#!/usr/bin/env python3

"""
Fit the light curve and SED starting from the SED-only seeds.

The seeds are the lowest cost minima of the cells of the SED-only grid search
(sed_grid_search.py, sed_minima_cells.py). Each seed fixes total mass, age and
[M/H]. For each seed and each argument of periastron (w) in a grid:

1. the period and eclipse time come from the BLS, and the eccentricity is set
   to match the eclipse times for the given w (`_match_eclipse_times` of
   FindStartingPositions);
2. the mass ratio and impact parameter are chosen to match both eclipse depths
   and the SED given the seed's stellar parameters (`_match_both_depths` with
   ``include_sed``: the SED barely constrains the mass ratio, the depths do,
   but without the SED the depths alone can pick a mass ratio the photometry
   rules out for these stars) and the limb darkening is fitted
   (`_fit_limbdark`);
3. starts that nearly repeat a better one are not fitted (`cull_starts()`): for
   nearly circular orbits most w values give practically the same start;
4. a least squares fit of all parameters follows. Once enough good fits have
   finished, a fit gaining too little over their median duration is stopped,
   and one far below the best that is improving too slowly to catch up is
   abandoned (`stop_slow_fit()`); function evaluation and wall-clock limits
   guard against endless runs.

w and w + 180 make different observed eclipses the primary one (the primary
star eclipsed), so both halves of the circle are tried. If neither BLS eclipse
is masked the orbit is taken as circular, where w only selects that geometry,
so only w = 0 and 180 are used.
"""

from bisect import bisect_right
from functools import partial
from multiprocessing import get_context
from types import SimpleNamespace
import logging
import time

import numpy
from astropy.table import Table

from binary import Binary
from command_line_util import create_parser
from find_starting_positions import FindStartingPositions
from log_likelihood import LogLikelihood
from sample_params import SampleParams
from sed_minima_cells import select_seeds
from utils import (
    default_give_up,
    fit_least_squares,
    get_age_fraction,
    params_to_sample,
)

_logger = logging.getLogger(__name__)

# The log-likelihood used by the worker processes (set before forking).
_target = None

# Shared by the worker processes (set before forking): the durations [s] of
# the finished fits, how many there are, and the lowest cost among them.
_progress = None


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument("tic_id", type=int, help="The TIC ID to fit.")
    parser.add_argument(
        "--sed-results",
        required=True,
        help="The .ecsv file written by sed_grid_search.py.",
    )
    parser.add_argument(
        "--omega-grid",
        type=float,
        nargs="+",
        default=[
            -150.0,
            -120.0,
            -60.0,
            -30.0,
            0.0,
            30.0,
            60.0,
            120.0,
            150.0,
            180.0,
        ],
        help="Arguments of periastron [deg] to start from (for circular orbits "
        "only 0 and 180 are used). w and w + 180 make different eclipses the "
        "primary one, so the full circle is needed; values near +-90 deg "
        "cannot shift the secondary eclipse from phase 0.5.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Use only this many seeds (lowest SED cost first).",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=20,
        help="Function evaluation limit of the least squares fit (not counting "
        "those for the numerical Jacobian).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1800.0,
        help="Wall-clock limit of the least squares fit [s].",
    )
    parser.add_argument(
        "--min-good-fits",
        type=int,
        default=10,
        help="Fits are only stopped for slow progress (see stop_slow_fit()) "
        "once at least this many good fits (within --good-delta of the best "
        "finished fit) have finished.",
    )
    parser.add_argument(
        "--good-delta",
        type=float,
        default=20.0,
        help="Finished fits within this of the best finished fit (in "
        "log-probability) are good; their median duration is the reference "
        "duration of stop_slow_fit().",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=1.0,
        help="Stop a fit whose log-probability rose by less than this over the "
        "last reference duration.",
    )
    parser.add_argument(
        "--abandon-time-factor",
        type=float,
        default=10.0,
        help="A fit far below the best finished one (see --abandon-below) is "
        "abandoned if, at its recent rate of improvement, it would need more "
        "than this many reference durations to catch up.",
    )
    parser.add_argument(
        "--abandon-below",
        type=float,
        default=100.0,
        help="A fit within this of the best finished fit (in log-probability) "
        "counts as caught up and is never abandoned.",
    )
    parser.add_argument(
        "--max-interpolation-error",
        type=float,
        default=0.01,
        help="Fail if the interpolated log-probability of any fit end point "
        "differs from the exact one by more than this (see "
        "check_interpolation()); the results are then saved with "
        "_failed_interpolation added to their name.",
    )
    parser.add_argument(
        "--cull-distance",
        type=float,
        default=1.0,
        help="Heuristic starts within this distance of a better one are not "
        "fitted (see cull_starts()).",
    )
    parser.add_argument(
        "--bad-sed",
        type=float,
        nargs=3,
        metavar=("THRESHOLD", "AMPLITUDE", "SHARPNESS"),
        default=None,
        help="Override the cached bad SED threshold, penalty amplitude and "
        "sharpness.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=8,
        help="How many starting points to fit in parallel.",
    )
    parser.add_argument(
        "--output",
        default="sed_seeded_fits_{tic_id:d}.ecsv",
        help="Where to save the results.",
    )
    return parser.parse_args()


def describe(sample):
    """Return a dict of the fit quality and key parameters of a sample."""

    params = _target.get_sample_params(sample)
    binary = Binary(from_mcmc=params)
    return {
        "log_prob": _target(sample)[0],
        "lc_log_like": _target.calc_lc_log_likelihood(binary, params.lc_sys),
        "sed_log_like": _target.calc_sed_log_likelihood(binary, params.sed_sys),
        "mtotal": params.mtotal,
        "mratio": params.mratio,
        "meh": params.meh,
        "age_gyr": params.age_gyr,
        "ecc": params.ecc,
        "w": params.w,
        "impact": params.primary_impact_param,
        "teff_primary": binary.component_teff["primary"],
        "teff_secondary": binary.component_teff["secondary"],
    }


def is_circular(finder):
    """True if neither BLS eclipse is masked (orbit treated as circular)."""

    # pylint: disable=protected-access
    return "masked" not in (
        finder._bls_eclipses["deeper"],
        finder._bls_eclipses["shallower"],
    )
    # pylint: enable=protected-access


def get_heuristic_start(seed, omega):
    """
    Return physical parameters matching the eclipses for one seed and w.

    Total mass, age and [M/H] come from the seed; eccentricity, mass ratio,
    impact parameter and limb darkening from the FindStartingPositions
    heuristics.
    """

    finder = FindStartingPositions(_target)
    bls = _target.best_fit_bls
    # pylint: disable=protected-access
    deeper_masked = finder._bls_eclipses["deeper"] == "masked"
    params = SampleParams(
        mtotal=float(seed["mtotal"]),
        mratio=float(seed["mratio"]),
        age_gyr=float(seed["age_gyr"]),
        meh=float(seed["meh"]),
        per=bls["period"][0] * (2 if is_circular(finder) else 1),
        ecc=0.0,
        w=omega,
        primary_impact_param=0.0,
        eclipse_time=(
            bls["masked_transit_time"] if deeper_masked else bls["transit_time"]
        ),
    )
    params = finder._match_eclipse_times(params, False)
    params = finder._match_both_depths(
        params, get_age_fraction(params, _target), include_sed=True
    )
    return finder._fit_limbdark(params)
    # pylint: enable=protected-access


def record_finished(seconds, cost):
    """Add a finished fit to the shared progress (worker process)."""

    with _progress.lock:
        _progress.seconds[_progress.num_finished.value] = seconds
        _progress.costs[_progress.num_finished.value] = cost
        _progress.num_finished.value += 1
        _progress.best_cost.value = min(_progress.best_cost.value, cost)


def stop_slow_fit(state, *, config):
    """
    Return why to stop a fit that is making too little progress, or None.

    Fits of one target differ widely in duration. The slowest are mostly far
    from the best solutions, or close to one but crawling, and can take most
    of the computing time. The reference duration is the median duration of
    the good finished fits (within ``config["good_delta"]`` of the best
    finished fit): how long a fit takes to reach a good solution for this
    target. It adapts to how expensive the likelihood is, slow bad fits do
    not inflate it, and waiting for ``config["min_good_fits"]`` good fits
    keeps the first, fastest finishers from making it too short.

    Once that many good fits have finished, a fit that has run longer than
    the reference is checked against the gain: the drop of its lowest cost
    (costs differ like -log_prob) over the last reference duration.

    - It is stopped if the gain is below ``config["min_gain"]``: a point
      that close to its optimum is good enough as a starting point.
    - It is abandoned if its lowest cost is more than
      ``config["abandon_below"]`` above the best finished fit and, gaining
      at that rate, it would need more than ``config["abandon_time_factor"]``
      reference durations to come within that of the best.

    Either way the fit's best point is kept as its end point.

    Args:
        state:    The state of the fit (see `utils.default_give_up()`); its
            ``cost_history`` (seconds, lowest cost) is kept here.

        config(dict):    The settings named above.
    """

    elapsed = time.monotonic() - state.start
    if not hasattr(state, "cost_history"):
        state.cost_history = ([], [])
    state.cost_history[0].append(elapsed)
    state.cost_history[1].append(state.best_cost)

    num_finished = _progress.num_finished.value
    best_cost = _progress.best_cost.value
    good = (
        numpy.array(_progress.costs[:num_finished])
        <= best_cost + config["good_delta"]
    )
    if good.sum() < config["min_good_fits"]:
        return None
    reference = numpy.median(
        numpy.array(_progress.seconds[:num_finished])[good]
    )
    if elapsed <= reference:
        return None
    gain = (
        state.cost_history[1][
            bisect_right(state.cost_history[0], elapsed - reference) - 1
        ]
        - state.best_cost
    )
    reason = (
        f"after {elapsed:.0f} s: its lowest cost dropped by {gain:.3g} in the "
        f"last {reference:.0f} s (the median duration of the {good.sum()} "
        "good finished fits)"
    )
    if gain < config["min_gain"]:
        return f"Stopped {reason}."
    excess = state.best_cost - best_cost
    if excess <= config["abandon_below"]:
        return None
    # Gaining ``gain`` per reference duration.
    needed_durations = (excess - config["abandon_below"]) / gain
    if needed_durations <= config["abandon_time_factor"]:
        return None
    return (
        f"Abandoned {reason}; it is {excess:.3g} above the best finished fit "
        f"and would need {needed_durations:.3g} times that to come within "
        f"{config['abandon_below']:g} of it (limit "
        f"{config['abandon_time_factor']:g})."
    )


def heuristic_start(arguments, edge_latent=10.0):
    """
    Return the summary of the heuristic start for one seed and w.

    Parameters the heuristics leave at the edge of their prior get latent
    values of +-``edge_latent`` (see `utils.params_to_sample()`), so the
    same seed and w always give the same start.
    """

    start_index, seed, omega = arguments
    summary = {"start": start_index, "omega_grid": omega}
    begin = time.time()
    try:
        sample = params_to_sample(
            get_heuristic_start(seed, omega), _target, edge_latent=edge_latent
        )
    except (AssertionError, ValueError, RuntimeError) as error:
        _logger.warning("Start %d heuristics failed: %s", start_index, error)
        summary["message"] = f"heuristics failed: {error}"
        return summary
    # One start hitting a bug should not end a run of hundreds of fits; the
    # traceback is logged and the start recorded as failed.
    except Exception as error:  # pylint: disable=broad-exception-caught
        _logger.exception("Start %d heuristics crashed", start_index)
        summary["message"] = (
            f"heuristics crashed: {type(error).__name__}: {error}"
        )
        return summary
    summary["heuristic_seconds"] = time.time() - begin
    summary["heuristic_x"] = sample
    summary.update(
        {f"heuristic_{key}": value for key, value in describe(sample).items()}
    )
    return summary


def cull_starts(summaries, threshold, scales=None):
    """
    Mark heuristic starts that nearly repeat a better one; return the rest.

    Each start is described by its total mass, mass ratio, log10(age), [M/H],
    eccentricity vector (e cos w, e sin w, so w stops mattering as e -> 0),
    impact parameter and which observed eclipse is the primary one, each
    divided by a scale. Going from the highest heuristic log-probability
    down, a start is kept unless it is within ``threshold`` (Euclidean
    distance in those units) of a start already kept; culled starts get a
    message naming the start they repeat. Starts of different eclipse
    geometries are never merged.

    Args:
        summaries([dict]):    The summaries from `heuristic_start()`;
            culled ones are modified in place.

        threshold(float):    See above.

        scales(dict or None):    The scale of each quantity. The stellar ones
            default to the kernel widths of the SED cells
            (`sed_minima_cells.select_seeds()`).

    Returns:
        [dict]:
            The kept summaries.
    """

    if scales is None:
        scales = {
            "mtotal": 0.05,
            "mratio": 0.1 / 3,
            "log_age": 0.1,
            "meh": 0.1,
            "e_cos_w": 0.001,
            "e_sin_w": 0.01,
            "impact": 0.02,
        }
    bls = _target.best_fit_bls
    # pylint: disable=protected-access
    deeper_masked = (
        FindStartingPositions(_target)._bls_eclipses["deeper"] == "masked"
    )
    # pylint: enable=protected-access
    deeper_time = (
        bls["masked_transit_time"] if deeper_masked else bls["transit_time"]
    )

    def get_coordinates(summary):
        params = _target.get_sample_params(summary["heuristic_x"])
        phase = ((params.eclipse_time - deeper_time) / params.per) % 1.0
        w = numpy.radians(params.w)
        return numpy.array(
            [
                params.mtotal / scales["mtotal"],
                params.mratio / scales["mratio"],
                numpy.log10(params.age_gyr) / scales["log_age"],
                params.meh / scales["meh"],
                params.ecc * numpy.cos(w) / scales["e_cos_w"],
                params.ecc * numpy.sin(w) / scales["e_sin_w"],
                params.primary_impact_param / scales["impact"],
                # The other eclipse is primary: never merged.
                numpy.inf if 0.25 < phase < 0.75 else 0.0,
            ]
        )

    built = [summary for summary in summaries if "heuristic_x" in summary]
    built.sort(key=lambda summary: -summary["heuristic_log_prob"])
    kept = []
    kept_coordinates = []
    for summary in built:
        coordinates = get_coordinates(summary)
        for other, other_coordinates in zip(kept, kept_coordinates):
            if numpy.isinf(coordinates[-1]) != numpy.isinf(
                other_coordinates[-1]
            ):
                continue
            distance = numpy.linalg.norm(
                coordinates[:-1] - other_coordinates[:-1]
            )
            if distance <= threshold:
                summary["message"] = (
                    f"culled: within {distance:.3g} of start "
                    f"{other['start']}"
                )
                break
        else:
            kept.append(summary)
            kept_coordinates.append(coordinates)
    _logger.info(
        "Kept %d of %d heuristic starts (%d failed, %d culled).",
        len(kept),
        len(summaries),
        len(summaries) - len(built),
        len(built) - len(kept),
    )
    return kept


def fit_start(arguments):
    """Run the least squares fit from a heuristic start (worker process)."""

    summary, config = arguments
    start_index = summary["start"]
    sample = summary["heuristic_x"]
    begin = time.time()
    try:
        result = fit_least_squares(
            _target,
            sample,
            max_nfev=config["max_nfev"],
            progress_interval=600.0,
            give_up=(
                partial(default_give_up, time_limit=config["time_limit"]),
                partial(stop_slow_fit, config=config),
            ),
        )
    except (AssertionError, ValueError) as error:
        _logger.warning("Start %d fit failed: %s", start_index, error)
        summary["message"] = f"fit failed: {error}"
        return summary
    except Exception as error:  # pylint: disable=broad-exception-caught
        _logger.exception("Start %d fit crashed", start_index)
        summary["message"] = f"fit crashed: {type(error).__name__}: {error}"
        return summary
    summary["fit_seconds"] = time.time() - begin
    record_finished(summary["fit_seconds"], result.cost)
    summary["fit_status"] = result.status
    summary["message"] = result.message
    summary["fit_nfev"] = result.nfev
    # Largest gradient component at the end; tells a true minimum from a fit
    # stopped by a tolerance while still descending.
    summary["fit_optimality"] = result.optimality
    summary["fit_x"] = result.x
    summary.update(
        {f"fit_{key}": value for key, value in describe(result.x).items()}
    )
    _logger.info(
        "Start %d done: heuristic log_prob %s, fit log_prob %s",
        start_index,
        repr(summary["heuristic_log_prob"]),
        repr(summary["fit_log_prob"]),
    )
    return summary


def interpolation_error(sample):
    """Return the exact minus interpolated log-probability (worker process)."""

    exact = _target(sample, exact=True)[0]
    interpolated = _target(sample)[0]
    # Equal also covers both -inf (unphysical either way).
    return 0.0 if exact == interpolated else exact - interpolated


def check_interpolation(summaries, pool):
    """
    Compare the exact and interpolated log-probability at every fit end point.

    The light-curve model is interpolated from a grid of orbital phases (see
    `LogLikelihood.exact_supersample_step`); this measures the error that
    causes where it matters. Each summary with an end point gets an
    ``interpolation_error`` entry (exact minus interpolated).

    Returns:
        (float, int or None):
            The largest absolute error and the start it is at (None if no fit
            finished).
    """

    fitted = [summary for summary in summaries if "fit_x" in summary]
    if not fitted:
        return 0.0, None
    errors = pool.map(
        interpolation_error,
        [summary["fit_x"] for summary in fitted],
        chunksize=1,
    )
    for summary, error in zip(fitted, errors):
        summary["interpolation_error"] = error
    errors = numpy.abs(errors)
    worst = int(numpy.argmax(errors))
    _logger.info(
        "Exact minus interpolated log-probability at %d end points: largest "
        "|difference| %.3g (start %d), median %.3g.",
        len(fitted),
        errors[worst],
        fitted[worst]["start"],
        numpy.median(errors),
    )
    return errors[worst], fitted[worst]["start"]


def main(config):
    """Fit from all seeds and w values and report the results."""

    global _target  # pylint: disable=global-statement
    _target = LogLikelihood(config.tic_id)
    if config.bad_sed is not None:
        # pylint: disable=protected-access
        _target._sed = (
            _target._sed[:2] + (tuple(config.bad_sed),) + _target._sed[3:]
        )
        # pylint: enable=protected-access

    seeds = select_seeds(Table.read(config.sed_results))[1]
    if config.max_seeds is not None:
        seeds = seeds[: config.max_seeds]
    omegas = (
        [0.0, 180.0]
        if is_circular(FindStartingPositions(_target))
        else config.omega_grid
    )
    starts = [(cell, row, omega) for cell, row, _ in seeds for omega in omegas]
    _logger.info(
        "Fitting from %d seeds x %d w values = %d starts.",
        len(seeds),
        len(omegas),
        len(starts),
    )

    context = get_context("fork")
    with context.Pool(config.num_parallel) as pool:
        summaries = pool.map(
            heuristic_start,
            [
                (index, row, omega)
                for index, (_, row, omega) in enumerate(starts)
            ],
            chunksize=1,
        )
    kept = cull_starts(summaries, config.cull_distance)

    fit_config = {
        "max_nfev": config.max_nfev,
        "time_limit": config.time_limit,
        "min_good_fits": config.min_good_fits,
        "good_delta": config.good_delta,
        "min_gain": config.min_gain,
        "abandon_time_factor": config.abandon_time_factor,
        "abandon_below": config.abandon_below,
    }
    global _progress  # pylint: disable=global-statement
    _progress = SimpleNamespace(
        lock=context.Lock(),
        seconds=context.Array("d", max(len(kept), 1), lock=False),
        costs=context.Array("d", max(len(kept), 1), lock=False),
        num_finished=context.Value("i", 0, lock=False),
        best_cost=context.Value("d", numpy.inf, lock=False),
    )
    with context.Pool(config.num_parallel) as pool:
        fitted = pool.map(
            fit_start, [(summary, fit_config) for summary in kept], chunksize=1
        )
    for summary in fitted:
        summaries[summary["start"]] = summary
    with context.Pool(config.num_parallel) as pool:
        worst_error, worst_start = check_interpolation(summaries, pool)

    rows = []
    for summary, (cell, seed, _) in zip(summaries, starts):
        row = {
            "cell": str(tuple(int(index) for index in cell)),
            "sed_cost": float(seed["cost"]),
            "seed_mtotal": float(seed["mtotal"]),
            "seed_mratio": float(seed["mratio"]),
            "seed_meh": float(seed["meh"]),
            "seed_age_gyr": float(seed["age_gyr"]),
        }
        row.update(summary)
        rows.append(row)
    output_fname = config.output.format(tic_id=config.tic_id)
    interpolation_failed = worst_error > config.max_interpolation_error
    if interpolation_failed:
        # Kept for inspection, but not under the name the next steps (and
        # prepare_sampling.sh's check for a finished step) look for.
        output_fname = (
            output_fname.rsplit(".", 1)[0]
            + "_failed_interpolation."
            + output_fname.rsplit(".", 1)[1]
        )
    columns = sorted({key for row in rows for key in row})
    Table(
        {
            key: [row.get(key, numpy.nan) for row in rows]
            for key in columns
            if key not in ("fit_x", "heuristic_x", "message")
        }
    ).write(output_fname, overwrite=True)
    numpy.save(
        output_fname.rsplit(".", 1)[0] + "_x.npy",
        numpy.array(
            [
                row.get(
                    "fit_x", numpy.full(len(SampleParams._fields), numpy.nan)
                )
                for row in rows
            ]
        ),
    )

    print(f"{len(rows)} starts. Sorted by fitted log-probability:")
    for row in sorted(
        rows, key=lambda entry: -entry.get("fit_log_prob", -numpy.inf)
    ):
        print(
            f"start {row['start']:3d} {row['cell']:>16s} w0 "
            f"{row['omega_grid']:6.1f} SED cost {row['sed_cost']:6.2f} | "
            f"heuristic {row.get('heuristic_log_prob', numpy.nan):.8g} | fit "
            f"{row.get('fit_log_prob', numpy.nan):.8g} (LC "
            f"{row.get('fit_lc_log_like', numpy.nan):.8g}, SED "
            f"{row.get('fit_sed_log_like', numpy.nan):.5g}) Mtot "
            f"{row.get('fit_mtotal', numpy.nan):.3f} q "
            f"{row.get('fit_mratio', numpy.nan):.3f} [M/H] "
            f"{row.get('fit_meh', numpy.nan):.2f} age "
            f"{row.get('fit_age_gyr', numpy.nan):.4g} e "
            f"{row.get('fit_ecc', numpy.nan):.3f} Teff "
            f"{row.get('fit_teff_primary', numpy.nan):.0f}/"
            f"{row.get('fit_teff_secondary', numpy.nan):.0f} | "
            f"{row.get('message', '')}"
        )
    if interpolation_failed:
        raise RuntimeError(
            f"The interpolated log-probability differs from the exact one by "
            f"{worst_error:.3g} (more than --max-interpolation-error "
            f"{config.max_interpolation_error:g}) at the end point of start "
            f"{worst_start}; results saved to {output_fname} for inspection."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(name)s: %(message)s",
    )
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    main(parse_command_line())
