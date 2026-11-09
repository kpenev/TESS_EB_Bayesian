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
3. a least squares fit of all parameters follows, limited in function
   evaluations, wall-clock time and stalling, so a stuck fit cannot hold up the
   rest. A fit running much longer than the finished ones while still far
   below the best of them is abandoned (see `abandon_hopeless()`).

w and w + 180 make different observed eclipses the primary one (the primary
star eclipsed), so both halves of the circle are tried. If neither BLS eclipse
is masked the orbit is taken as circular, where w only selects that geometry,
so only w = 0 and 180 are used.
"""

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
        "--stall-evaluations",
        type=int,
        default=200,
        help="Stop the fit if the cost has not improved by a fraction of "
        "--stall-tolerance in this many residual evaluations.",
    )
    parser.add_argument(
        "--stall-tolerance",
        type=float,
        default=1e-4,
        help="See --stall-evaluations.",
    )
    parser.add_argument(
        "--abandon-time-factor",
        type=float,
        default=5.0,
        help="A fit running longer than this many times the median duration "
        "of the finished fits is abandoned if it is far from the best (see "
        "--abandon-below).",
    )
    parser.add_argument(
        "--abandon-below",
        type=float,
        default=100.0,
        help="How far below the best finished fit (in log-probability) a slow "
        "fit must be to be abandoned.",
    )
    parser.add_argument(
        "--abandon-min-finished",
        type=int,
        default=10,
        help="Fits are only abandoned once at least this many have finished.",
    )
    parser.add_argument(
        "--abandon-min-fraction",
        type=float,
        default=0.1,
        help="... and at least this fraction of all the fits.",
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
        _progress.num_finished.value += 1
        _progress.best_cost.value = min(_progress.best_cost.value, cost)


def abandon_hopeless(state, *, min_finished, time_factor, max_cost_excess):
    """
    Return why to abandon a fit that is slow and far from the best, or None.

    Fits of one target differ widely in duration, and the slowest ones are
    mostly far from the best solutions (still improving, so the stall check
    does not stop them). Once ``min_finished`` fits have finished, a fit
    running longer than ``time_factor`` times their median duration is
    abandoned if its lowest cost so far is more than ``max_cost_excess``
    above the lowest of the finished fits (costs differ like -log_prob). The
    reference is the target's own fits, so it adapts to how expensive its
    likelihood is.

    Args:
        state:    The state of the fit (see `utils.default_give_up()`).

        min_finished(int):    How many fits must have finished first.

        time_factor(float):    How many times the median duration a fit may
            run before it is checked.

        max_cost_excess(float):    How far above the best finished fit (in
            cost, i.e. -log_prob) a checked fit may be.
    """

    num_finished = _progress.num_finished.value
    if num_finished < min_finished:
        return None
    elapsed = time.monotonic() - state.start
    median = numpy.median(_progress.seconds[:num_finished])
    if elapsed <= time_factor * median:
        return None
    excess = state.best_cost - _progress.best_cost.value
    if excess <= max_cost_excess:
        return None
    return (
        f"Abandoned after {elapsed:.0f} s ({time_factor:g} x the median "
        f"{median:.0f} s of {num_finished} finished fits): its lowest cost is "
        f"{excess:.3g} above the best finished fit."
    )


def fit_start(arguments):
    """Find the heuristic start and run the least squares fit from it."""

    start_index, seed, omega, config = arguments
    summary = {"start": start_index, "omega_grid": omega}
    begin = time.time()
    try:
        sample = params_to_sample(get_heuristic_start(seed, omega), _target)
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
    summary.update(
        {f"heuristic_{key}": value for key, value in describe(sample).items()}
    )

    begin = time.time()
    try:
        result = fit_least_squares(
            _target,
            sample,
            max_nfev=config["max_nfev"],
            progress_interval=600.0,
            give_up=(
                partial(
                    default_give_up,
                    time_limit=config["time_limit"],
                    stall_evaluations=config["stall_evaluations"],
                    stall_tolerance=config["stall_tolerance"],
                ),
                partial(
                    abandon_hopeless,
                    min_finished=config["abandon_min_finished"],
                    time_factor=config["abandon_time_factor"],
                    max_cost_excess=config["abandon_below"],
                ),
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

    fit_config = {
        "max_nfev": config.max_nfev,
        "time_limit": config.time_limit,
        "stall_evaluations": config.stall_evaluations,
        "stall_tolerance": config.stall_tolerance,
        "abandon_time_factor": config.abandon_time_factor,
        "abandon_below": config.abandon_below,
        "abandon_min_finished": max(
            config.abandon_min_finished,
            int(numpy.ceil(config.abandon_min_fraction * len(starts))),
        ),
    }
    context = get_context("fork")
    global _progress  # pylint: disable=global-statement
    _progress = SimpleNamespace(
        lock=context.Lock(),
        seconds=context.Array("d", len(starts), lock=False),
        num_finished=context.Value("i", 0, lock=False),
        best_cost=context.Value("d", numpy.inf, lock=False),
    )
    with context.Pool(config.num_parallel) as pool:
        summaries = pool.map(
            fit_start,
            [
                (index, row, omega, fit_config)
                for index, (_, row, omega) in enumerate(starts)
            ],
            chunksize=1,
        )

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
    columns = sorted({key for row in rows for key in row})
    Table(
        {
            key: [row.get(key, numpy.nan) for row in rows]
            for key in columns
            if key not in ("fit_x", "message")
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(name)s: %(message)s",
    )
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    main(parse_command_line())
