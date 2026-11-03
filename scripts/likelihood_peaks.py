#!/usr/bin/env python3

"""Find likelihood peaks to use as extra bound points for dynesty."""

from collections import namedtuple
from multiprocessing import Pool
import logging
from os import path
from traceback import format_exc

from astropy.table import Table
import h5py
import numpy
from scipy.stats import norm

from general_purpose_python_modules.multiprocessing_util import (
    setup_process,
    setup_process_map,
)

from command_line_util import create_parser
from find_starting_positions import (
    FindStartingPositions,
    add_scenario_grid_arguments,
)
from log_likelihood import LogLikelihood
from paths import samples as samples_fname_pattern
from paths import release as release_fname_pattern
from sample_params import SampleParams
from utils import fit_least_squares, params_to_sample

_logger = logging.getLogger(__name__)

# The dynesty prior transform maps the unit cube coordinate of w to
# w = W_MIN + W_PERIOD * u, so w covers exactly one period, matching the
# (-180, 180) convention of FindStartingPositions.
W_MIN, W_PERIOD = -180.0, 360.0

StartingPoint = namedtuple(
    "StartingPoint", ["source", "sample", "scenario_index", "scenario"]
)
StartingPoint.__doc__ = """
A place to start a least-squares fit from.

Either ``sample`` (emcee latent coordinates) is given, or ``scenario`` and
``scenario_index`` (from ``FindStartingPositions.get_init_scenarios()``),
which is turned into a sample by the heuristic optimization when refined.
``source`` records where the starting point came from.
"""

PeakCandidate = namedtuple(
    "PeakCandidate",
    ["source", "start_sample", "lstsq_sample", "log_prob", "jtj"],
)
PeakCandidate.__doc__ = """
The result of refining one starting point.

``start_sample`` and ``lstsq_sample`` are in emcee latent coordinates.
``log_prob`` is the emcee log-posterior at ``lstsq_sample``: the quantity
the least-squares fit maximizes. ``jtj`` is J^T J of the fit residuals at
``lstsq_sample`` (21 x 21), the local curvature used to tell peaks apart.
"""


def scenario_grid_starts(log_likelihood, config):
    """
    Return one starting point per scenario of the ``FindStartingPositions``
    grid of (log-age, [M/H], w).

    The heuristic optimization of each scenario is done later, by
    ``refine()``, so it runs in parallel.
    """

    scenarios = FindStartingPositions(log_likelihood).get_init_scenarios(
        config
    )
    return [
        StartingPoint(f"scenario_grid:{index}", None, index, scenario)
        for index, scenario in enumerate(scenarios)
    ]


def previous_fit_start(log_likelihood, samples_fname, release_fname):
    """
    Return the highest log-probability sample of a previous emcee fit.

    The samples file is used if it exists: only ``log_prob`` for the filled
    iterations of each group and the selected chain entry are read, giving
    the maximum over all saved samples, already in latent coordinates. The
    release file, which holds a subset of the samples in physical
    parameters, is used only if there is no samples file.

    Returns:
        StartingPoint or None:
            None if neither file exists.
    """

    if path.exists(samples_fname):
        best_log_prob, best_sample, best_group = -numpy.inf, None, None
        with h5py.File(samples_fname, "r") as samples_file:
            for group_name, group in samples_file.items():
                num_filled = group.attrs.get("iteration", 0)
                if "log_prob" not in group or num_filled == 0:
                    continue
                log_prob = group["log_prob"][:num_filled]
                # (step, walker) of the largest log_prob in this group
                index = numpy.unravel_index(
                    numpy.nanargmax(log_prob), log_prob.shape
                )
                if log_prob[index] > best_log_prob:
                    best_log_prob = log_prob[index]
                    best_sample = group["chain"][index]
                    best_group = group_name
        if best_sample is not None:
            _logger.info(
                "Previous fit maximum log_prob %s found in %s/%s",
                repr(best_log_prob),
                samples_fname,
                best_group,
            )
            return StartingPoint(
                f"previous_fit:{samples_fname}:{best_group}",
                best_sample,
                None,
                None,
            )

    if path.exists(release_fname):
        release = Table.read(release_fname)
        best_row = release[numpy.nanargmax(release["log_prob"])]
        _logger.info(
            "Previous fit maximum log_prob %s found in %s",
            repr(best_row["log_prob"]),
            release_fname,
        )
        params = SampleParams(
            *(best_row[field] for field in SampleParams._fields)
        )
        return StartingPoint(
            f"previous_fit:{release_fname}",
            params_to_sample(params, log_likelihood),
            None,
            None,
        )

    _logger.info(
        "No previous fit found (%s, %s)", samples_fname, release_fname
    )
    return None


def refine(start, log_likelihood):
    """
    Least-squares refine one starting point.

    Scenario starting points are first turned into a sample with the
    heuristic ``FindStartingPositions.optimize_scenario()``.

    Returns:
        PeakCandidate or None:
            None if the heuristic optimization or the fit failed; the
            reason is logged.
    """

    try:
        sample = start.sample
        if sample is None:
            params = FindStartingPositions(log_likelihood).optimize_scenario(
                start.scenario_index, start.scenario
            )
            sample = params_to_sample(params, log_likelihood, True)
        fit = fit_least_squares(log_likelihood, sample)
        return PeakCandidate(
            source=start.source,
            start_sample=sample,
            lstsq_sample=fit.x,
            log_prob=log_likelihood(fit.x)[0],
            jtj=fit.jac.T @ fit.jac,
        )
    # One bad starting point should not stop the others.
    except Exception:  # pylint: disable=broad-exception-caught
        _logger.warning(
            "Refining starting point from %s failed:\n%s",
            start.source,
            format_exc(),
        )
        return None


def find_peak_candidates(starts, log_likelihood, config):
    """Refine all starting points in parallel; return the successful ones."""

    with Pool(
        config.num_parallel,
        initializer=setup_process_map,
        initargs=[vars(config)],
        maxtasksperchild=1,
    ) as pool:
        candidates = pool.starmap(
            refine,
            [(start, log_likelihood) for start in starts],
            chunksize=1,
        )
    return [candidate for candidate in candidates if candidate is not None]


def latent_to_unit_cube(sample, log_likelihood):
    """
    Convert an emcee latent sample to dynesty unit cube coordinates.

    Both formalisms use the same parameter ranges, so u = Phi(x), except
    for w: emcee covers ``log_likelihood.get_range("w")``, while dynesty
    covers a single period, w = W_MIN + W_PERIOD * u.
    """

    unit_cube = norm.cdf(sample)
    w_index = SampleParams._fields.index("w")
    w_low, w_high = log_likelihood.get_range("w")
    w = w_low + (w_high - w_low) * unit_cube[w_index]
    unit_cube[w_index] = ((w - W_MIN) % W_PERIOD) / W_PERIOD
    return unit_cube


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument(
        "tic_id", type=int, help="The TIC identifier to find peaks for."
    )
    add_scenario_grid_arguments(parser)
    parser.add_argument(
        "--previous-samples-pattern",
        default=samples_fname_pattern,
        help="Samples file of a previous emcee fit to take a starting point "
        "from (the highest log_prob sample). Use ``{tic_id:d}`` for the TIC "
        "ID.",
    )
    parser.add_argument(
        "--previous-release-pattern",
        default=release_fname_pattern,
        help="Release file of a previous fit, used only if the samples file "
        "does not exist. Use ``{tic_id:d}`` for the TIC ID.",
    )
    parser.add_argument(
        "--no-previous-fit",
        action="store_true",
        help="If passed, do not use a previous fit as a starting point.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="The number of parallel processes to use.",
    )
    parser.add_argument(
        "--logging-verbosity",
        "--verbosity",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="The lowest importance level of logging messages to issue.",
    )
    return parser.parse_args()


def main(config):
    """Find and report the peak candidates for one TIC."""

    setup_process(task="likelihood_peaks", **vars(config))
    log_likelihood = LogLikelihood(config.tic_id)
    starts = scenario_grid_starts(log_likelihood, config)
    if not config.no_previous_fit:
        previous = previous_fit_start(
            log_likelihood,
            config.previous_samples_pattern.format(tic_id=config.tic_id),
            config.previous_release_pattern.format(tic_id=config.tic_id),
        )
        if previous is not None:
            starts.append(previous)
    _logger.info("Refining %d starting points", len(starts))

    candidates = find_peak_candidates(starts, log_likelihood, config)
    _logger.info(
        "%d of %d starting points refined", len(candidates), len(starts)
    )
    for candidate in sorted(
        candidates, key=lambda candidate: candidate.log_prob, reverse=True
    ):
        _logger.info(
            "%s: log_prob=%s\n\tunit cube: %s",
            candidate.source,
            repr(candidate.log_prob),
            repr(latent_to_unit_cube(candidate.lstsq_sample, log_likelihood)),
        )


if __name__ == "__main__":
    main(parse_command_line())
