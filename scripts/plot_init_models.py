#!/usr/bin/env python3

"""
Plot the starting models for sampling, one plot per family (mode).

Reads the starting positions that initial_walkers.py saved in a samples file.
Each walker's ``log_prob_results`` row holds its log-probability and the
log-probability and latent coordinates of its family's best point, so walkers
sharing a best point form a family. For each family the plot shows the best
point highlighted (the red model of the BUI "best" plot) and the family's
walkers as the faint models, over the light curve (full, folded, residuals,
zoomed eclipses) and the SED.
"""

from argparse import ArgumentParser, Namespace
import logging

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import h5py
import numpy
from emcee import State

from general_purpose_python_modules.emcee_util import load_initial_positions
from exclude_data import exclude_data
from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import LogLikelihood
from sample_params import SampleParams
from visualize import main as visualize

# pylint: enable=wrong-import-position

_logger = logging.getLogger(__name__)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC ID.")
    parser.add_argument(
        "samples",
        help="The samples file with the starting positions saved by "
        "initial_walkers.py.",
    )
    parser.add_argument(
        "output_prefix",
        help="Family k (best first) is plotted to OUTPUT_PREFIX_k.pdf (its "
        "temporary samples file is OUTPUT_PREFIX_k_samples.h5).",
    )
    parser.add_argument(
        "--max-walker-models",
        type=int,
        default=50,
        help="Show at most this many of each family's walkers (a random "
        "subset if there are more).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for choosing the walkers shown.",
    )
    return parser.parse_args()


def load_families(samples_fname):
    """
    Return the families of the starting positions in a samples file.

    Every walker is saved with an exact copy of its family's best point, so
    walkers with identical best points form a family.

    Returns:
        [(array, float, array, array)]:
            For each family, best first: its best point, the log-probability
            of that point, its walkers (one per row) and their
            log-probabilities.
    """

    with h5py.File(samples_fname, "r") as samples_file:
        num_walkers = len(samples_file["mcmc/starting_positions/positions"])
    num_params = len(SampleParams._fields)
    positions, log_prob, best, found = load_initial_positions(
        samples_fname,
        num_params=num_params,
        num_walkers=num_walkers,
        blobs_dtype=[("log_likelihood", float)]
        + [(f"s{index:02d}", float) for index in range(num_params)],
    )
    assert found.all(), f"Not all starting positions saved in {samples_fname}"
    best_log_prob = best["log_likelihood"]
    best = numpy.array([list(row)[1:] for row in best])
    families = []
    for center in numpy.unique(best, axis=0):
        members = (best == center).all(axis=1)
        families.append(
            (
                center,
                best_log_prob[members][0],
                positions[members],
                log_prob[members],
            )
        )
    return sorted(families, key=lambda family: -family[1])


def write_samples_file(fname, tic_id, samples, log_prob, log_likelihood):
    """Write the samples as a one-step samples file (one walker each)."""

    physical = numpy.array(
        [log_likelihood.get_sample_params(sample) for sample in samples]
    )
    backend = HDFBackend(fname, name="mcmc")
    backend.reset(len(samples), samples.shape[1])
    backend.grow(1, physical)
    backend.save_step(
        State(
            samples,
            log_prob=log_prob,
            blobs=physical,
            random_state=numpy.random.get_state(),
        ),
        numpy.ones(len(samples), dtype=bool),
    )
    with h5py.File(fname, "a") as samples_file:
        samples_file.attrs["TICID"] = tic_id


def plot_models(tic_id, samples_fname, num_models, output):
    """
    Plot all walkers of a one-step samples file, the first one highlighted.

    Uses the configuration of bui/select_ticids/plots.py:best(), except the
    models: walker 0 is listed first, so it is drawn as the highlighted model
    whatever its log-probability, and the rest are drawn faint.
    """

    visualize(
        Namespace(
            tic_id=tic_id,
            remove_lc_trend="moving_median",
            plot_lightcurve=[
                output,
                "[[full,         full,        zoom_primary],"
                " [folded,       folded,      zoom_secondary],"
                " [folded_diff,  folded_diff, sed]]",
            ],
            show_model_with_lc=[f"0,{walker}" for walker in range(num_models)],
            data_on_top=False,
            samples_fname=samples_fname,
            chain_name="mcmc",
            burn_in=0,
            thin=1,
            sample_condition=None,
            eclipse_model_only="OOE" in exclude_data.get(tic_id, []),
        )
    )


def main(config):
    """Make the plot of each family."""

    log_likelihood = LogLikelihood(config.tic_id)
    rng = numpy.random.default_rng(config.seed)
    for family, (best, best_log_prob, walkers, log_prob) in enumerate(
        load_families(config.samples)
    ):
        shown = numpy.arange(len(walkers))
        if len(walkers) > config.max_walker_models:
            shown = rng.choice(
                len(walkers), config.max_walker_models, replace=False
            )
        prefix = f"{config.output_prefix}_{family:d}"
        write_samples_file(
            prefix + "_samples.h5",
            config.tic_id,
            numpy.concatenate([best[None, :], walkers[shown]]),
            numpy.concatenate([[best_log_prob], log_prob[shown]]),
            log_likelihood,
        )
        plot_models(
            config.tic_id,
            prefix + "_samples.h5",
            len(shown) + 1,
            prefix + ".pdf",
        )
        _logger.info(
            "Family %d: best point log_prob %s, %d of its %d walkers shown "
            "(log_prob %s to %s), plotted to %s.pdf",
            family,
            repr(best_log_prob),
            len(shown),
            len(walkers),
            repr(log_prob.min()),
            repr(log_prob.max()),
            prefix,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
