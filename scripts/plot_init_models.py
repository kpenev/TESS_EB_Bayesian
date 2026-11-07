#!/usr/bin/env python3

"""
Plot the starting models for sampling, one plot per family (mode).

For each family (a row of the points file of select_families.py) the plot
shows the family's best least squares fit highlighted (the red model of the
BUI "best" plot) and the initial walkers drawn around it by initial_walkers.py
as the faint models, over the light curve (full, folded, residuals, zoomed
eclipses) and the SED. The family of each walker is read from the
``*_family.npy`` file that initial_walkers.py writes next to the walkers.
"""

from argparse import ArgumentParser, Namespace
from multiprocessing import get_context
import logging

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import h5py
import numpy
from emcee import State

from exclude_data import exclude_data
from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import LogLikelihood
from visualize import main as visualize

# pylint: enable=wrong-import-position

_logger = logging.getLogger(__name__)

# The log-likelihood used by the worker processes (set before forking).
_target = None


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC ID.")
    parser.add_argument(
        "points",
        help="The best point of each family (.npy, one latent sample per row, "
        "from select_families.py).",
    )
    parser.add_argument(
        "walkers",
        help="The initial walkers (.npy, from initial_walkers.py); their "
        "families are read from the matching _family.npy file.",
    )
    parser.add_argument(
        "output_prefix",
        help="Family k is plotted to OUTPUT_PREFIX_k.pdf (its temporary "
        "samples file is OUTPUT_PREFIX_k_samples.h5).",
    )
    parser.add_argument(
        "--max-walker-models",
        type=int,
        default=50,
        help="Show at most this many of each family's walkers (a random "
        "subset if there are more).",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="How many log-likelihood evaluations to run in parallel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for choosing the walkers shown.",
    )
    return parser.parse_args()


def _log_prob(sample):
    """Return the log-probability of a latent sample (worker process)."""

    return _target(sample)[0]


def write_samples_file(fname, tic_id, samples, log_prob):
    """Write the samples as a one-step samples file (one walker each)."""

    physical = numpy.array(
        [_target.get_sample_params(sample) for sample in samples]
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

    global _target  # pylint: disable=global-statement
    _target = LogLikelihood(config.tic_id)
    points = numpy.atleast_2d(numpy.load(config.points))
    walkers = numpy.load(config.walkers)
    families = numpy.load(config.walkers.rsplit(".", 1)[0] + "_family.npy")
    assert len(families) == len(walkers)
    rng = numpy.random.default_rng(config.seed)
    for family, best in enumerate(points):
        members = walkers[families == family]
        if len(members) > config.max_walker_models:
            members = members[
                rng.choice(
                    len(members), config.max_walker_models, replace=False
                )
            ]
        samples = numpy.concatenate([best[None, :], members])
        with get_context("fork").Pool(config.num_parallel) as pool:
            log_prob = numpy.array(pool.map(_log_prob, samples))
        prefix = f"{config.output_prefix}_{family:d}"
        write_samples_file(
            prefix + "_samples.h5", config.tic_id, samples, log_prob
        )
        plot_models(
            config.tic_id, prefix + "_samples.h5", len(samples), prefix + ".pdf"
        )
        _logger.info(
            "Family %d: best fit log_prob %s, %d of its %d walkers shown "
            "(log_prob %s to %s), plotted to %s.pdf",
            family,
            repr(log_prob[0]),
            len(members),
            numpy.sum(families == family),
            repr(log_prob[1:].min()) if len(members) else "-",
            repr(log_prob[1:].max()) if len(members) else "-",
            prefix,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
