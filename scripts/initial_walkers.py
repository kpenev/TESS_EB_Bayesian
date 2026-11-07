#!/usr/bin/env python3

"""
Draw initial emcee walkers around the best points of one or more families.

A family is a separate group of good solutions (e.g. young vs old stars). For
the best point of each family (a least squares fit end point, in latent
coordinates):

1. The Jacobian J of the least squares residuals (`utils.get_lstsq_residuals`)
   gives the Gauss-Newton estimate of the posterior covariance, (J^T J)^-1.
   Its eigenvectors are the principal axes.
2. Along each principal axis, in both directions, the log-probability is
   followed until it drops by ``--scan-drop`` below its value at the best
   point. That distance divided by sqrt(2 * scan_drop) (the number of sigma of
   that drop for a Gaussian) is the width on that side, so non-Gaussian and
   asymmetric shapes are measured instead of assumed.
3. The family's proposal is a split normal along each principal axis with
   those widths.

With several families, ``--num-weight-draws`` draws from each family's
proposal estimate its relative posterior mass (the average of posterior /
proposal density), which sets how many walkers start in it. The walkers are
drawn from the proposals, redrawing any with non-finite log-probability.

The walkers are saved as the starting positions of a new samples file, in
the format of the starting position search (`FindStartingPositions`): each
walker's ``log_prob_results`` row holds its log-probability, followed by the
log-probability and latent coordinates of its family's best point. The run is
marked final: restarts, meant to escape poor starting positions, would only
reset how the walkers are split between the families. ``mcmc_sampling.py``
then samples from them. The number of walkers must match its scenario grid
(``--initial-num-ages`` x ``--initial-num-mehs`` x ``--initial-num-ws`` +
``--num-random-walkers``; 256 by default).
"""

from argparse import ArgumentParser
from multiprocessing import get_context
from os import path, remove, replace
import logging

import git
import h5py
import numpy
from scipy import optimize
from scipy.special import logsumexp

from general_purpose_python_modules.emcee_util import save_initial_position
from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import LogLikelihood
from utils import get_lstsq_residuals

_logger = logging.getLogger(__name__)

# The log-likelihood used by the worker processes (set before forking).
_target = None


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC ID to sample.")
    parser.add_argument(
        "points",
        help="A .npy file with the best latent sample of each family, one "
        "per row.",
    )
    parser.add_argument(
        "samples",
        help="The samples file to create for sampling (must not exist). The "
        "walkers are saved as its starting positions and the run is marked "
        "final, so mcmc_sampling.py samples from them without restarts.",
    )
    parser.add_argument(
        "--num-walkers",
        type=int,
        default=256,
        help="The number of walkers; must match the sampling configuration.",
    )
    parser.add_argument(
        "--scan-drop",
        type=float,
        default=2.0,
        help="How far below the best log-probability to follow each principal "
        "axis (2 is 2 sigma for a Gaussian).",
    )
    parser.add_argument(
        "--num-weight-draws",
        type=int,
        default=1000,
        help="Draws per family used to estimate the relative masses of the "
        "families (only with more than one family).",
    )
    parser.add_argument(
        "--min-family-walkers",
        type=int,
        default=8,
        help="The fewest walkers to start in any family, so each stays "
        "represented and walkers can hop between families.",
    )
    parser.add_argument(
        "--jacobian-step",
        type=float,
        default=1e-5,
        help="The step [latent units] of the central differences for the "
        "Jacobian.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="How many log-likelihood evaluations to run in parallel.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: random)."
    )
    return parser.parse_args()


def _log_prob(sample):
    """Return the log-probability of a latent sample (worker process)."""

    return _target(sample)[0]


def _residuals(sample):
    """Return the least squares residuals of a sample (worker process)."""

    return get_lstsq_residuals(_target, sample)


def get_principal_axes(center, pool, step):
    """
    Return the principal axes and Gaussian widths of the posterior at center.

    Returns:
        (array, array):
            The axes (unit vectors, one per column) and the standard deviation
            along each, from the Gauss-Newton covariance (J^T J)^-1.
    """

    num_params = center.size
    offsets = step * numpy.eye(num_params)
    residuals = pool.map(
        _residuals, list(center + offsets) + list(center - offsets)
    )
    if any(entry is None for entry in residuals):
        raise ValueError(
            f"Residuals are not defined within {step} of {center!r}."
        )
    jacobian = (
        numpy.array(residuals[:num_params])
        - numpy.array(residuals[num_params:])
    ).T / (2.0 * step)
    # J^T J >= identity, because the latent sample is part of the residuals.
    eigenvalues, axes = numpy.linalg.eigh(jacobian.T @ jacobian)
    return axes, 1.0 / numpy.sqrt(eigenvalues)


def scan_axis(arguments):
    """
    Return the width of the posterior along one direction (worker process).

    Follows ``center + t * direction`` (t > 0) until the log-probability is
    ``drop`` below ``center_log_prob``, and returns that t divided by
    sqrt(2 * drop).
    """

    center, center_log_prob, direction, guess, drop = arguments
    target = center_log_prob - drop

    def excess(distance):
        """How far above the target the log-probability is at distance."""

        log_prob = _log_prob(center + distance * direction)
        return max(log_prob, target - 1e10) - target

    lower, upper = 0.0, guess * numpy.sqrt(2.0 * drop)
    for _ in range(30):
        if excess(upper) < 0:
            break
        lower, upper = upper, 2.0 * upper
    else:
        _logger.warning(
            "No drop of %s within %s along %s; using that distance.",
            repr(drop),
            repr(upper),
            repr(direction),
        )
        return upper / numpy.sqrt(2.0 * drop)
    distance = optimize.brentq(excess, lower, upper, rtol=1e-2)
    return distance / numpy.sqrt(2.0 * drop)


class SplitNormalProposal:
    """Split normal along given axes: different widths on the two sides."""

    def __init__(self, center, axes, widths_plus, widths_minus):
        """Set up the proposal (axes are unit vectors, one per column)."""

        self.center = center
        self.axes = axes
        self.widths_plus = widths_plus
        self.widths_minus = widths_minus

    def draw(self, num_draws, rng):
        """Return num_draws samples, one per row."""

        normal = rng.standard_normal((num_draws, self.center.size))
        coordinates = normal * numpy.where(
            normal > 0, self.widths_plus, self.widths_minus
        )
        return self.center + coordinates @ self.axes.T

    def log_density(self, samples):
        """Return the log of the proposal density at each sample (row)."""

        coordinates = (samples - self.center) @ self.axes
        widths = numpy.where(
            coordinates > 0, self.widths_plus, self.widths_minus
        )
        return numpy.sum(
            numpy.log(2.0 / (self.widths_plus + self.widths_minus))
            - 0.5 * numpy.log(2.0 * numpy.pi)
            - 0.5 * (coordinates / widths) ** 2,
            axis=1,
        )


def get_proposal(center, pool, config):
    """Return the split normal proposal of the family best at center."""

    center_log_prob = _log_prob(center)
    axes, gaussian_widths = get_principal_axes(
        center, pool, config.jacobian_step
    )
    widths = numpy.array(
        pool.map(
            scan_axis,
            [
                (center, center_log_prob, sign * axis, width, config.scan_drop)
                for sign in (1.0, -1.0)
                for axis, width in zip(axes.T, gaussian_widths)
            ],
        )
    ).reshape(2, center.size)
    _logger.info(
        "Family at log_prob %s: widths along the principal axes (Gaussian, "
        "+ side, - side):\n%s",
        repr(center_log_prob),
        numpy.array2string(
            numpy.stack([gaussian_widths, widths[0], widths[1]], axis=1),
            precision=3,
        ),
    )
    return SplitNormalProposal(center, axes, widths[0], widths[1])


def get_family_walkers(proposals, pool, rng, config):
    """Return how many walkers to start in each family."""

    if len(proposals) == 1:
        return [config.num_walkers]
    log_masses = []
    for proposal in proposals:
        samples = proposal.draw(config.num_weight_draws, rng)
        log_weights = numpy.array(
            pool.map(_log_prob, samples)
        ) - proposal.log_density(samples)
        log_masses.append(
            logsumexp(log_weights) - numpy.log(config.num_weight_draws)
        )
        weights = numpy.exp(log_weights - log_weights.max())
        _logger.info(
            "Family at %s: log mass %s, effective sample size %.1f of %d.",
            repr(proposal.center[:4]),
            repr(log_masses[-1]),
            weights.sum() ** 2 / numpy.sum(weights**2),
            config.num_weight_draws,
        )
    fractions = numpy.exp(numpy.array(log_masses) - max(log_masses))
    fractions /= fractions.sum()
    counts = numpy.maximum(
        config.min_family_walkers,
        numpy.round(fractions * config.num_walkers).astype(int),
    )
    counts[numpy.argmax(counts)] += config.num_walkers - counts.sum()
    _logger.info(
        "Mass fractions %s -> walkers per family %s.",
        repr(fractions),
        repr(counts),
    )
    return list(counts)


def draw_walkers(proposal, num_walkers, pool, rng, max_tries=20):
    """
    Return walkers drawn from the proposal, all with finite log_prob.

    Returns:
        (array, array):
            The walkers (one per row) and their log-probabilities.
    """

    walkers = proposal.draw(num_walkers, rng)
    log_prob = numpy.array(pool.map(_log_prob, walkers))
    for _ in range(max_tries):
        bad = ~numpy.isfinite(log_prob)
        if not bad.any():
            return walkers, log_prob
        walkers[bad] = proposal.draw(bad.sum(), rng)
        log_prob[bad] = pool.map(_log_prob, walkers[bad])
    raise RuntimeError(
        f"Could not draw {num_walkers} walkers with finite log-probability."
    )


def main(config):
    """Find the proposals, split the walkers and save them."""

    global _target  # pylint: disable=global-statement
    _target = LogLikelihood(config.tic_id)
    if path.exists(config.samples):
        raise FileExistsError(f"{config.samples!r} already exists.")
    centers = numpy.atleast_2d(numpy.load(config.points))
    rng = numpy.random.default_rng(config.seed)
    with get_context("fork").Pool(config.num_parallel) as pool:
        center_log_prob = numpy.array(pool.map(_log_prob, centers))
        proposals = [get_proposal(center, pool, config) for center in centers]
        counts = get_family_walkers(proposals, pool, rng, config)
        drawn = [
            draw_walkers(proposal, count, pool, rng)
            for proposal, count in zip(proposals, counts)
        ]
    save_walkers(config, centers, center_log_prob, drawn)


def save_walkers(config, centers, center_log_prob, drawn):
    """
    Create the samples file with the walkers as its starting positions.

    Args:
        config:    The command line configuration.

        centers(array):    The best point of each family.

        center_log_prob(array):    The log-probability of each center.

        drawn([(array, array)]):    For each family, its walkers and their
            log-probabilities (see `draw_walkers()`).
    """

    num_walkers = sum(len(walkers) for walkers, _ in drawn)
    # Written under a temporary name and renamed when complete, so an
    # interrupted run never leaves a partial samples file behind.
    partial_fname = config.samples + ".partial"
    if path.exists(partial_fname):
        remove(partial_fname)
    HDFBackend(partial_fname).reset(num_walkers, centers.shape[1])
    with h5py.File(partial_fname, "a") as samples_file:
        samples_file.attrs["TICID"] = config.tic_id
        samples_file.attrs["GitHash"] = str(
            git.repo.Repo(
                path.dirname(path.dirname(path.abspath(__file__)))
            ).commit()
        )
        samples_file["mcmc"].attrs["final_run"] = True
    index = 0
    for center, best_log_prob, (walkers, log_prob) in zip(
        centers, center_log_prob, drawn
    ):
        for walker, walker_log_prob in zip(walkers, log_prob):
            save_initial_position(
                walker,
                partial_fname,
                nwalkers=num_walkers,
                index=index,
                log_prob_result=(walker_log_prob, best_log_prob)
                + tuple(center),
            )
            index += 1
    replace(partial_fname, config.samples)
    _logger.info(
        "Saved %d walkers (%s per family) to %s.",
        num_walkers,
        repr([len(walkers) for walkers, _ in drawn]),
        config.samples,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
