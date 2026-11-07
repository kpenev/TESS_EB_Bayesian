#!/usr/bin/env python3

"""
Group least squares fit end points into families of the posterior.

Two end points belong to the same family if the log-probability along the
straight line between them (checked at a few points, see ``--line-points``)
never drops more than ``--barrier`` below the lower of the two. Families are
the connected groups of that relation: if A-B and B-C pass, A, B and C are one
family even if A-C does not, so a curved ridge traced by several fits stays
one family.

The barrier should be well above the drop used to set the spread of the
starting walkers around each family (``initial_walkers.py --scan-drop``):
emcee crosses shallower barriers by itself, and a deep barrier keeps the
families' starting clouds apart.

Pairs are tested in order of increasing distance and skipped once in the same
family, so only a small fraction of all pairs is evaluated.

The output is the best end point of each family (best first), one per row,
for ``initial_walkers.py``.
"""

from argparse import ArgumentParser
from multiprocessing import get_context
import logging

from astropy.table import Table
import numpy

from log_likelihood import LogLikelihood

_logger = logging.getLogger(__name__)

# The log-likelihood used by the worker processes (set before forking).
_target = None


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC ID of the fits.")
    parser.add_argument(
        "results",
        help="The .ecsv file of sed_seeded_fits.py; the end points are read "
        "from the matching _x.npy file.",
    )
    parser.add_argument(
        "output", help="Where to save the best point of each family (.npy)."
    )
    parser.add_argument(
        "--keep-delta",
        type=float,
        default=20.0,
        help="Only end points within this much of the best log-probability "
        "are grouped.",
    )
    parser.add_argument(
        "--barrier",
        type=float,
        default=10.0,
        help="End points stay in separate families only if the "
        "log-probability between them drops more than this below the lower "
        "of the two.",
    )
    parser.add_argument(
        "--line-points",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Where along the line between two end points (as fractions) to "
        "check the log-probability.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="How many log-likelihood evaluations to run in parallel.",
    )
    return parser.parse_args()


def _log_prob(sample):
    """Return the log-probability of a latent sample (worker process)."""

    return _target(sample)[0]


def get_candidates(config, pool):
    """Return the good end points and their log-probabilities, best first."""

    results = Table.read(config.results)
    points = numpy.load(config.results.rsplit(".", 1)[0] + "_x.npy")
    assert len(points) == len(results)
    usable = numpy.isfinite(points).all(axis=1) & numpy.isfinite(
        numpy.asarray(results["fit_log_prob"], dtype=float)
    )
    points = points[usable]
    # Evaluated again so the grouping uses the current log-likelihood.
    log_prob = numpy.array(pool.map(_log_prob, points))
    good = log_prob > log_prob.max() - config.keep_delta
    order = numpy.argsort(-log_prob[good])
    _logger.info(
        "%d of %d end points are within %s of the best log_prob %s.",
        good.sum(),
        len(results),
        repr(config.keep_delta),
        repr(log_prob.max()),
    )
    return points[good][order], log_prob[good][order]


class _Families:
    """Union-find over the candidate indices."""

    def __init__(self, size):
        """Start with every candidate in its own family."""

        self._parent = list(range(size))

    def find(self, index):
        """Return the representative of the family of index."""

        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def merge(self, first, second):
        """Put first and second in one family, rooted at its lowest index."""

        first, second = sorted((self.find(first), self.find(second)))
        self._parent[second] = first


def group(points, log_prob, config, pool):
    """Return the family label of each point (the index of its best point)."""

    families = _Families(len(points))
    distances = numpy.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=-1
    )
    pairs = sorted(
        (
            (first, second)
            for first in range(len(points))
            for second in range(first + 1, len(points))
        ),
        key=lambda pair: distances[pair],
    )
    fractions = numpy.array(config.line_points)
    batch_size = max(1, config.num_parallel // len(fractions))
    num_tested = 0
    while pairs:
        batch = []
        while pairs and len(batch) < batch_size:
            pair = pairs.pop(0)
            if families.find(pair[0]) != families.find(pair[1]):
                batch.append(pair)
        if not batch:
            continue
        line_log_prob = numpy.array(
            pool.map(
                _log_prob,
                [
                    points[first] + fraction * (points[second] - points[first])
                    for first, second in batch
                    for fraction in fractions
                ],
            )
        ).reshape(len(batch), len(fractions))
        num_tested += len(batch)
        for (first, second), on_line in zip(batch, line_log_prob):
            lowest_end = min(log_prob[first], log_prob[second])
            if on_line.min() >= lowest_end - config.barrier:
                families.merge(first, second)
    _logger.info(
        "Tested %d of %d pairs.",
        num_tested,
        len(points) * (len(points) - 1) // 2,
    )
    # Points are sorted best first and each family is rooted at its lowest
    # index, so the root is the family's best point.
    return numpy.array([families.find(index) for index in range(len(points))])


def main(config):
    """Group the end points and save the best of each family."""

    global _target  # pylint: disable=global-statement
    _target = LogLikelihood(config.tic_id)
    with get_context("fork").Pool(config.num_parallel) as pool:
        points, log_prob = get_candidates(config, pool)
        labels = group(points, log_prob, config, pool)
    best = numpy.unique(labels)
    numpy.save(config.output, points[best])
    for index in best:
        params = _target.get_sample_params(points[index])
        print(
            f"Family of {numpy.sum(labels == index):3d} end points: best "
            f"log_prob {log_prob[index]:.2f}, Mtot {params.mtotal:.3f} q "
            f"{params.mratio:.3f} age {params.age_gyr:.4g} Gyr [M/H] "
            f"{params.meh:.2f} e {params.ecc:.4f} w {params.w:.1f} b "
            f"{params.primary_impact_param:.3f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
