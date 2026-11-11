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
family.

The model has exact symmetries, so each solution appears in several copies:
swapping the stars (mass ratio q -> 1/q), mirroring the orbit (impact
parameter b -> -b) and shifting the eclipse time by whole orbital periods.
The prior keeps all of them, so walkers can cross q = 1, b = 0 and any
eclipse time, but the end points are compared in one copy: q <= 1 and b >= 0,
and each pair through the eclipse time copies that take the short way round.
Every transformed point is checked to keep its likelihood; if one does not,
the target fails. (w has no copies: e and w are sampled as a point in a plane,
see `LogLikelihood.orbit_from_latent()`.)

The output is the best end point of each family (best first), with the
eclipse time within half a period of the middle of its
prior, one per row, for ``initial_walkers.py``.
"""

from argparse import ArgumentParser
from multiprocessing import get_context
import logging

from astropy.table import Table
import numpy
from scipy.stats import norm

from binary import Binary
from log_likelihood import LogLikelihood
from sample_params import SampleParams

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


def _exact_log_prob(sample):
    """Same as `_log_prob()`, but with the light curve computed exactly."""

    return _target(sample, exact=True)[0]


def get_physical(sample, param):
    """Return the physical value of one parameter of a latent sample."""

    return _target.prior_transform(
        param, sample[SampleParams._fields.index(param)]
    )


def set_physical(sample, param, value):
    """Return a copy of a latent sample with one physical parameter set."""

    index, latent = _target.inverse_prior(param, value)
    result = numpy.array(sample)
    result[index] = latent
    return result


def swap_stars(sample):
    """
    Return the sample describing the same binary with the stars swapped.

    The per-star parameters (limb darkening, rotation, reflection and
    beaming) have the same priors for both stars, so their latent values are
    exchanged directly. w + 180 is the negated latents of e and w
    (`LogLikelihood.orbit_from_latent()`). Of the rest, only the mass ratio
    (inverted), the eclipse time (that of the other star's eclipse) and the
    impact parameter change; they are read off the binary after
    `BinaryParams.swap_components()` (the other eclipse, and the semimajor
    axis in the other star's radii, so the impact parameter is a cos i),
    except the mass ratio, which is inverted from the sample since the
    binary's is that of the current, not initial, masses. Everything else
    keeps its latent value, so parameters at a prior edge (where converting
    the physical value back would give an infinite latent value) survive.
    """

    params = _target.get_sample_params(sample)
    binary = Binary(from_mcmc=params)
    binary.swap_components()
    result = numpy.array(sample)
    for name in (
        "limb_dark_1",
        "limb_dark_2",
        "prot",
        "reflection_coef",
        "beaming_coef",
    ):
        primary = SampleParams._fields.index(f"primary_{name}")
        secondary = SampleParams._fields.index(f"secondary_{name}")
        result[[primary, secondary]] = result[[secondary, primary]]
    for param in ("ecc", "w"):
        result[SampleParams._fields.index(param)] *= -1.0
    for param, value in (
        ("mratio", 1.0 / params.mratio),
        ("eclipse_time", binary.t0),
        (
            "primary_impact_param",
            binary.a * numpy.cos(numpy.radians(binary.inc)),
        ),
    ):
        result = set_physical(result, param, value)
    return result


def canonical(sample):
    """
    Return the copy of a sample with mass ratio <= 1 and impact parameter >= 0.

    Swapping the stars (`swap_stars()`) and b -> -b (mirroring the orbit across
    the sky plane) give the same model. The prior keeps all the copies, so
    walkers can cross q = 1 and b = 0, but the end points of the fits are
    compared in one copy.
    """

    if get_physical(sample, "mratio") > 1.0:
        sample = swap_stars(sample)
    impact = get_physical(sample, "primary_impact_param")
    if impact >= 0:
        return sample
    return set_physical(sample, "primary_impact_param", -impact)


def shift_periodic(sample, param, period, target):
    """
    Return the copy of a sample with ``param`` nearest ``target``.

    The parameter is shifted by a whole number of periods, staying strictly
    inside its prior range (which spans several periods, so one of the
    candidates always does).
    """

    value = get_physical(sample, param)
    low, high = _target.get_range(param)
    turns = numpy.round((target - value) / period)
    nearest = min(
        (
            value + period * shift
            for shift in (turns, turns - 1.0, turns + 1.0)
            if low < value + period * shift < high
        ),
        key=lambda candidate: abs(candidate - target),
    )
    if nearest == value:
        return sample
    return set_physical(sample, param, nearest)


def nearest_copy(sample, reference):
    """
    Return the copy of a sample with the eclipse time nearest a reference's.

    Eclipse times a whole number of orbital periods apart give the same
    model, and the prior spans several periods so walkers can cross any
    value. Comparing two end points through their nearest copies takes the
    short way round. (w needs no copies: its latents are a point in a plane,
    see `LogLikelihood.orbit_from_latent()`.)
    """

    return shift_periodic(
        sample,
        "eclipse_time",
        get_physical(sample, "per"),
        get_physical(reference, "eclipse_time"),
    )


def check_same_solution(originals, copies, pool, tolerance=1e-6):
    """
    Raise AssertionError unless the copies describe the same binaries.

    Compares the log-probability without the latent unit normal prior density
    (i.e. likelihood plus physical prior): the latent density itself differs
    between copies, since the prior transform stretches differently at each.
    The light curve model is computed exactly (not interpolated): each star's
    phase grid is anchored to its own eclipse time, so after a star swap the
    interpolation error differs slightly (~1e-5) between the copies.
    """

    if len(originals) == 0:
        return

    def physical_log_prob(samples):
        return numpy.array(pool.map(_exact_log_prob, samples)) - norm.logpdf(
            samples
        ).sum(axis=1)

    difference = numpy.abs(
        physical_log_prob(copies) - physical_log_prob(originals)
    )
    _logger.info(
        "%d copies checked; largest log-probability change %s.",
        len(copies),
        repr(difference.max()),
    )
    assert (
        difference <= tolerance
    ).all(), f"Copies changed the log-probability by up to {difference.max()!r}"


def fold_periodic(sample):
    """
    Return the copy of a sample with the eclipse time mid-prior.

    The eclipse time is brought within half an orbital period of the middle
    of its prior range, as far as possible from its edges.
    """

    return shift_periodic(
        sample,
        "eclipse_time",
        get_physical(sample, "per"),
        numpy.mean(_target.get_range("eclipse_time")),
    )


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
    # Compared in one copy of the symmetric solutions (see canonical()); the
    # log-probability is evaluated again, since the latent prior density
    # differs between copies.
    copies = numpy.array([canonical(point) for point in points[good]])
    changed = (copies != points[good]).any(axis=1)
    check_same_solution(points[good][changed], copies[changed], pool)
    points[good] = copies
    log_prob[good] = pool.map(_log_prob, copies)
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
    # Each pair is compared through the copy of its second point with w and
    # eclipse time nearest those of the first, so the line between them takes
    # the short way round.
    nearest = {
        (first, second): nearest_copy(points[second], points[first])
        for first in range(len(points))
        for second in range(first + 1, len(points))
    }
    pairs = sorted(
        nearest,
        key=lambda pair: numpy.linalg.norm(nearest[pair] - points[pair[0]]),
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
                    points[first]
                    + fraction * (nearest[first, second] - points[first])
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
        # The walkers start around these, so w and the eclipse time are kept
        # far from the edges of their prior ranges.
        seeds = numpy.array([fold_periodic(points[index]) for index in best])
        changed = (seeds != points[best]).any(axis=1)
        check_same_solution(points[best][changed], seeds[changed], pool)
    numpy.save(config.output, seeds)
    for index, seed in zip(best, seeds):
        params = _target.get_sample_params(seed)
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
