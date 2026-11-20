"""
Distances between MCMC walkers, relative to how far each walker moves.

Used to find walkers that never exchange (postprocess.py, step 1b of
TEC-142) and to test whether a local chain leaks into other groups
(local_chain.py).
"""

import numpy
from scipy.spatial.distance import cdist


def whiten(points, eigenvalue_floor=1e-6):
    """
    Return points in coordinates whitened by their own covariance.

    Each parameter is first standardized (so the parameters' very different
    scales do not matter), then the correlation matrix is whitened. Its
    eigenvalues are floored at ``eigenvalue_floor``, so constant or exactly
    degenerate parameters do not break the transformation; thin directions
    above the floor count fully, as they should (they are thin because the
    data constrain them).

    Args:
        points(array):    The positions, one per row.

    Returns:
        callable:    Transforms positions (one per row) to the whitened
            coordinates.
    """

    mean = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale == 0] = 1.0
    eigenvalues, eigenvectors = numpy.linalg.eigh(
        numpy.atleast_2d(numpy.cov(((points - mean) / scale).T))
    )
    transform = eigenvectors / numpy.sqrt(
        numpy.maximum(eigenvalues, eigenvalue_floor)
    )
    return lambda values: ((values - mean) / scale) @ transform


def get_walker_distances(walker_positions, rng, max_points):
    """
    Return the closest approach between every pair of walkers, per spread.

    Distances are in coordinates whitened by the covariance of all the given
    positions (see `whiten()`). Each pair's closest approach is divided by
    the larger of the two walkers' spreads (median distance of a walker's
    positions from their mean), so it says whether the walkers came within
    each other's range of motion, whatever the other walkers do: walkers
    sampling one region come within a fraction of their spread of each
    other, while walkers each stuck in its own hole stay many spreads apart.

    Args:
        walker_positions([array]):    The positions of each walker, one per
            row.

        rng(numpy.random.Generator):    Chooses the positions used.

        max_points(int):    Each walker is represented by at most this many
            of its distinct positions, chosen at random.

    Returns:
        array:    The closest approach of each pair of walkers divided by the
            larger of their spreads (infinite on the diagonal), shape
            (walkers, walkers).
    """

    transform = whiten(numpy.concatenate(walker_positions))
    subsets = []
    for points in walker_positions:
        points = numpy.unique(points, axis=0)
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        subsets.append(transform(points))
    spreads = numpy.array(
        [
            numpy.median(
                numpy.linalg.norm(points - points.mean(axis=0), axis=1)
            )
            for points in subsets
        ]
    )
    starts = numpy.cumsum([0] + [len(points) for points in subsets[:-1]])
    pooled = numpy.concatenate(subsets)
    result = numpy.empty((len(subsets), len(subsets)))
    for walker, points in enumerate(subsets):
        result[walker] = numpy.minimum.reduceat(
            cdist(points, pooled).min(axis=0), starts
        )
    result = numpy.minimum(result, result.T)
    numpy.fill_diagonal(result, numpy.inf)
    with numpy.errstate(divide="ignore"):
        return result / numpy.maximum.outer(spreads, spreads)
