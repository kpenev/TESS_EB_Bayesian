"""
The masses of a target's regions, by bridge sampling (step 4 of TEC-142).

Each region's log evidence is estimated by cross-fitted bridge sampling
(`bridge_sampling.cross_fit_log_evidence()`), with draws from the reference
densities assigned to regions by the target's `regions.RegionRule`: a draw
in the region is evaluated with the posterior the main chain sampled, every
other draw gets -inf.
"""

import logging

import numpy

_logger = logging.getLogger(__name__)


def evaluate_log_probability(points, log_probability, pool, num_workers):
    """
    Return the posterior's log-probability at points, evaluated in a pool.

    Args:
        points(array):    The points, one per row.

        log_probability:    The function the main chain sampled (see
            `MainChain.log_probability`): called with one position, returns
            the log-probability followed by the blobs.

        pool:    Evaluates ``log_probability`` in parallel (``map()``).

        num_workers(int):    The pool's number of workers: the points are
            sent in about four chunks per worker.

    Returns:
        array:    The log-probability at each point.
    """

    return numpy.array(
        [
            result[0]
            for result in pool.map(
                log_probability,
                points,
                chunksize=max(1, points.shape[0] // (4 * num_workers)),
            )
        ]
    )


def evaluate_in_region(draws, region, rule, evaluate):
    """
    Return the posterior's log-probability at draws, -inf outside a region.

    Only the draws the region rule puts in the region are evaluated, so a
    reference density wider than its region costs no extra evaluations.

    Args:
        draws(array):    The draws, one per row.

        region(int):    The region (group label).

        rule(RegionRule):    Assigns points to regions.

        evaluate:    Called with points (one per row), returns the
            posterior's log-probability at each, as the main chain stored it
            (e.g. `evaluate_log_probability()` with its other arguments
            bound).

    Returns:
        array:    The log-probability at each draw, -inf where the rule
            assigns it to another region.
    """

    inside = rule.classify(draws) == region
    log_prob = numpy.full(draws.shape[0], -numpy.inf)
    if inside.any():
        log_prob[inside] = evaluate(draws[inside])
    _logger.info(
        "Region %d: %d of %d draws inside.",
        region,
        inside.sum(),
        draws.shape[0],
    )
    return log_prob
