"""
The masses of a target's regions, by bridge sampling (step 4 of TEC-142).

Each region's log evidence is estimated by cross-fitted bridge sampling
(`bridge_sampling.cross_fit_log_evidence()`), with draws from the reference
densities assigned to regions by the target's `regions.RegionRule`: a draw
in the region is evaluated with the posterior the main chain sampled, every
other draw gets -inf.
"""

from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

import numpy
from scipy.special import logsumexp

from bridge_sampling import cross_fit_log_evidence
from local_chain import check_likelihood

_logger = logging.getLogger(__name__)

RegionMass = namedtuple(
    "RegionMass",
    [
        "estimate",
        "weight",
        "log_weight_error_lower",
        "log_weight_error_upper",
        "chain_fraction",
    ],
)


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


def get_chain_region(main_chain, labels, region):
    """
    Return a large region's main-chain samples, as bridge sampling takes them.

    Args:
        main_chain(MainChain):    The main chain.

        labels(array):    The group of each of its positions after merges,
            shape (steps, walkers).

        region(int):    The region (group label).

    Returns:
        dict:    One row per sample of "positions", "log_prob" (as the
            chain stored it) and "walkers".
    """

    in_region = labels == region
    return {
        "positions": main_chain.positions[in_region],
        "log_prob": main_chain.log_prob[in_region],
        "walkers": numpy.nonzero(in_region)[1],
    }


def get_weights(log_evidences, errors):
    """
    Return each region's weight and the error of its log.

    The weight of region r is w_r = Z_r / sum_j Z_j. With independent
    errors e_j of the log evidences, the error of log w_r is
    sqrt((1 - w_r)^2 e_r^2 + sum_{j != r} w_j^2 e_j^2), which equals
    sqrt(e_r^2 (1 - 2 w_r) + sum_j w_j^2 e_j^2).

    Args:
        log_evidences(array):    The log evidence of each region.

        errors(array):    The standard deviation of each log evidence.

    Returns:
        (array, array):    The weights and the errors of their logs.
    """

    weights = numpy.exp(log_evidences - logsumexp(log_evidences))
    return weights, numpy.sqrt(
        errors**2 * (1.0 - 2.0 * weights) + numpy.sum(weights**2 * errors**2)
    )


def cross_fit_regions(main_chain, rule, regions, config):
    """
    Return the cross-fitted bridge estimate of each region's log evidence.

    One pool of workers evaluates the draws of all regions. Before any
    region is bridged, the likelihood must reproduce the log-probabilities
    stored with some of its samples.

    Args:
        main_chain(MainChain):    The main chain and the function it
            sampled.

        rule(RegionRule):    Assigns the draws to regions.

        regions(dict):    The samples of each region, as bridge sampling
            takes them (see `get_chain_region()`).

        config:    The command line configuration (``seed``,
            ``num_parallel``, ``worker_max_tasks`` and the ``bridge_*``
            options).

    Returns:
        dict:    The `bridge_sampling.CrossFitEstimate` of each region.
    """

    rng = numpy.random.default_rng(config.seed)
    with ProcessPoolExecutor(
        max_workers=config.num_parallel,
        max_tasks_per_child=config.worker_max_tasks,
    ) as pool:
        evaluate = partial(
            evaluate_log_probability,
            log_probability=main_chain.log_probability,
            pool=pool,
            num_workers=config.num_parallel,
        )
        estimates = {}
        for region, samples in regions.items():
            check_likelihood(
                main_chain.log_probability,
                samples["positions"],
                samples["log_prob"],
                pool,
            )
            estimates[region] = cross_fit_log_evidence(
                samples,
                partial(
                    evaluate_in_region,
                    region=region,
                    rule=rule,
                    evaluate=evaluate,
                ),
                rng,
                config,
            )
    return estimates


def combine_estimates(estimates, chain_fractions):
    """
    Return each region's weight, from the log evidences of all of them.

    Args:
        estimates(dict):    The `bridge_sampling.CrossFitEstimate` of each
            region.

        chain_fractions(dict):    The fraction of the main chain's positions
            labelled with each region.

    Returns:
        dict:    The `RegionMass` of each region.
    """

    log_evidences = numpy.array(
        [estimate.log_evidence for estimate in estimates.values()]
    )
    weights, lower = get_weights(
        log_evidences,
        numpy.array([estimate.error_lower for estimate in estimates.values()]),
    )
    upper = get_weights(
        log_evidences,
        numpy.array([estimate.error_upper for estimate in estimates.values()]),
    )[1]
    masses = {
        region: RegionMass(
            estimate,
            weights[index],
            lower[index],
            upper[index],
            chain_fractions[region],
        )
        for index, (region, estimate) in enumerate(estimates.items())
    }
    for region, mass in masses.items():
        _logger.info(
            "Region %d: log Z %.4f (error %.4f to %.4f, halves %.1f sigma "
            "apart), weight %.4g (error of its log %.4f to %.4f); fraction "
            "of main-chain positions %.4g.",
            region,
            mass.estimate.log_evidence,
            mass.estimate.error_lower,
            mass.estimate.error_upper,
            mass.estimate.agreement,
            mass.weight,
            mass.log_weight_error_lower,
            mass.log_weight_error_upper,
            mass.chain_fraction,
        )
    return masses


def estimate_region_masses(main_chain, rule, labels, small_samples, config):
    """
    Return the mass of each of a target's regions (step 4 of TEC-142).

    A large region's samples are the main-chain positions with its label; a
    confirmed small region's are its local chain's kept samples. The stuck
    walkers of small regions and the discarded groups are samples of no
    region. A target with a single region needs no bridge: its weight is 1.

    Args:
        main_chain(MainChain):    The main chain and the function it
            sampled.

        rule(RegionRule):    Assigns points to regions.

        labels(array):    The group of each main-chain position after
            merges, shape (steps, walkers).

        small_samples(dict):    The samples of each confirmed small region
            (see `local_chain.resolve_small_groups()`).

        config:    The command line configuration (see
            `cross_fit_regions()`).

    Returns:
        dict:    The `RegionMass` of each region: the cross-fitted estimate
            of its log evidence (None for a single region), its weight, the
            lower and upper bounds of the error of the weight's log, and the
            fraction of main-chain positions with its label (its weight if
            the main chain mixed).
    """

    regions = {
        region: get_chain_region(main_chain, labels, region)
        for region in rule.large_regions
    }
    regions.update(small_samples)
    chain_fractions = {
        region: float((labels == region).mean()) for region in regions
    }
    if len(regions) == 1:
        (region,) = regions
        _logger.info("Region %d is the only one: weight 1.", region)
        return {
            region: RegionMass(None, 1.0, 0.0, 0.0, chain_fractions[region])
        }
    return combine_estimates(
        cross_fit_regions(main_chain, rule, regions, config), chain_fractions
    )
