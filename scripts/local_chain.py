"""
Local chains for small groups of walkers (step 3 of TEC-142).

A small group whose walkers do not barely move (see postprocess.py) gets a
chain of its own, with stretch moves only, started from its walkers'
positions. If a local-chain position is typical of another large group
(`has_leaked()`), the group was not isolated: it leaked and is merged back,
and the main chain stands.
Otherwise, once converged, the local chain's samples describe the posterior
in the group's region, for bridge sampling. Local chains are saved and
resumed: reaching the step limit only pauses one (the group stays
unresolved), since slow convergence says nothing about the region's mass.
"""

from concurrent.futures import ProcessPoolExecutor
from os import makedirs, path
import logging

from emcee import EnsembleSampler, walkers_independent
from emcee.moves import StretchMove
import git.repo
import h5py
import numpy

from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import LogLikelihood
from mcmc_sampling import get_isochrone_interpolation
from walker_distances import whiten

_logger = logging.getLogger(__name__)
_git_hash = str(
    git.repo.Repo(path.dirname(path.dirname(path.abspath(__file__)))).commit()
)

# The group of each local chain file holding the chain (and its verdict).
_chain_name = "local"


def check_likelihood(log_likelihood, positions, log_prob, pool, num_checks=5):
    """
    Raise if the likelihood does not reproduce stored log-probabilities.

    Args:
        log_likelihood(LogLikelihood):    The likelihood for the local chain.

        positions(array):    Positions from the main chain, one per row.

        log_prob(array):    The log-probabilities the main chain stored for
            them.

        pool:    Evaluates the likelihood in parallel.
    """

    indices = numpy.linspace(
        0, len(positions) - 1, min(num_checks, len(positions))
    ).astype(int)
    computed = numpy.array(
        [entry[0] for entry in pool.map(log_likelihood, positions[indices])]
    )
    if not numpy.allclose(computed, log_prob[indices], rtol=1e-8, atol=1e-8):
        raise RuntimeError(
            f"The likelihood gives {computed!r} where the main chain stored "
            f"{log_prob[indices]!r}: a different likelihood (code, cache "
            "rows or isochrone interpolation)."
        )


def get_starting_walkers(positions, num_walkers, rng, jitter=0.1):
    """
    Return starting walkers drawn from a group's distinct positions.

    If the group has fewer distinct positions than walkers, positions are
    reused with Gaussian jitter of ``jitter`` times their spread in each
    parameter.

    Args:
        positions(array):    The group's positions, one per row.

        num_walkers(int):    How many walkers to start.

        rng(numpy.random.Generator):    Chooses the positions and jitter.

    Returns:
        array:    The starting walkers, one per row.
    """

    distinct = numpy.unique(positions, axis=0)
    if len(distinct) >= num_walkers:
        walkers = distinct[
            rng.choice(len(distinct), num_walkers, replace=False)
        ]
    else:
        scale = distinct.std(axis=0)
        if (scale == 0).any():
            raise RuntimeError(
                f"Only {len(distinct)} distinct positions, not varying in "
                "every parameter: cannot start a local chain from them."
            )
        walkers = distinct[
            rng.choice(len(distinct), num_walkers)
        ] + jitter * scale * rng.standard_normal(
            (num_walkers, distinct.shape[1])
        )
    if not walkers_independent(walkers):
        raise RuntimeError(
            "The starting walkers of the local chain are not linearly "
            "independent."
        )
    return walkers


def get_typical_region(group_positions, percentile):
    """
    Return how to tell whether positions are typical of a group.

    A position is typical if its squared distance from the group's mean, in
    coordinates whitened by the group's own covariance (`whiten()`), is
    within the ``percentile`` of the group's own positions. Whitening by
    the group's own covariance keeps the separation from other groups at
    its true size (whitening several groups together compresses it), and a
    high percentile ignores the group's few stray positions. It treats the
    group as one roughly elliptical region, which holds in the latent
    coordinates (e.g. eccentricity and argument of periapsis are straight
    there).

    Args:
        group_positions(array):    The group's positions, one per row.

        percentile(float):    The percentile (0-100) of the group's own
            squared distances that bounds the typical region.

    Returns:
        (callable, float):
            Transforms positions (one per row) to the group's whitened
            coordinates, and the largest typical squared distance there.
    """

    transform = whiten(group_positions)
    return transform, numpy.percentile(
        numpy.sum(transform(group_positions) ** 2, axis=1), percentile
    )


def has_leaked(local_positions, typical_regions):
    """
    Return whether any local-chain position is typical of another group.

    Entering another group's typical region even once means the regions
    connect.

    Args:
        local_positions(array):    Recent positions of the local chain,
            shape (steps, walkers, parameters).

        typical_regions([(callable, float)]):    The typical region of each
            other large group (see `get_typical_region()`).
    """

    points = local_positions.reshape(-1, local_positions.shape[-1])
    return any(
        (numpy.sum(transform(points) ** 2, axis=1) <= threshold).any()
        for transform, threshold in typical_regions
    )


def start_local_chain(group, log_likelihood, rng, config):
    """
    Return the backend of a group's local chain and its starting walkers.

    A new local chain records its provenance (see `run_local_chain()`). An
    existing one is resumed (starting walkers None), unless it already has a
    final verdict, which is returned instead of the starting walkers.

    Args:
        group(dict):    See `run_local_chain()`.

        log_likelihood(LogLikelihood):    The likelihood.

        rng(numpy.random.Generator):    Chooses the starting walkers.

        config:    The command line configuration.

    Returns:
        (HDFBackend, array or None or str):
            The local chain's backend and its starting walkers, None to
            resume, or the existing final verdict.
    """

    makedirs(path.dirname(group["fname"]) or ".", exist_ok=True)
    backend = HDFBackend(group["fname"], name=_chain_name)
    if backend.initialized and backend.iteration > 0:
        with h5py.File(group["fname"], "r") as local_file:
            verdict = local_file[_chain_name].attrs["verdict"]
        _logger.info(
            "Local chain %s has %d steps: %s.",
            repr(group["fname"]),
            backend.iteration,
            verdict,
        )
        return backend, (None if verdict == "unresolved" else verdict)
    walkers = get_starting_walkers(
        group["positions"], config.local_chain_walkers, rng
    )
    backend.reset(*walkers.shape)
    with h5py.File(group["fname"], "a") as local_file:
        local_file.attrs["TICID"] = config.tic_id
        local_file.attrs["GitHash"] = _git_hash
        local_file.attrs["IsochroneInterpolation"] = (
            log_likelihood.isochrone_interpolation
        )
        local_file.attrs["Walkers"] = group["walkers"]
        local_file.attrs["BurnIn"] = config.burn_in
        local_file[_chain_name].attrs["verdict"] = "unresolved"
        local_file[_chain_name].create_dataset(
            "starting_positions", data=walkers
        )
    return backend, walkers


def run_local_chain(group, typical_regions, log_likelihood, config):
    """
    Run or resume one group's local chain; return its verdict.

    Runs blocks of ``config.local_chain_block`` steps until a position is
    typical of another large group (verdict "leaked"), the second half of
    the chain is
    at least ``config.local_chain_autocorr_factor`` integrated
    autocorrelation times long ("converged", the second half are samples of
    the group's region), or ``config.local_chain_max_steps`` steps were added
    in this call ("unresolved": resumed next time).

    Args:
        group(dict):    The group: its walkers ("walkers"), their positions
            in it ("positions", one per row) and the stored log-probabilities
            there ("log_prob"), and the local chain's file name ("fname").

        typical_regions([(callable, float)]):    The typical region of each
            large group (see `get_typical_region()`).

        log_likelihood(LogLikelihood):    The likelihood.

        config:    The command line configuration.

    Returns:
        str:    The verdict.
    """

    rng = numpy.random.default_rng(config.seed)
    with ProcessPoolExecutor(
        max_workers=config.num_parallel,
        max_tasks_per_child=config.worker_max_tasks,
    ) as pool:
        check_likelihood(
            log_likelihood, group["positions"], group["log_prob"], pool
        )
        backend, state = start_local_chain(group, log_likelihood, rng, config)
        if isinstance(state, str):
            return state
        sampler = EnsembleSampler(
            *backend.shape,
            log_likelihood,
            backend=backend,
            pool=pool,
            moves=StretchMove(),
        )
        verdict = "unresolved"
        for _ in range(
            config.local_chain_max_steps // config.local_chain_block
        ):
            sampler.run_mcmc(state, config.local_chain_block)
            state = None
            if has_leaked(
                backend.get_chain(
                    discard=backend.iteration - config.local_chain_block
                ),
                typical_regions,
            ):
                verdict = "leaked"
                break
            autocorr_time = backend.get_autocorr_time(
                discard=backend.iteration // 2, tol=0
            ).max()
            _logger.info(
                "Local chain %s: %d steps, autocorrelation time %.0f.",
                repr(group["fname"]),
                backend.iteration,
                autocorr_time,
            )
            if (
                backend.iteration // 2
                >= config.local_chain_autocorr_factor * autocorr_time
            ):
                verdict = "converged"
                break
    with h5py.File(group["fname"], "a") as local_file:
        local_file[_chain_name].attrs["verdict"] = verdict
    return verdict


def resolve_small_groups(backend, positions, labels, discarded, config):
    """
    Run the local chains of the small groups that are not discarded.

    Args:
        backend(HDFBackend):    The main chain.

        positions(array):    The main chain's positions after the burn-in
            (thinned), shape (steps, walkers, parameters).

        labels(array):    The group of each position, shape (steps,
            walkers).

        discarded(set):    The discarded groups.

        config:    The command line configuration.

    Returns:
        dict:    The verdict of each small group's local chain ("leaked",
            "converged" or "unresolved").
    """

    num_walkers = [
        (labels == group).any(axis=0).sum() for group in range(labels.max() + 1)
    ]
    small_groups = [
        group
        for group, count in enumerate(num_walkers)
        if group not in discarded and count < config.group_min_large_walkers
    ]
    if not small_groups:
        return {}
    log_prob = backend.get_log_prob(discard=config.burn_in, thin=config.thin)
    log_likelihood = LogLikelihood(
        config.tic_id,
        isochrone_interpolation=get_isochrone_interpolation(backend.filename),
    )
    # Small groups have too few positions for a reliable covariance, so
    # leaks are only detected into large groups.
    typical_regions = [
        get_typical_region(positions[labels == group], config.leak_percentile)
        for group, count in enumerate(num_walkers)
        if count >= config.group_min_large_walkers
    ]
    verdicts = {}
    for group in small_groups:
        in_group = labels == group
        walkers = numpy.flatnonzero(in_group.any(axis=0))
        verdicts[group] = run_local_chain(
            {
                "walkers": walkers.tolist(),
                "positions": positions[in_group],
                "log_prob": log_prob[in_group],
                "fname": config.local_chain_fname_pattern.format(
                    tic_id=config.tic_id,
                    walkers="-".join(str(walker) for walker in walkers),
                ),
            },
            typical_regions,
            log_likelihood,
            config,
        )
        _logger.info(
            "Group %d (walkers %s): local chain %s.",
            group,
            walkers.tolist(),
            verdicts[group],
        )
    return verdicts
