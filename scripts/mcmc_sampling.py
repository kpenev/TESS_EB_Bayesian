#!/usr/bin/env python3

"""Use EMCEE to sample a TESS eclipsing binary."""

from os import path, makedirs
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
from itertools import count
from traceback import format_exc

from configargparse import ArgumentParser, DefaultsFormatter
from emcee import EnsembleSampler, walkers_independent
import h5py
import numpy
import git.repo

from general_purpose_python_modules.multiprocessing_util import (
    setup_process,
    setup_process_map,
)
from general_purpose_python_modules.emcee_util import (
    save_initial_position,
    load_initial_positions,
)

from sqlalchemy import select, update
from bui.db_interface import Session
from bui.select_ticids.data_model import JobGroup, get_ticid_select_tables
from utils import lmfit_and_tweak
from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import SampleParams, LogLikelihood
from log_likelihood_priors import LogLikelihoodPriorsOnly
from paths import results_dir, samples as samples_fname_pattern
from find_starting_positions import FindStartingPositions

_logger = logging.getLogger(__name__)
_git_hash = str(
    git.repo.Repo(path.dirname(path.dirname(path.abspath(__file__)))).commit()
)

default_logging_format = (
    "%(levelname)s %(asctime)s %(name)s: %(message)s | "
    "%(pathname)s.%(funcName)s:%(lineno)d"
)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["mcmc_sampling.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "tic_id", type=int, help="The TIC identifier to sample."
    )
    parser.add_argument(
        "--priors-only",
        action="store_true",
        help="If passed, the distribution sampled is just the priors.",
    )
    parser.add_argument(
        "--samples-fname-pattern",
        default=samples_fname_pattern,
        help="The filename where to save samples. If the file already exists, "
        "sampling continues, adding more points to the existing chain.",
    )
    parser.add_argument(
        "--num-random-walkers",
        type=int,
        default=0,
        help="Number of additional walkers to start at random positions in "
        "addition to those for which optimizing the starting point is "
        "attempted. The total number of walkers used is "
        "``--num-random-walkers`` + ``--initial-num-ages`` * "
        "``--initial-num-mehs`` * ``--initial-num-ws``.",
    )
    parser.add_argument(
        "--initial-num-ages",
        type=int,
        default=8,
        help="The number of different initial ages to optimize starting "
        "positions for.",
    )
    parser.add_argument(
        "--initial-logage-smear",
        type=float,
        default=0.02,
        help="The initial log age are approximately on a log-uniform grid "
        "between the minimum and maximum age allowed by the stellar evolution "
        "interpolation for the given star, with this much smear added to the "
        "fractional log-age (i.e. each log10(age) gets an independent uniform "
        "random variable with range +- half of this value * the interpolation "
        "range added to it.",
    )
    parser.add_argument(
        "--initial-num-mehs",
        type=int,
        default=4,
        help="The number of different initial [M/H] to optimize starting "
        "positions for.",
    )
    parser.add_argument(
        "--initial-meh-smear",
        type=float,
        default=0.1,
        help="See ``--initial-logage-smear``",
    )
    parser.add_argument(
        "--initial-num-ws",
        type=int,
        default=8,
        help="The number of different initial arguments of periapses to "
        "optimize starting positions for.",
    )
    parser.add_argument(
        "--initial-w-smear",
        type=float,
        default=10.0,
        help="See ``--initial-logage-smear``",
    )
    parser.add_argument(
        "--restart-steps",
        type=int,
        default=3000,
        help="To avoid samples being stuck in local minima which under the "
        "emcee algorithm may never be drained, every this many steps the "
        "sampling stars from scratch, initialized with the top distinct samples"
        " from all currently accumulated steps until all walkers end up within "
        "``--restart-log-likelihood-range`` of each other on the last of these "
        "steps. After this, normal sampling to converge proceeds.",
    )
    parser.add_argument(
        "--restart-log-likelihood-range",
        type=float,
        default=30,
        help="If the log-likelihood spread between most and least likely "
        "walker at the end of ``--restart-steps`` is less than this, the true "
        "sampling begins.",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="If passed, the current run will not be considered final.",
    )
    parser.add_argument(
        "--skip-restart-lstsq",
        action="store_true",
        help="If passed, the least-squares optimization step when finding "
        "restart samples is skipped.",
    )
    parser.add_argument(
        "--continue-status",
        type=int,
        default=2,
        help="The status assigned to the tics for which sampling should "
        "continue.",
    )
    parser.add_argument(
        "--changed-likelihood-statuses",
        type=int,
        default=[],
        nargs="+",
        help="The status(es) assigned to the tics for which likelihood has "
        "changed since the last sampling run. This will cause the sampling to "
        "restart from the last step of the existing chain, rather than "
        "continuing.",
    )

    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="The number of parallel processes to use.",
    )
    parser.add_argument(
        "--fname-datetime-format",
        default="%Y%m%d%H%M%S",
        help="How to format date and time as part of filenames (e.g. when "
        "creating output files for multiprocessing.",
    )
    parser.add_argument(
        "--std-out-err-fname",
        default=path.join(
            results_dir, "logs", "tess{tic_id:d}_{task}_{now!s}_{pid:d}.outerr"
        ),
        help="Filename to redirect worker process stdout and stderr to during "
        "multiprocessing. Should include at least `{pid:d}` (worker process "
        "id) substitution to avoid mangling, but may also include `{tic_id:d}`"
        " and `{now}` (approximate date and time the process started).",
    )
    parser.add_argument(
        "--logging-fname",
        default=path.join(
            results_dir, "logs", "tess{tic_id:d}_{task}_{now!s}_{pid:d}.log"
        ),
        help="Filename for log mesasges from sampling. See "
        "``--std-out-err-fname`` for possible substitutions.",
    )
    parser.add_argument(
        "--logging-verbosity",
        "--verbosity",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="The lowest importance level of logging messages to issue.",
    )
    parser.add_argument(
        "--logging-datetime-format",
        default=None,
        help="How to format date and time as part of filenames (e.g. when "
        "creating output files for multiprocessing.",
    )
    parser.add_argument(
        "--logging-message-format",
        "--logging-format",
        "--log-fmt",
        default=default_logging_format,
        help="How to format logging messages. See python logging module "
        "documentation for details.",
    )
    parser.add_argument(
        "--starting-positions-only",
        action="store_true",
        help="If passed, no sampling is performed. Instead a samples files is "
        "created containing only the starting walker positions.",
    )
    parser.add_argument(
        "--overwrite-cache",
        nargs="+",
        default=[],
        type=str.upper,
        choices=["BLS", "SED"],
        help="If passed, the specified cache will be re-computed and "
        "overwritten.",
    )
    parser.add_argument(
        "--ignore-git-hash",
        action="store_true",
        help="If specified the current git hash is not checked against what is "
        "in the file.",
    )
    parser.add_argument(
        "--update-git-hash",
        action="store_true",
        help="If specified the current git hash relpaces what is in the file.",
    )
    parser.add_argument(
        "--max-mcmc-steps",
        type=int,
        default=None,
        help="The maximum number of steps the MCMC sampler is allowed to add. "
        "If specified, restarting to eliminate bad local minima is disabled.",
    )

    return parser.parse_args()


def detect_likelihood_change(tic_id, config):
    """Return True if the likelihood changed for tic_id, False to continue.

    Looks up tic_id in the BUI job-group tables and compares its status
    to --continue-status and --changed-likelihood-statuses.  Raises
    ValueError if the TIC is not found or its status matches neither.
    """

    status = None
    matched_job_group = None
    with Session.begin() as db_session:  # pylint: disable=no-member
        job_groups = db_session.execute(select(JobGroup)).scalars().all()
        for job_group in job_groups:
            SelectTable = get_ticid_select_tables(
                job_group.select_tic_table, must_exist=True
            )[0]
            tic_entry = db_session.execute(
                select(SelectTable).filter_by(id=tic_id, job_group=job_group.id)
            ).scalar_one_or_none()
            if tic_entry is None:
                continue
            assert status is None, f"TIC {tic_id} found in multiple job groups!"
            status = tic_entry.status
            matched_job_group = job_group

        if status is None:
            raise ValueError(f"TIC {tic_id} not found in any job group table.")
        if status == config.continue_status:
            return False, matched_job_group.select_tic_table
        if status in config.changed_likelihood_statuses:
            return True, matched_job_group.select_tic_table
        raise ValueError(
            f"TIC {tic_id} has status {status} in table "
            f"{matched_job_group.select_tic_table!r} "
            f"(job group {matched_job_group.id})"
            f", which is neither the continue status "
            f"({config.continue_status}) nor one of the "
            f"changed-likelihood statuses "
            f"({config.changed_likelihood_statuses})."
        )


def get_backend(samples_fname, config):
    """Return properly configured EMCEE backend to store generated samples."""

    samples_dir = path.dirname(samples_fname)
    try:
        makedirs(samples_dir)
    except FileExistsError:
        if not path.isdir(samples_dir):
            raise

    backend = HDFBackend(samples_fname)
    reset = True
    if path.exists(samples_fname):
        print("Found existing samples file:", samples_fname)
        with h5py.File(
            samples_fname, "r+" if config.update_git_hash else "r"
        ) as samples_file:
            if config.update_git_hash:
                samples_file.attrs["GitHash"] = _git_hash
            assert (
                config.ignore_git_hash
                or samples_file.attrs["GitHash"] == _git_hash
            ), f"Git commit hash changed since {samples_fname!r} was created"
            if "mcmc" in samples_file:
                reset = False
        if not reset:
            _logger.info(
                "Existing chain with %d samples found for TIC ID: %d. "
                "Extending.",
                backend.iteration,
                config.tic_id,
            )
    if reset:
        backend.reset(
            config.num_random_walkers
            + config.initial_num_ages
            * config.initial_num_mehs
            * config.initial_num_ws,
            len(SampleParams._fields),
        )
        with h5py.File(samples_fname, "a") as samples_file:
            samples_file.attrs["TICID"] = config.tic_id
            samples_file.attrs["GitHash"] = _git_hash
        _logger.info(
            "Starting new chain for TIC ID: %d.",
            config.tic_id,
        )

    with h5py.File(samples_fname, "r") as samples_file:
        return backend, samples_file["mcmc"].attrs.get("final_run", False)


def get_last_chain_source(backend):
    """Return ``(log_prob, source_chain_name)`` for the most recent chain.

    ``log_prob`` is loaded fully (it is ~``ndim`` times smaller than the chain
    and needed to rank walkers) from the current ``mcmc`` chain when it has
    data, otherwise from the most recent non-empty ``prelim_mcmc_*`` chain
    (recovery when the process was killed mid-restart, after the previous chain
    was archived but before the first new step was written). The returned name
    is the group those log-probs came from, so the caller can read the matching
    chain rows from it without materializing the whole chain.

    Earlier chains are intentionally ignored: some prelim chains are created
    when the likelihood definition changes, so their log-probabilities are not
    comparable across chains.
    """

    if backend.iteration > 0:
        return backend.get_log_prob(), "mcmc"

    last_name = None
    with h5py.File(backend.filename, "r") as samples_f:
        for prelim in count():
            name = f"prelim_mcmc_{prelim}"
            if name not in samples_f:
                break
            if samples_f[name].attrs.get("iteration", 0) > 0:
                last_name = name
    assert last_name is not None, "No chain with data to restart from."
    source = HDFBackend(backend.filename, name=last_name, read_only=True)
    return source.get_log_prob(), last_name


def get_restart_source(backend):
    """Archive the current chain and return ``(log_prob, source_backend)``.

    Picks the chain to seed the restart from (see ``get_last_chain_source``).
    When the current ``mcmc`` chain has data it is moved to a fresh
    ``prelim_mcmc_*`` slot and ``mcmc`` is reset, so sampling restarts from an
    empty chain. Returns that chain's log-probabilities (fully, for ranking)
    and a read-only backend from which to pull the selected seed rows.
    """

    log_prob, source_name = get_last_chain_source(backend)
    chain_shape = backend.shape
    if backend.iteration > 0:
        with h5py.File(backend.filename, "r+") as samples_f:
            for prelim in count():
                chain_name = f"prelim_mcmc_{prelim}"
                if chain_name not in samples_f:
                    samples_f.move("mcmc", chain_name)
                    # The current chain now lives under its prelim name; read
                    # the seed rows from there.
                    source_name = chain_name
                    break
        backend.reset(*chain_shape)

    _logger.info(
        "Restarting sampling. Last step log-likelihood spread: %s. Choosing "
        "top samples from %d.",
        repr(log_prob[-1].max() - log_prob[-1].min()),
        log_prob.size,
    )
    return (
        log_prob,
        HDFBackend(backend.filename, name=source_name, read_only=True),
    )


def prepare_restart(backend):
    """Prepare to restart sampling and return independent samples & log-prob."""

    if backend.iteration == 0:
        with h5py.File(backend.filename, "r") as samples_f:
            if "seed_log_prob" in samples_f["mcmc"]:
                assert "seed_samples" in samples_f["mcmc"]
                result = (
                    samples_f["mcmc/seed_log_prob"][:],
                    samples_f["mcmc/seed_samples"][:],
                )
            else:
                result = (None, None)
        if result[0] is not None:
            assert result[1] is not None
            result += load_initial_positions(
                backend.filename,
                num_walkers=backend.shape[0],
                num_params=backend.shape[1],
                chain_name="mcmc",
                blobs_dtype=[("tweaked_log_prob", float)],
            )
            if len(result) == 4:
                return result[:3] + (None, None) + (result[3],)
            assert len(result) == 6
            return result

    log_prob, source = get_restart_source(backend)
    chain_shape = backend.shape
    num_walkers = chain_shape[0]

    ordered_indices = numpy.unique(log_prob, return_index=True)[1]
    for select_from in range(num_walkers, ordered_indices.size + 1):
        flat_indices = numpy.random.choice(
            ordered_indices[-select_from:], num_walkers
        )
        _logger.debug(
            "Top indices (shape: %s): %s", flat_indices.shape, flat_indices
        )
        top_indices = numpy.unravel_index(flat_indices, log_prob.shape)
        steps, walkers = top_indices[0], top_indices[1]
        initial_state = source.get_chain_rows(steps, walkers)
        if walkers_independent(initial_state):
            with h5py.File(backend.filename, "r+") as samples_f:
                samples_f["mcmc"].create_dataset(
                    "seed_log_prob", data=log_prob[steps, walkers]
                )
                samples_f["mcmc"].create_dataset(
                    "seed_samples", data=initial_state
                )
            return (
                log_prob[steps, walkers],
                initial_state,
                numpy.empty(chain_shape, dtype=float),
                numpy.empty(num_walkers, dtype=float),
                None,
                numpy.zeros(num_walkers, dtype=bool),
            )

    return (
        None,
        source.get_chain_rows(
            numpy.zeros(num_walkers, dtype=int), numpy.arange(num_walkers)
        ),
    ) + 4 * (None,)


def find_restart_samples(input_queue, output_queue, log_likelihood, config):
    """Find a restart sample given one of the top samples of the old chain."""

    try:
        numpy.random.seed()
        setup_process(task="find_restart_samples", **vars(config))
        for input_ind, input_sample in iter(input_queue.get, "STOP"):
            if config.skip_restart_lstsq:
                output_queue.put(
                    (
                        input_ind,
                        input_sample,
                        input_sample,
                        numpy.nan,
                        numpy.nan,
                    )
                )
            else:
                _logger.info("Found suitable sample for index %d.", input_ind)
                output_queue.put(
                    (input_ind,) + lmfit_and_tweak(input_sample, log_likelihood)
                )

    except:  # pylint: disable=bare-except
        _logger.critical("Restart sample worker failed:\n%s", format_exc())
        output_queue.put(None)


def restart_sampling(
    backend, config, log_likelihood
):  # pylint: disable=too-many-locals
    """Prepare the next round of sampling."""

    (
        seed_log_prob,
        seed_samples,
        initial_state,
        _,
        _,
        positions_found,
    ) = prepare_restart(backend)
    if seed_log_prob is None:
        assert seed_samples is not None
        _logger.warning(
            "Failed to find a set of independent walkers. Continuing "
            "preliminary MCMC."
        )
        return backend, seed_samples, False

    if config.changed_likelihood:
        _logger.info(
            "Log-likelihood function changed since last sampling. "
            "Re-evaluating seed log-likelihoods."
        )
        with ProcessPoolExecutor(
            max_workers=config.num_parallel,
            initializer=setup_process_map,
            initargs=(vars(config),),
            # no max_tasks_per_child: recycling crashed workers on Vista
        ) as pool:
            seed_log_prob = numpy.array(
                [e[0] for e in pool.map(log_likelihood, seed_samples)]
            )

    _logger.debug(
        "Spread in seed log probabilities: %s",
        repr(seed_log_prob.max() - seed_log_prob.min()),
    )

    needed_indices = numpy.flatnonzero(numpy.logical_not(positions_found))
    input_queue = Queue()
    for i in needed_indices:
        input_queue.put((i, seed_samples[i]))

    for _ in range(config.num_parallel):
        input_queue.put("STOP")

    output_queue = Queue()

    workers = [
        Process(
            target=find_restart_samples,
            args=(input_queue, output_queue, log_likelihood, config),
        )
        for _ in range(config.num_parallel)
    ]
    for process in workers:
        process.start()
    for _ in needed_indices:
        result = output_queue.get()
        if result is None:
            # pylint: disable=invalid-name
            for w in workers:
                w.terminate()
            # pylint: enable=invalid-name
            raise RuntimeError("Failed to find initial walker positions.")
        _logger.debug(
            "Proposed initial position %d:\n%s\n%s\n->\n%s\n%s\n->\n%s\n%s\n"
            "Log likelihood: %s (%s) -> %s -> %s. ",
            result[0],
            seed_samples[result[0]],
            log_likelihood.get_sample_params(seed_samples[result[0]]),
            result[1],
            log_likelihood.get_sample_params(result[1]),
            result[2],
            log_likelihood.get_sample_params(result[2]),
            seed_log_prob[result[0]],
            log_likelihood(seed_samples[result[0]])[0],
            result[3],
            result[4],
        )
        start_sample = (
            result[2]
            if result[4] > seed_log_prob[result[0]]
            else seed_samples[result[0]]
        )
        _logger.info(
            "Using %s sample for walker %d:\n%s",
            "tweaked" if result[4] > seed_log_prob[result[0]] else "original",
            result[0],
            start_sample,
        )
        save_initial_position(
            start_sample,
            backend.filename,
            nwalkers=backend.shape[0],
            index=result[0],
            log_prob_result=result[3:5],
        )
        initial_state[result[0]] = start_sample
    return (
        backend,
        initial_state,
        seed_log_prob[-1].max() - seed_log_prob[-1].min()
        < config.restart_log_likelihood_range,
    )


def reinitialize_sampling(backend, final_run):
    """Restart sampling with the last step of an existing chain."""

    _logger.info("Starting sampling from the last step of the existing chain.")
    backend_shape = backend.shape
    initial_state = prepare_restart(backend)[1]
    backend.reset(*backend_shape)
    assert walkers_independent(initial_state)
    with h5py.File(backend.filename, "r+") as samples_f:
        samples_f["mcmc"].attrs["final_run"] = final_run

    return backend, initial_state


def check_log_likelihood_consistency(backend, log_likelihood, config):
    """Raise if recomputed log-likelihoods differ from those stored in file."""

    _logger.info("Verifying log-likelihood consistency with existing chain.")
    # Read only the last step. get_chain()/get_log_prob() without discard
    # pull the whole chain into RAM (tens of GB) just to take the last row.
    last = backend.iteration - 1
    last_positions = backend.get_chain(discard=last)[0]
    stored_log_probs = backend.get_log_prob(discard=last)[0]
    with ProcessPoolExecutor(
        max_workers=config.num_parallel,
        initializer=setup_process_map,
        initargs=(vars(config),),
        # no max_tasks_per_child: recycling crashed workers on Vista
    ) as pool:
        computed = numpy.array(
            [e[0] for e in pool.map(log_likelihood, last_positions)]
        )
    mismatches = numpy.logical_not(
        numpy.isclose(computed, stored_log_probs, rtol=1e-8, atol=1e-08)
    )
    if mismatches.any():
        raise RuntimeError(
            f"Log-likelihood mismatch for {mismatches.sum()} walker(s):\n"
            + "\n".join(
                f"  Walker {i}: computed {computed[i]}, "
                f"stored {stored_log_probs[i]}"
                for i in numpy.flatnonzero(mismatches)
            )
        )
    _logger.info("Log-likelihood consistency check passed.")


def main(config):
    """Avoid polluting global namespace."""

    setup_process(task="mcmc_sampling", **vars(config))

    config.changed_likelihood, config.ticid_select_table = (
        detect_likelihood_change(config.tic_id, config)
    )
    _logger.info(
        "Preparing to sample TIC ID %d, %s log-likelihood change",
        config.tic_id,
        "with" if config.changed_likelihood else "without",
    )

    samples_fname = config.samples_fname_pattern.format(tic_id=config.tic_id)
    backend, final_run = get_backend(samples_fname, config)
    if config.force_restart:
        final_run = False
    log_likelihood = (
        LogLikelihoodPriorsOnly if config.priors_only else LogLikelihood
    )(config.tic_id, overwrite_cache=config.overwrite_cache)
    if log_likelihood.sed_recomputed:
        config.changed_likelihood = True

    initial_state = None
    if backend.iteration == 0:
        with h5py.File(backend.filename, "r") as samples_file:
            has_prelim = "prelim_mcmc_0" in samples_file
        if has_prelim:
            backend, initial_state, final_run = restart_sampling(
                backend, config, log_likelihood
            )
        else:
            initial_state = FindStartingPositions(log_likelihood)(config)
        _logger.info("Full set of initial positions found. Starting sampling.")
    else:
        if config.changed_likelihood:
            backend, initial_state = reinitialize_sampling(backend, final_run)
            if initial_state is not None:
                for pos_ind, pos in enumerate(initial_state):
                    save_initial_position(
                        pos,
                        backend.filename,
                        nwalkers=backend.shape[0],
                        index=pos_ind,
                    )
        elif not final_run and backend.iteration >= config.restart_steps:
            backend, initial_state, final_run = restart_sampling(
                backend, config, log_likelihood
            )
        else:
            check_log_likelihood_consistency(backend, log_likelihood, config)

    with h5py.File(backend.filename, "r+") as samples_f:
        samples_f["mcmc"].attrs["final_run"] = final_run

    if config.starting_positions_only:
        return

    if config.changed_likelihood:
        with Session.begin() as db_session:  # pylint: disable=no-member
            db_session.execute(
                update(
                    get_ticid_select_tables(
                        config.ticid_select_table, must_exist=True
                    )[0]
                )
                .filter_by(id=config.tic_id)
                .values(status=config.continue_status)
            )

    while True:
        _logger.info(
            "Starting%s MCMC run with initial state:\n%s.",
            " final" if final_run else "",
            repr(initial_state),
        )
        with ProcessPoolExecutor(
            max_workers=config.num_parallel,
            initializer=setup_process_map,
            initargs=(vars(config),),
            # no max_tasks_per_child: recycling crashed workers on Vista
        ) as pool:
            EnsembleSampler(
                *backend.shape, log_likelihood, backend=backend, pool=pool
            ).run_mcmc(
                initial_state,
                nsteps=config.max_mcmc_steps
                or (
                    1024**2
                    if final_run
                    else (
                        (int(backend.iteration / config.restart_steps) + 1)
                        * config.restart_steps
                        - backend.iteration
                    )
                ),
                skip_initial_state_check=(
                    initial_state is None and not config.changed_likelihood
                ),
            )
            if not final_run:
                backend, initial_state, final_run = restart_sampling(
                    backend, config, log_likelihood
                )


if __name__ == "__main__":
    numpy.set_printoptions(edgeitems=20)
    main(parse_command_line())
