#!/usr/bin/env python3

"""Use EMCEE to sample a TESS eclipsing binary."""

from os import path, makedirs
import logging
from multiprocessing import Pool
from itertools import count

from configargparse import ArgumentParser, DefaultsFormatter
from emcee import EnsembleSampler
import h5py
import numpy

from general_purpose_python_modules.multiprocessing_util import (
    setup_process,
    setup_process_map,
)
from general_purpose_python_modules.emcee_util import save_initial_position

from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import SampleParams, LogLikelihood, LogLikelihoodPriorsOnly
from paths import results_dir
from paths import samples as samples_fname_pattern
from find_starting_positions import FindStartingPositions

_logger = logging.getLogger(__name__)

_default_logging_format = (
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
        default=0.1,
        help="The initial age are approximately on a log-uniform grid with this"
        "much smear added (i.e. each log10(age) gets an independent uniform "
        "random variable with range +- half of this value added to it.",
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
        default=300,
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
        default=300,
        help="If the log-likelihood spread between most and least likely "
        "walker at the end of ``--restart-steps`` is less than this, the true "
        "sampling begins.",
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
        default=_default_logging_format,
        help="How to format logging messages. See python logging module "
        "documentation for details.",
    )

    return parser.parse_args()


def get_backend(samples_fname, config):
    """Return properly configured EMCEE backend to store generated samples."""

    samples_dir = path.dirname(samples_fname)
    try:
        makedirs(samples_dir)
    except FileExistsError:
        if not path.isdir(samples_dir):
            raise

    backend = HDFBackend(samples_fname)
    if path.exists(samples_fname):
        _logger.info(
            "Existing chain with %d samples found for TIC ID: %d. Extending.",
            backend.iteration,
            config.tic_id,
        )
    else:
        backend.reset(
            config.num_random_walkers
            + config.initial_num_ages
            * config.initial_num_mehs
            * config.initial_num_ws,
            len(SampleParams._fields),
        )
        with h5py.File(samples_fname, "a") as samples_file:
            samples_file.attrs["TICID"] = config.tic_id
        _logger.info(
            "Starting new chain for TIC ID: %d.",
            config.tic_id,
        )

    with h5py.File(samples_fname, "r") as samples_file:
        return backend, samples_file["mcmc"].attrs.get("final_run", False)


def restart_sampling(backend, config):
    """Prepare the next round of sampling."""

    log_prob = backend.get_log_prob()
    last_spread = log_prob[-1].max() - log_prob[-1].min()
    if last_spread < config.restart_log_likelihood_range:
        _logger.info(
            "Log-likelihood spread (%s) within %s. Starting final sampling.",
            repr(last_spread),
            repr(config.restart_log_likelihood_range),
        )
        with h5py.File(backend.filename, "a") as samples_file:
            samples_file["mcmc"].attrs["final_run"] = True
        return backend, None, True
    _logger.info(
        "Log-likelihood spread %s. Restarting sampling.",
        repr(last_spread),
    )
    top_indices = numpy.unique(log_prob, return_index=True)[1]
    assert top_indices.size >= backend.shape[0]
    top_indices = top_indices[-backend.shape[0] :]
    _logger.debug("Top indices (shape: %s): %s", top_indices.shape, top_indices)
    top_indices = numpy.unravel_index(top_indices, log_prob.shape)
    initial_state = backend.get_chain()[top_indices]
    backend_shape = backend.shape
    with h5py.File(backend.filename, "r+") as samples_f:
        group_name = "mcmc"
        for i in count():
            group_name = f"prelim_mcmc_{i}"
            if group_name not in samples_f:
                break
        samples_f.move("mcmc", group_name)
    backend.reset(*backend_shape)
    for pos_ind, pos in enumerate(initial_state):
        save_initial_position(
            pos, backend.filename, nwalkers=backend.shape[0], index=pos_ind
        )
    return backend, initial_state, False


def main(config):
    """Avoid polluting global namespace."""

    setup_process(task="mcmc_sampling", **vars(config))

    samples_fname = config.samples_fname_pattern.format(tic_id=config.tic_id)
    backend, final_run = get_backend(samples_fname, config)
    log_likelihood = (
        LogLikelihoodPriorsOnly if config.priors_only else LogLikelihood
    )(config.tic_id)

    initial_state = None
    if backend.iteration == 0:
        initial_state = FindStartingPositions(log_likelihood)(config)
        _logger.info("Full set of initial positions found. Starting sampling.")
    elif not final_run and backend.iteration >= config.restart_steps:
        backend, initial_state, final_run = restart_sampling(backend, config)

    while True:
        with Pool(
            config.num_parallel,
            initializer=setup_process_map,
            initargs=[vars(config)],
        ) as pool:
            EnsembleSampler(
                *backend.shape, log_likelihood, backend=backend, pool=pool
            ).run_mcmc(
                initial_state,
                nsteps=(
                    1024**2
                    if final_run
                    else (config.restart_steps - backend.iteration)
                ),
            )
            if not final_run:
                backend, initial_state, final_run = restart_sampling(
                    backend, config
                )


if __name__ == "__main__":
    main(parse_command_line())
