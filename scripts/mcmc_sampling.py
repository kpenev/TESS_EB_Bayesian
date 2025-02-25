#!/usr/bin/env python3

"""Use EMCEE to sample a TESS eclipsing binary."""

from os import path, makedirs
import logging
from multiprocessing import Pool

from configargparse import ArgumentParser, DefaultsFormatter
from emcee import EnsembleSampler
import h5py

from general_purpose_python_modules.multiprocessing_util import (
    setup_process,
    setup_process_map,
)

from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import SampleParams, LogLikelihood, LogLikelihoodPriorsOnly
from paths import results_dir
from init_mcmc import get_initial_mcmc_state

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
        default=path.join(
            results_dir,
            "tess{tic_id:d}_samples.h5",
        ),
        help="The filename where to save samples. If the file already exists, "
        "sampling continues, adding more points to the existing chain.",
    )
    parser.add_argument(
        "--num-walkers",
        type=int,
        default=256,
        help="The number of walkers to use if starting a new chain. Ignored if "
        "the samples file already exsits.",
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
            results_dir, "logs", "TESS{tic_id:d}_{now!s}_{pid:d}.outerr"
        ),
        help="Filename to redirect worker process stdout and stderr to during "
        "multiprocessing. Should include at least `{pid:d}` (worker process "
        "id) substitution to avoid mangling, but may also include `{tic_id:d}`"
        " and `{now}` (approximate date and time the process started).",
    )
    parser.add_argument(
        "--logging-fname",
        default=path.join(
            results_dir, "logs", "TESS{tic_id:d}_{now!s}_{pid:d}.log"
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
        backend.reset(config.num_walkers, len(SampleParams._fields))
        with h5py.File(samples_fname, 'a') as samples_file:
            samples_file.attrs['TIC'] = config.tic_id
        _logger.info(
            "Starting new chain for TIC ID: %d.",
            config.tic_id,
        )

    return backend


def main(config):
    """Avoid polluting global namespace."""

    setup_process(**vars(config))

    samples_fname = config.samples_fname_pattern.format(tic_id=config.tic_id)
    backend = get_backend(samples_fname, config)
    log_likelihood = (
        LogLikelihoodPriorsOnly if config.priors_only else LogLikelihood
    )(config.tic_id)

    initial_state = None
    if backend.iteration == 0:
        initial_state = get_initial_mcmc_state(
            log_likelihood, config, samples_fname
        )
        _logger.info('Full set of initial positions found. Starting sampling.')

    with Pool(
        config.num_parallel,
        initializer=setup_process_map,
        initargs=[vars(config)],
    ) as pool:
        EnsembleSampler(
            *backend.shape, log_likelihood, backend=backend, pool=pool
        ).run_mcmc(initial_state, nsteps=1024**2)


if __name__ == "__main__":
    main(parse_command_line())
