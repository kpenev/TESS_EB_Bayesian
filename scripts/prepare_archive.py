#!/usr/bin/env python3
"""Burn-in and thin sample files for archiving."""

import os.path
import pickle

from matplotlib import pyplot
import scipy.stats
import numpy
import pandas
from configargparse import ArgumentParser, DefaultsFormatter

# from general_purpose_python_modules.emcee_quantile_convergence import (
#    find_quantile_burnin,
#    diagnose_emcee_quantile,
# )
# from general_purpose_python_modules import ensure_directory

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams
import paths


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["archive.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "tic_ids",
        type=int,
        nargs="+",
        help="List of TIC IDs whose sample files to prepare for archiving.",
    )
    parser.add_argument(
        "--samples-fname-pattern",
        default=paths.samples,
        help="The filename where to save samples. If the file already exists, "
        "sampling continues, adding more points to the existing chain.",
    )
    parser.add_argument(
        "--chain-name",
        default="mcmc",
        help="The name of the HDF5 group containin the MCMC chain to "
        "visualize.",
    )
    parser.add_argument(
        "--num-archive-steps",
        type=int,
        default=256,
        help="How many steps to include in the archive.",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="If passed, the burn-in and thinning are determined based on the "
        "MCMC samples instead of the blobs. NOT IMPLEMENTED!!!",
    )
    parser.add_argument(
        "--cdf-tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for matching CDF values when chosing the thinning and "
        "burn-in to apply.",
    )
    parser.add_argument(
        "--diagnostic-quantiles",
        type=float,
        nargs="+",
        default=list(numpy.linspace(0.1, 0.9, 9)),
        help="The quantiles at which to use for the Raftery-Lewis diagnostic to"
        " determine convergence.",
    )
    parser.add_argument(
        "--burnin-tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for the Raftery-Lewis burn-in estimate.",
    )
    parser.add_argument(
        "--quantile-variance-realizations",
        type=int,
        default=10000,
        help="The number of realizations to use for the Raftery-Lewis variance "
        "estimate for the quantiles.",
    )
    return parser.parse_args()


def get_raw_data(samples_fname, config, nsamples, thin):
    """Return the samples in the given file as pandas DataFrame."""

    backend = HDFBackend(samples_fname, name=config.chain_name, read_only=True)

    def format_result(data):
        return pandas.DataFrame(
            data.flatten().reshape(
                backend.iteration * backend.shape[0], backend.shape[1]
            ),
            columns=(
                (f"var{i}" for i in range(len(SampleParams._fields)))
                if config.use_samples
                else SampleParams._fields
            ),
        )

    raw_data = backend.get_blobs(
        discard=backend.iteration - 2 * nsamples * thin, thin=thin
    )

    return (
        format_result(raw_data[nsamples:]),
        format_result(raw_data[:nsamples]),
        backend.shape[0],
    )


def get_quantile_burnin(raw_data, config, num_walkers):
    """Find the burn-in required for quantiles to converge."""

    num_steps = raw_data.shape[0] // num_walkers
    assert num_walkers * num_steps == raw_data.shape[0]

    num_entries = len(raw_data.columns) * len(config.diagnostic_quantiles)
    print(f"Initializing convergence data with {num_entries} entries")
    result = (
        numpy.empty(num_entries, dtype=float),
        numpy.empty(num_entries, dtype=int),
    )
    result_ind = 0
    for column in raw_data.columns:
        for cdf_value in config.diagnostic_quantiles:
            print(
                f"Finding burn-in for quantile {cdf_value} for column {column}"
            )
            burnin, quantile = find_quantile_burnin(
                raw_data[column].values.reshape(num_steps, num_walkers),
                cdf_value,
                config.burnin_tolerance,
                max(1, num_steps // 10),
            )
            print(f"Quantile: {quantile}, burnin: {burnin}")
            result[0][result_ind] = quantile
            result[1][result_ind] = burnin

            result_ind += 1

    return result


def get_archive_burnin(raw_data, quantile_burnin, config, num_walkers):
    """Return the burn-in to apply for archiving."""

    num_steps = raw_data.shape[0] // num_walkers
    quantile_ind = -1
    for column in raw_data.columns:
        for target_cdf in config.diagnostic_quantiles:
            quantile_ind += 1
            if quantile_burnin[1][quantile_ind] < 0:
                continue
            samples = raw_data[column].values.reshape(num_steps, num_walkers)
            cdf_estimates, cdf_stddev, thin = diagnose_emcee_quantile(
                samples=samples[quantile_burnin[1][quantile_ind] :],
                num_walkers=num_walkers,
                variance_realizations=config.quantile_variance_realizations,
                quantile=quantile_burnin[0][quantile_ind],
            )
            print(
                f"For {column}, target CDF={target_cdf}: "
                + "\n\t".join(
                    [
                        f"CDF estimates: {cdf_estimates}",
                        f"CDF stddev: {cdf_stddev}",
                        f"Thin: {thin}",
                    ]
                )
            )


def validate_selection(raw_data, thin, config):
    """Return True iff the given burnin and thinning are suitable."""

    archive = raw_data[-config.num_archive_steps * thin :, :, :]
    compare = raw_data[
        -2 * config.num_archive_steps * thin : -config.num_archive_steps * thin,
        :,
        :,
    ]
    for quantity in ["mtotal", "mratio", "age_gyr", "meh", "ecc"]:
        quantity_i = SampleParams._fields.index(quantity)
        ks_test = scipy.stats.anderson_ksamp(
            (archive[:, :, quantity_i].flatten(),
            compare[:, :, quantity_i].flatten()),
            #method="asymp",
        )
        pyplot.hist(
            archive[:, :, quantity_i].flatten(),
            bins=100,
            facecolor='none',
            edgecolor='r',
            density=True
        )
        pyplot.hist(
            compare[:, :, quantity_i].flatten(),
            bins=100,
            facecolor='none',
            edgecolor='b',
            density=True
        )
        pyplot.show()

        if ks_test.statistic > config.cdf_tolerance:
            return False
    return True


def find_best_thinning(raw_data, config):
    """Select the samples to include in the archive to preserve distrbutions."""

    thin_upper = 1
    while validate_selection(raw_data, thin_upper, config):
        thin_lower = thin_upper
        print(f"{thin_lower} <= thin < {thin_upper}")
        thin_upper *= 2
    while thin_upper > thin_lower + 1:
        thin_try = (thin_upper + thin_lower) // 2
        if validate_selection(raw_data, thin_try, config):
            thin_lower = thin_try
        else:
            thin_upper = thin_try
    return thin_lower


def main(config):
    """Avoid global variables."""

    for tic_id in config.tic_ids:
        backend = HDFBackend(
            config.samples_fname_pattern.format(tic=tic_id),
            name=config.chain_name,
            read_only=True,
        )
        raw_data = backend.get_blobs()

        thin = find_best_thinning(raw_data, config)
        print(
            f"Keeping final {thin * config.num_archive_steps} steps, "
            f"discarding {backend.iteration - thin * config.num_archive_steps} "
            f"with thinning {thin}."
        )

        raw_data = raw_data[-config.num_archive_steps * thin_lower :, :, :]
    return

    convergence_pickle = paths.convergence_pickle.format(
        tic_id=config.tic_ids[0]
    )
    if os.path.exists(convergence_pickle):
        with open(convergence_pickle, "rb") as f:
            quantile_burnin = pickle.load(f)
    else:
        quantile_burnin = get_quantile_burnin(data, config, num_walkers)
        ensure_directory(convergence_pickle)
        with open(convergence_pickle, "wb") as f:
            pickle.dump(quantile_burnin, f)

    print(quantile_burnin)
    get_archive_burnin(data, quantile_burnin, config, num_walkers)


if __name__ == "__main__":
    main(parse_command_line())
