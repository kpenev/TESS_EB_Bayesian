#!/usr/bin/env python3
"""Prepare FITS files of samples for public release."""

import numpy
from general_purpose_python_modules.emcee_quantile_convergence import (
    find_emcee_quantiles,
)
import pandas

from command_line_util import create_parser
import paths
from visualize import get_plot_data
from log_likelihood import LogLikelihood


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument(
        "tic_id_list",
        nargs="+",
        type=int,
        help="The TIC IDs for which to prepare a relaese of the samples.",
    )
    parser.add_argument(
        "--samples-fname-pattern",
        default=paths.samples,
        help="The filename pattern where samples were saved.",
    )
    parser.add_argument(
        "--release-fname-pattern",
        default=paths.release,
        help="The filename pattern to create for each TIC ID.",
    )
    parser.add_argument(
        "--burnin-tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for the Raftery-Lewis burn-in estimate.",
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
        "--diagnostic-params",
        nargs="+",
        default=[
            "mtotal",
            "mratio",
            "age_gyr",
            "meh",
            "per",
            "ecc",
            "w",
            "primary_impact_param",
            "eclipse_time",
        ],
        help="The quantiles at which to use for the Raftery-Lewis diagnostic to"
        " determine convergence.",
    )

    parser.add_argument(
        "--release-num-steps",
        default=1024,
        help="The number of steps to include in the release. Note that for each"
        " step there will be many walkers.",
    )
    return parser.parse_args()


def get_burnin(plot_data, num_steps, num_walkers, config):
    """Return the burn-in to use for the TIC ID currently set in config."""

    if not hasattr(get_burnin, "burnin"):
        get_burnin.burnin = pandas.read_csv(
            "release_burnin.txt", sep=r"\s+", header=0, index_col="TIC"
        )
    try:
        return get_burnin.burnin.loc[config.tic_id]
    except KeyError:
        burnin = 0
        for column in config.diagnostic_params:
            for cdf_value in config.diagnostic_quantiles:
                burnin = max(
                    burnin,
                    find_emcee_quantiles(
                        plot_data[column].values.reshape(
                            num_steps, num_walkers
                        ),
                        cdf_value,
                        config.burnin_tolerance,
                        0,
                        max(1, num_steps // 10),
                    )[1],
                )
        get_burnin.burnin.loc[config.tic_id] = burnin
        get_burnin.burnin.to_csv("release_burnin.txt", sep=" ", mode="w")
        return burnin


def create_release(config):
    """Create the release of a single TIC ID."""

    log_likelihood = LogLikelihood(config.tic_id)
    plot_data, raw_data, log_prob, selected, backend = get_plot_data(
        config,
        log_likelihood,
    )
    num_steps = backend.iteration
    num_walkers = backend.shape[0]

    burnin = get_burnin(plot_data, num_steps, num_walkers, config)
    print(f"Burn-in for {config.tic_id}: {burnin}")


def main(config):
    """Avoid polluting global namespace."""

    config.chain_name = "mcmc"
    for config.tic_id in config.tic_id_list:
        config.samples_fname = config.samples_fname_pattern.format(
            tic_id=config.tic_id
        )
        config.burn_in = 0
        config.thin = 1
        create_release(config)


if __name__ == "__main__":
    main(parse_command_line())
