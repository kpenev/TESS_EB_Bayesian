#!/usr/bin/env python3
"""Prepare FITS files of samples for public release."""

import numpy
from matplotlib import pyplot
from general_purpose_python_modules.emcee_quantile_convergence import (
    find_emcee_quantiles,
)
from general_purpose_python_modules import ensure_directory
import pandas
from astropy.table import QTable
from astropy import units as u

from command_line_util import create_parser
import paths
from visualize import get_plot_data
from sample_params import SampleParams, param_units, param_descriptions


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
        default=1e-3,
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
        return int(get_burnin.burnin.loc[config.tic_id])
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


def plot_release_cdf(orig_data, release_data, config):
    """
    Plot cumulative distributions comparing all post-burnin samples to release.

    For each parameter in ``config.diagnostic_params`` a panel is drawn with two
    CDFs: one from the full chain (``orig_data`` after discarding the first
    ``burnin * num_walkers`` entries) and one from the thinned release data.

    Args:
        orig_data:      DataFrame of all samples (rows = steps * walkers).

        release_data:   ``QTable`` of the thinned release samples.

        burnin:         Number of burn-in *steps* (rows to skip =
            ``burnin * num_walkers``).

        num_walkers:    Number of MCMC walkers.

        config:         Parsed command-line config providing
            ``diagnostic_params``, ``tic_id``, and ``release_fname``.
    """

    def setup_figure():
        """Set up the figure and axes for the CDF comparison plot."""

        ncols = min(3, len(config.diagnostic_params))
        nrows = (len(config.diagnostic_params) + ncols - 1) // ncols

        fig, axes = pyplot.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows)
        )
        axes = numpy.atleast_1d(axes).flatten()
        return fig, axes

    fig, axes = setup_figure()

    for ax, param in zip(axes, config.diagnostic_params):
        full_vals = numpy.sort(orig_data[param].values)
        release_vals = numpy.sort(numpy.array(release_data[param]))

        full_cdf = numpy.arange(1, len(full_vals) + 1) / len(full_vals)
        release_cdf = numpy.arange(1, len(release_vals) + 1) / len(release_vals)

        ax.plot(full_vals, full_cdf, label="full chain")
        ax.plot(release_vals, release_cdf, label="release", linestyle="--")
        ax.set_xlabel(param)
        ax.set_ylabel("CDF")
        ax.legend(fontsize="small")

    for ax in axes[len(config.diagnostic_params) :]:
        ax.set_visible(False)

    fig.suptitle(f"TIC {config.tic_id}")
    fig.tight_layout()

    pdf_fname = config.release_fname.replace(".fits", "_cdf.pdf")
    fig.savefig(pdf_fname)
    pyplot.close(fig)
    print(f"CDF comparison plot saved to: {pdf_fname!r}")


def create_release(config):
    """Create the release of a single TIC ID."""

    plot_data, blobs, log_prob, _, backend = get_plot_data(
        config,
        None,
    )
    num_steps = backend.iteration
    num_walkers = backend.shape[0]

    burnin = get_burnin(plot_data, num_steps, num_walkers, config)
    print(f"Burn-in for {config.tic_id}: {burnin}")
    thin = (num_steps - burnin) // config.release_num_steps
    if thin == 0:
        raise ValueError(
            f"Insufficient number of MCMC steps ({num_steps - burnin}) after "
            f"burnin ({burnin}) to release {config.release_num_steps} steps!"
        )
    burnin = num_steps - thin * (config.release_num_steps - 1) - 1
    release_data = QTable(
        rows=blobs[burnin::thin, :, :].reshape(
            config.release_num_steps * num_walkers, backend.shape[1]
        ),
        names=SampleParams._fields,
        units=param_units,
        descriptions=param_descriptions,
    )
    release_data.add_column(
        log_prob[burnin::thin, :].flatten() * u.dimensionless_unscaled,
        name="log_prob",
    )
    plot_release_cdf(
        plot_data.iloc[burnin * num_walkers :], release_data, config
    )
    ensure_directory(config.release_fname)
    print(f"Release data stats: {release_data.info(['attributes', 'stats'])}")
    release_data.write(config.release_fname, format="fits", overwrite=True)
    print(f"Written to: {config.release_fname!r}")


def main(config):
    """Avoid polluting global namespace."""

    config.chain_name = "mcmc"
    for config.tic_id in config.tic_id_list:
        config.samples_fname = config.samples_fname_pattern.format(
            tic_id=config.tic_id
        )
        config.release_fname = config.release_fname_pattern.format(
            tic_id=config.tic_id
        )
        config.burn_in = 0
        config.thin = 1
        create_release(config)


if __name__ == "__main__":
    main(parse_command_line())
