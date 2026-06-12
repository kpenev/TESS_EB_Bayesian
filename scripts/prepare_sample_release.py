#!/usr/bin/env python3
"""Prepare FITS files of samples for public release."""

import os

import numpy
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
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
        default=[0.1, 0.5, 0.9],
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
        type=int,
        default=(1024, 512),
        help="The optimal and nimimum number of steps to include in the "
        "release. Note that for each step there will be many walkers.",
    )
    return parser.parse_args()


def get_burnin(plot_data, num_steps, num_walkers, config):
    """Return the burn-in to use for the TIC ID currently set in config."""

    if not hasattr(get_burnin, "burnin"):
        if os.path.exists("release_burnin.txt"):
            get_burnin.burnin = pandas.read_csv(
                "release_burnin.txt", sep=r"\s+", header=0, index_col="TIC"
            )
        else:
            get_burnin.burnin = pandas.DataFrame(
                data={"TIC": [], "burn-in": []}, dtype=int
            ).set_index("TIC")
    try:
        return int(get_burnin.burnin.loc[config.tic_id])
    except KeyError:
        print(f"No cached burnin found for TIC {config.tic_id}, estimating...")
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


def setup_diagnostic_figure(config):
    """Create a figure with one axes per diagnostic parameter.

    Args:
        config:     Parsed command-line config providing
            ``diagnostic_params``.

    Returns:
        tuple:  ``(fig, axes)`` where ``axes`` is a flat numpy array of
            ``Axes`` objects, one per diagnostic parameter (extra axes are
            hidden).
    """

    params = config.diagnostic_params
    ncols = min(3, len(params))
    nrows = (len(params) + ncols - 1) // ncols

    fig, axes = pyplot.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = numpy.atleast_1d(axes).flatten()

    for ax in axes[len(params) :]:
        ax.set_visible(False)

    fig.suptitle(f"TIC {config.tic_id}")
    return fig, axes


def plot_release_cdf(orig_data, release_data, config, pdf, max_points=1000):
    """
    Plot cumulative distributions comparing all post-burnin samples to release.

    For each parameter in ``config.diagnostic_params`` a panel is drawn with two
    CDFs: one from the full chain (``orig_data``) and one from the thinned
    release data.

    Args:
        orig_data:      DataFrame of post-burnin samples from the full chain.

        release_data:   ``QTable`` of the thinned release samples.

        config:         Parsed command-line config providing
            ``diagnostic_params`` and ``tic_id``.

        pdf:            ``PdfPages`` object to save the figure to.
    """

    fig, axes = setup_diagnostic_figure(config)

    for ax, param in zip(axes, config.diagnostic_params):
        full_vals = numpy.sort(orig_data[param].values)
        release_vals = numpy.sort(numpy.array(release_data[param]))

        full_cdf = numpy.arange(1, len(full_vals) + 1) / len(full_vals)
        release_cdf = numpy.arange(1, len(release_vals) + 1) / len(release_vals)

        if full_vals.size > max_points:
            full_thin = full_vals.size // max_points
            full_vals = full_vals[::full_thin]
            full_cdf = full_cdf[::full_thin]

        ax.plot(full_vals, full_cdf, label="full chain")
        ax.plot(release_vals, release_cdf, label="release", linestyle="--")
        ax.set_xlabel(param)
        ax.set_ylabel("CDF")
        ax.legend(fontsize="small")

    fig.tight_layout()
    pdf.savefig(fig)
    pyplot.close(fig)


def plot_release_quantile_diff(orig_data, release_data, config, pdf):
    """
    Plot the difference between release and expected quantile fractions.

    For each parameter in ``config.diagnostic_params``, computes 99 equally
    spaced quantiles (1%, 2%, ..., 99%) from ``orig_data``, then measures the
    actual fraction of ``release_data`` points below each quantile value and
    plots the difference (actual fraction - expected fraction) vs the expected
    percent.

    Args:
        orig_data:      DataFrame of post-burnin samples from the full chain.

        release_data:   ``QTable`` of the thinned release samples.

        config:         Parsed command-line config providing
            ``diagnostic_params`` and ``tic_id``.

        pdf:            ``PdfPages`` object to save the figure to.
    """

    percentiles = numpy.arange(1, 100)

    fig, axes = setup_diagnostic_figure(config)

    for ax, param in zip(axes, config.diagnostic_params):
        quantile_vals = numpy.percentile(orig_data[param].values, percentiles)
        release_vals = numpy.array(release_data[param])
        release_fractions = (
            numpy.searchsorted(numpy.sort(release_vals), quantile_vals)
            / len(release_vals)
            * 100.0
        )

        ax.plot(percentiles, release_fractions - percentiles)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("expected percentile")
        ax.set_ylabel("release - expected [pp]")
        ax.set_title(param, fontsize="medium")

    fig.tight_layout()
    pdf.savefig(fig)
    pyplot.close(fig)


def create_release(config):
    """Create the release of a single TIC ID."""

    plot_data, blobs, log_prob, _, backend = get_plot_data(config, None)
    num_steps = log_prob.shape[0]
    num_walkers = backend.shape[0]

    burnin = get_burnin(plot_data, num_steps, num_walkers, config)
    print(
        f"Burn-in for {config.tic_id}: {burnin!r} out of {num_steps!r} steps, "
        f"deciding thinning to release {config.release_num_steps!r} steps."
    )
    if num_steps - burnin < config.release_num_steps[1]:
        raise ValueError(
            f"Insufficient number of MCMC steps ({num_steps - burnin}) after "
            f"burnin ({burnin}) to release {config.release_num_steps} steps!"
        )

    if num_steps - burnin < config.release_num_steps[0]:
        thin = 1
    else:
        thin = (num_steps - burnin) // config.release_num_steps
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
    ensure_directory(config.release_fname)

    print("Generating diagnostic plots")
    plot_data = plot_data.iloc[burnin * num_walkers :]
    pdf_fname = config.release_fname.replace(".fits", "_diag.pdf")
    with PdfPages(pdf_fname) as pdf:
        plot_release_cdf(plot_data, release_data, config, pdf)
        plot_release_quantile_diff(plot_data, release_data, config, pdf)
    print(f"Diagnostic plots saved to: {pdf_fname!r}")
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
        if os.path.exists(config.release_fname):
            continue
        config.burn_in = 0
        config.thin = 1
        try:
            create_release(config)
        except ValueError as err:
            print(f"Skipping {config.tic_id}: {err}")


if __name__ == "__main__":
    main(parse_command_line())
