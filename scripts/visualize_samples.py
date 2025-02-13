#!/usr/bin/env python3

"""Create plots of emcee sampling results."""


import numpy
from configargparse import ArgumentParser, DefaultsFormatter
import pandas

from general_purpose_python_modules.visuals import make_corner_plot

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["plotting.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument("samples_fname", help="The saved samples to plot.")
    parser.add_argument(
        "--corner-plot-fname",
        "--corner-plot",
        "--corner",
        default=None,
        help="If specified, a corner plot is created and saved with the given "
        "filename.",
    )
    parser.add_argument(
        "--corner-plot-log-params",
        default=[],
        nargs="+",
        help="Specify a list of parameters for which log10(parameter) instead "
        "of parameter should be plotted in corner plot.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help="How many steps to discard from the beginning of the chains.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="The thinning factor to apply to the chains (1 for no thinning).",
    )

    return parser.parse_args()


def main(config):
    """Avoid polluting global namespace."""

    backend = HDFBackend(config.samples_fname)
    raw_data = backend.get_blobs()
    plot_data = pandas.DataFrame(
        raw_data[config.burn_in :: config.thin, :, :]
        .flatten()
        .reshape(
            ((backend.iteration + config.thin - 1) // config.thin)
            * backend.shape[0],
            backend.shape[1],
        ),
        columns=SampleParams._fields,
    )
    plot_data.insert(
        0,
        "logprob",
        backend.get_log_prob()[config.burn_in :: config.thin, :].flatten(),
    )
    for param in config.corner_plot_log_params:
        plot_data[param] = numpy.log10(plot_data[param])
    plot_data.rename(
        columns={
            param: f"log10({param})" for param in config.corner_plot_log_params
        },
        inplace=True
    )

    make_corner_plot(
        plot_data,
        corner_plot_fname=config.corner_plot_fname,
        plot_contours=False,
        bins=30,
    )


if __name__ == "__main__":
    main(parse_command_line())
