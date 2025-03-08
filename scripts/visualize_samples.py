#!/usr/bin/env python3

"""Create plots of emcee sampling results."""

from os import path, remove, makedirs
from subprocess import run
from glob import glob
import logging

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from configargparse import ArgumentParser, DefaultsFormatter
import pandas
import h5py

from general_purpose_python_modules.visuals import make_corner_plot
from general_purpose_python_modules.emcee_util import load_initial_positions

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams
from autowisp import Evaluator
from log_likelihood import LogLikelihood


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
        "--model-to-data-plot",
        default=None,
        nargs=2,
        metavar=('STEP_IND', 'PLOT_FNAME'),
        help="If specified, create a plot comparing model to observed LCs for "
        "the a particular MCMC sample (walker initial positions are sample "
        "-1).",
    )
    parser.add_argument(
        "--plot-expressions",
        nargs="+",
        default=[],
        help="If specified, the first argument should be a filaneme, followed "
        "by expression for the x value followed by any number of y expressions "
        "that are all shown on the same plot.",
    )
    parser.add_argument(
        "--expression-movie",
        action="store_true",
        help="If passed, a movie frame is generated per ``--plot-expressions`` "
        "for each iteration and a movie is created.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        help="The x range for plotting expressions.",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        help="The y range for plotting expressions.",
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
    parser.add_argument(
        "--histogram-movie",
        nargs=2,
        help="Create a movie where each frame shows a histogram of some "
        "expression of the sample variables at fixed iteration. Thinning and "
        "burn-in control which iterations are included as frames.",
    )
    parser.add_argument(
        "--histogram-resolution",
        type=int,
        default=20,
        help="The number of bins to use when creating histograms for the "
        "``--histogram-movie`` option.",
    )
    parser.add_argument(
        "--histogram-range",
        type=float,
        nargs=2,
        default=None,
        help="Specify a custom range for the histograms for the "
        "``--histogram-movie`` option.",
    )

    return parser.parse_args()


class MovieMaker:
    """Allow creating a movie one frame at a time."""

    _frame_dir = "movie_frames"
    _frame_fname_pattern = path.join(_frame_dir, "%06d.png")

    def __init__(self, movie_fname, num_frames=None):
        """Prepare to create a movie with the given filename."""

        self._fname = movie_fname
        self._num_frames = num_frames
        self._frame_ind = 0

    def __enter__(self):
        """Clean-up the frames directory."""

        if path.exists(self._frame_dir):
            for fname in glob(path.join(self._frame_dir, "*.png")):
                remove(fname)
        else:
            makedirs(self._frame_dir)
        if path.exists(self._fname):
            remove(self._fname)
        self._frame_ind = 0
        return self

    def __exit__(self, _, __, ___):
        """Assemble the frames into a movie."""

        run(
            [
                "ffmpeg",
                "-i",
                self._frame_fname_pattern,
                "-vcodec",
                "mpeg4",
                "-r",
                "10",
                self._fname,
            ],
            check=True,
        )

    def add_frame(self):
        """Add the current figure as a movie frame."""

        if self._frame_ind % 10 == 0:
            print(
                f"Creating frames: {self._frame_ind}"
                + (f" / {self._num_frames}" if self._num_frames else "")
                + "\r"
            )
        pyplot.savefig(self._frame_fname_pattern % self._frame_ind)
        pyplot.cla()
        pyplot.clf()
        self._frame_ind += 1


def create_corner_plot(plot_data, config):
    """Create and save a corner plot."""

    for param in config.corner_plot_log_params:
        plot_data[param] = numpy.log10(plot_data[param])
    plot_data.rename(
        columns={
            param: f"log10({param})" for param in config.corner_plot_log_params
        },
        inplace=True,
    )

    make_corner_plot(
        plot_data,
        corner_plot_fname=config.corner_plot_fname,
        plot_contours=False,
        bins=30,
    )
    pyplot.cla()
    pyplot.clf()


def create_expressions_plot(plot_data, config, num_walkers=None):
    """Plot expressions inolving sampling vars vs common x."""

    assert len(config.plot_expressions) >= 3
    evaluate = Evaluator(plot_data)
    plot_x = evaluate(config.plot_expressions[1])
    plot_y = [
        evaluate(y_expression) for y_expression in config.plot_expressions[2:]
    ]
    if config.expression_movie:
        assert num_walkers is not None
        shape = (plot_x.size // num_walkers, num_walkers)
        plot_x = plot_x.reshape(shape)
        plot_y = [y.reshape(shape) for y in plot_y]
        with MovieMaker(config.plot_expressions[0], plot_x.shape[0]) as movie:
            for frame_ind in range(plot_x.shape[0]):
                for y, label in zip(plot_y, config.plot_expressions[2:]):
                    pyplot.plot(
                        plot_x[frame_ind], y[frame_ind], ".", label=label
                    )
                pyplot.legend()
                pyplot.xlim(config.x_range)
                pyplot.ylim(config.y_range)
                movie.add_frame()
    else:
        for y, label in zip(plot_y, config.plot_expressions[2:]):
            pyplot.plot(plot_x, y, ",", label=label)
        pyplot.legend()
        pyplot.xlim(config.x_range)
        pyplot.ylim(config.y_range)
        pyplot.savefig(config.plot_expressions[0])
        pyplot.cla()
        pyplot.clf()


def create_histogram_movie(plot_data, config, num_walkers):
    """See `--histogram-movie` command line argument description."""

    values = Evaluator(plot_data)(config.histogram_movie[1])
    values = values.reshape(values.size // num_walkers, num_walkers)
    with MovieMaker(config.histogram_movie[0], values.shape[0]) as movie:
        for iter_data in values:
            pyplot.hist(
                iter_data,
                bins=config.histogram_resolution,
                range=config.histogram_range,
                density=True,
            )
            movie.add_frame()


def create_model_to_data_plot(config):
    """Compare model to observed LCs for the initial walker positions."""

    step = int(config.model_to_data_plot[0])
    assert step >= -1
    if step == -1:
        positions = load_initial_positions(config.samples_fname)
    else:
        backend = HDFBackend(config.samples_fname, read_only=True)
        positions = backend.get_chain(discard=step, thin=1000000)[0]

    with h5py.File(config.samples_fname, "r") as samples_file:
        tic = int(samples_file.attrs["TICID"])
    print(f'TIC: {tic!r} ({type(tic)})')
    log_likelihood = LogLikelihood(tic)
    with PdfPages(config.model_to_data_plot[1]) as pdf:
        for pos in positions:
            print(f'Parameters: {log_likelihood.get_sample_params(pos)}')
            print(5*'\n')
            log_likelihood.plot_lc_model_comparison(pos, pdf)


def main(config):
    """Avoid polluting global namespace."""

    logging.basicConfig(level=logging.DEBUG)
    if config.model_to_data_plot:
        create_model_to_data_plot(config)

    backend = HDFBackend(config.samples_fname, read_only=True)
    raw_data = backend.get_blobs()
    plot_data = pandas.DataFrame(
        raw_data[config.burn_in :: config.thin, :, :]
        .flatten()
        .reshape(
            (
                (backend.iteration - config.burn_in + config.thin - 1)
                // config.thin
            )
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
    if config.corner_plot_fname:
        create_corner_plot(plot_data, config)

    if config.plot_expressions:
        create_expressions_plot(plot_data, config, backend.shape[0])

    if config.histogram_movie:
        create_histogram_movie(plot_data, config, backend.shape[0])


if __name__ == "__main__":
    main(parse_command_line())
