#!/usr/bin/env python3

"""Create plots of emcee sampling results or lightcurves."""

from os import path, remove, makedirs
from subprocess import run
from glob import glob
import logging

from matplotlib import pyplot, rcParams
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from configargparse import ArgumentParser, DefaultsFormatter
import pandas
import h5py
from asteval import Interpreter

from general_purpose_python_modules.visuals import make_corner_plot
from general_purpose_python_modules.emcee_util import load_initial_positions
from autowisp import Evaluator

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams
from log_likelihood import LogLikelihood
from tess_target import TESSTarget
from paths import samples as samples_fname
from binary import Binary


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
    parser.add_argument("tic_id", type=int, help="TIC ID to create plots for.")
    parser.add_argument(
        "--samples-fname-pattern",
        default=samples_fname,
        help="The filename where to save samples. If the file already exists, "
        "sampling continues, adding more points to the existing chain.",
    )
    parser.add_argument(
        "--corner-plot-fname",
        "--corner-plot",
        "--corner",
        default=None,
        help="If specified, a corner plot is created and saved with the given "
        "filename. Can use ``{tic_id}`` substitution.",
    )
    parser.add_argument(
        "--model-to-data-plot",
        default=None,
        nargs=2,
        metavar=("STEP_IND", "PLOT_FNAME"),
        help="If specified, create a plot comparing model to observed LCs for "
        "the a particular MCMC sample (walker initial positions are sample "
        "-1). Can use ``{tic_id}`` substitution in filename.",
    )
    parser.add_argument(
        "--plot-expressions",
        nargs="+",
        default=[],
        help="If specified, the first argument should be a filaneme, followed "
        "by expression for the x value followed by any number of y expressions "
        "that are all shown on the same plot. Can use ``{tic_id}`` substitution"
        " in filename.",
    )
    parser.add_argument(
        "--expression-movie",
        action="store_true",
        help="If passed, a movie frame is generated per ``--plot-expressions`` "
        "for each iteration and a movie is created.",
    )
    parser.add_argument(
        "--histogram-movie",
        nargs=2,
        metavar=("FILENAME", "EXPRESSION"),
        help="Create a movie where each frame shows a histogram of some "
        "expression of the sample variables at fixed iteration. Thinning and "
        "burn-in control which iterations are included as frames. Can use "
        "``{tic_id}`` substitution in filename.",
    )
    parser.add_argument(
        "--plot-lightcurve",
        nargs=2,
        metavar=("FILENAME", "MOSAIC"),
        help="If specified a plot is created showing the TESS lightcurve of the"
        " selected object. The first argument is a filename (possibly including"
        " ``{tic-id}`` substitution) and the second specifies a layout (see "
        "``pyplot.subplot_mosaic()`` documentation for the syntax. Separate "
        "figure is created for each TESS sector and saved as a page in a PDF "
        "if file extension is PDF, otherwise a long figure is created with all "
        "plots arranged vertically. Plot labels in the mosaic should be one "
        "of the following:\n"
        "\t* full: plot the lightcurve for a sector without any folding.\n"
        "\t* folded: phot the lightcurve folded at the best fit BLS period.\n"
        "\t* zoom_default: plot only vicinity of BLS transit (folded).\n"
        "\t* zoom_even: plot only vicinity of even BLS transits (folded).\n"
        "\t* zoom_odd: plot only vicinity of odd BLS transits (folded).\n"
        "\t* zoom_masked: plot only vicinity of masked BLS transits (folded on "
        "masked period).",
    )
    parser.add_argument(
        "--show-best-model-with-lc",
        action="store_true",
        help="If specified, also finds the highest log-likelihood point among "
        "the samples and plots its model along with the lightcurve for "
        "``--plot-lightcurve``.",
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

    result = parser.parse_args()
    result.samples_fname = result.samples_fname_pattern.format(
        tic_id=result.tic_id
    )
    return result


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
    print(f"TIC: {tic!r} ({type(tic)})")
    log_likelihood = LogLikelihood(tic)
    with PdfPages(config.model_to_data_plot[1]) as pdf:
        for pos in positions:
            print(f"Parameters: {log_likelihood.get_sample_params(pos)}")
            print(5 * "\n")
            try:
                log_likelihood.plot_lc_model_comparison(pos, pdf)
            except ValueError:
                continue


# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
def create_lightcurve_plot(config, binary=None):
    """Create the plot(s) described in the ``--plot-lightcurve`` cmdline arg."""

    def plot_full(lightcurve, model_lc):
        """Show the full unfolded lightcurve for a sector."""

        pyplot.plot(lightcurve["time"], lightcurve["flux"], ".")
        if model_lc is not None:
            pyplot.plot(lightcurve["time"], model_lc, "-k")
        pyplot.xlabel("Time [d]")
        pyplot.ylabel("Flux [ppm]")

    def plot_vs_phase(lightcurve, phase, model_lc):
        """Create a plot of the lightcurve vs the given phase."""

        phase_order = numpy.argsort(phase)
        ordered_phase = phase[phase_order]
        pyplot.plot(ordered_phase, lightcurve["flux"][phase_order], ".")
        if model_lc is not None:
            pyplot.plot(ordered_phase, model_lc[phase_order], "-k")
        pyplot.xlabel("Phase")
        pyplot.ylabel("Flux [ppm]")

    def plot_folded(lightcurve, bls, model_lc):
        """Show sector lightcurve folded by the best fit BLS period."""

        if binary is None:
            period = bls["period"][0]
            if abs(bls["depth_even"][0] - bls["depth_odd"][0]) > 5.0 * (
                bls["depth_even"][1] + bls["depth_odd"][1]
            ):
                period *= 2
        else:
            period = binary.per
        phase = (lightcurve["time"] % period) / period
        plot_vs_phase(lightcurve, phase, model_lc)
        pyplot.xlim(0, 1)
        return period

    def plot_zoomed(lightcurve, bls, zoom, model_lc):
        """
        Show folded lightcurve near some BLS transit.

        Possible values for zoom are "default", "even", "odd", "masked".
        """

        if zoom in ["primary", "secondary"]:
            assert binary is not None
            if zoom == "secondary":
                binary.swap_components()
            period = binary.per
            time_reference = binary.t0
            eval_t = numpy.linspace(
                time_reference - period / 2,
                time_reference + period / 2,
                100,
            )
            eclipse_lc = binary.eclipse(eval_t)
            eclipsed = eval_t[eclipse_lc < 1] - time_reference
            half_xrange = max(abs(eclipsed.min()), eclipsed.max())
            if zoom == "secondary":
                binary.swap_components()
        else:
            if zoom == "default":
                period = bls["period"][0]
                time_reference = bls["transit_time"]
            elif zoom == "masked":
                period = bls["masked_period"][0]
                time_reference = bls["masked_transit_time"]
            else:
                period = 2 * bls["period"][0]
                if zoom == "even":
                    time_reference = bls["transit_time"]
                else:
                    assert zoom == "odd"
                    time_reference = bls["transit_time"] + bls["period"][0]
            half_xrange = (
                bls[("masked_" if zoom == "masked" else "") + "duration"]
                / period
            )

        phase = (
            (lightcurve["time"] - time_reference + period / 2) % period
        ) / period - 0.5
        plot_vs_phase(lightcurve, phase, model_lc)
        pyplot.xlim(-half_xrange, half_xrange)

    tess_target = TESSTarget(config.tic_id)
    bls = None

    full_figure = pyplot.figure(
        figsize=(
            rcParams["figure.figsize"][0],
            rcParams["figure.figsize"][1] * 1.5 * len(tess_target.lcs),
        ),
        layout="constrained",
    )
    mosaic_spec = Interpreter(
        user_symbols={
            plot_type: plot_type
            for plot_type in [
                "full",
                "folded",
                "zoom_default",
                "zoom_even",
                "zoom_odd",
                "zoom_masked",
                "zoom_primary",
                "zoom_secondary",
            ]
        }
    )(config.plot_lightcurve[1])
    subfigures = [
        subfig[0]
        for subfig in full_figure.subfigures(
            len(tess_target.lcs), 1, hspace=0.03, squeeze=False
        )
    ]
    title_pre = f"TIC {config.tic_id}\n"
    model_lc = None
    for (header, lightcurve), subfig in zip(tess_target.lcs, subfigures):
        if binary is not None:
            model_lc = LogLikelihood.get_model(
                binary, header, lightcurve, binary.lc_sys_err
            )[0]
        for plot_type, axis in subfig.subplot_mosaic(
            mosaic_spec, gridspec_kw={"wspace": 0.3, "hspace": 0.3}, sharey=True
        ).items():
            pyplot.sca(axis)
            if plot_type == "full":
                plot_full(lightcurve, model_lc)
            else:
                if bls is None:
                    bls = LogLikelihood.get_cached_sed_and_bls(config.tic_id)[1]
                    if bls is None:
                        bls = tess_target.fit_bls()
                if plot_type == "folded":
                    period = plot_folded(lightcurve, bls, model_lc)
                    if title_pre:
                        title_pre = (
                            title_pre.strip() + f" BLS Porb = {period!r}\n"
                        )
                else:
                    assert plot_type.startswith("zoom_")
                    plot_zoomed(
                        lightcurve, bls, plot_type[len("zoom_") :], model_lc
                    )
        subfig.suptitle(
            title_pre
            + f"Sector {header['sector']} {header['provenance']} lightcurve"
        )
        title_pre = ""
    if isinstance(config.plot_lightcurve[0], tuple):
        pyplot.savefig(
            config.plot_lightcurve[0][0], format=config.plot_lightcurve[0][1]
        )
    else:
        pyplot.savefig(config.plot_lightcurve[0])


# pylint: enable=too-many-statements
# pylint: enable=too-many-locals


def main(config):
    """Avoid polluting global namespace."""

    logging.basicConfig(level=logging.DEBUG)
    if config.model_to_data_plot:
        create_model_to_data_plot(config)

    backend = HDFBackend(config.samples_fname, read_only=True)
    raw_data = backend.get_blobs()
    log_prob = backend.get_log_prob()
    iteration = backend.iteration

    if config.plot_lightcurve:
        if config.show_best_model_with_lc:
            best_index = numpy.unravel_index(
                numpy.argmax(log_prob), log_prob.shape
            )
            best_params = SampleParams(*raw_data[best_index])
            binary = Binary(from_mcmc=best_params)
            binary.lc_sys_err = best_params.lc_sys
        else:
            binary = None
        create_lightcurve_plot(config, binary)

    plot_data = pandas.DataFrame(
        raw_data[config.burn_in :: config.thin, :, :]
        .flatten()
        .reshape(
            (
                (iteration - config.burn_in + config.thin - 1)
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
        log_prob[config.burn_in :: config.thin, :].flatten(),
    )
    if config.corner_plot_fname:
        create_corner_plot(plot_data, config)

    if config.plot_expressions:
        create_expressions_plot(plot_data, config, backend.shape[0])

    if config.histogram_movie:
        create_histogram_movie(plot_data, config, backend.shape[0])


if __name__ == "__main__":
    main(parse_command_line())
