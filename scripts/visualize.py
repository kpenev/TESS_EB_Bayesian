#!/usr/bin/env python3 #pylint: disable=too-many-lines

"""Create plots of emcee sampling results or lightcurves."""

from os import path, remove, makedirs
from subprocess import run
from glob import glob
import logging
from itertools import repeat, count

from matplotlib import pyplot, rcParams, legend_handler

try:
    from matplotlib import colormaps
except ImportError:
    # For older matplotlib versions (< 3.5)
    from matplotlib import cm as colormaps
import h5py
import numpy
from configargparse import ArgumentParser, DefaultsFormatter
import pandas
from asteval import Interpreter

from general_purpose_python_modules.visuals import make_corner_plot
from general_purpose_python_modules.emcee_util import load_initial_positions
from general_purpose_python_modules.emcee_quantile_convergence import (
    find_emcee_quantiles,
    diagnose_emcee_quantile,
)
from general_purpose_python_modules.multi_pickle import MultiPickle

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams
from log_likelihood import LogLikelihood
from utils import fit_least_squares
from paths import samples as samples_fname
from binary import Binary
from light_curve_plotter import LightCurvePlotter


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
        "--max-plot-steps",
        type=int,
        default=100000,
        help="If the chain contains more than this many steps, only the last "
        "--max-plot-steps are used for plotting.",
    )
    parser.add_argument(
        "--chain-name",
        default="mcmc",
        help="The name of the HDF5 group containin the MCMC chain to "
        "visualize.",
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
        "--plot-quantiles",
        nargs="+",
        metavar=("FILENAME", "CDF1"),
        help=create_quantile_plot.__doc__,
    )
    parser.add_argument(
        "--quantile-plot-show-burnin",
        action="store_true",
        help="If specified, the quantile plot shows the burn-in estimate as "
        "change in line width.",
    )
    parser.add_argument(
        "--show-lstsq",
        action="store_true",
        help="If specified, the generated plots also shows the least-squares "
        "fit to the highest likelihood sample.",
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
        "--show-model-with-lc",
        action="append",
        help="Specify a model or models to show with the lightcurve for "
        "``--plot-lightcurve``. Possible values are:\n"
        "\t* bls Plot the best fit BLS model.\n"
        "\t* top<N:int>: Plot the N highest likelihood samples.\n"
        "\t* random<N:int>: Randomly select N samples.\n"
        "\t* <STEP:int>,<WALKER:int>: use the specified sample. Step indexs "
        "ignores ``--burn-in`` and ``--thin``. Step ``-1`` refers to the "
        "initial walker positions (before sampling started). The walker "
        "specification can be ommitted to show all samples for a given step.\n"
        "If ``--sample-condition`` is specified, the top and random points are "
        "selected only among surviving samples.",
    )
    parser.add_argument(
        "--show-lc-detrending",
        action="store_true",
        default=False,
        help="If specified, ``full`` and ``zoom`` plots show the detrending "
        "applied to the lightcurve and the undetrended lightcurve in addition "
        "to the detrended one.",
    )
    parser.add_argument(
        "--eclipse-model-only",
        action="store_true",
        default=False,
        help="If specified, only the eclipses in the lightcurve will be "
        "modeled, along with simple near-eclipse baseline flux model.",
    )
    parser.add_argument(
        "--plot-convergence",
        nargs=2,
        metavar=("FILENAME", "MOSAIC"),
        help="If specified, a figure is created showing comparison between the "
        "chain length to Raftery-Lewis burn-in estimate and/or quantile "
        "precision estimate for either the directly sampled quantities or those"
        " specified in ``--chain-expression``.",
    )
    parser.add_argument(
        "--check-pickled",
        action="store_true",
        help="If specified, the pickled convergence data is checked for "
        "consistency with the current configuration and reused if consistent, "
        "or computed and pickle if no match is found. If not specified, any "
        "pickled convergence data is ignored.",
    )
    parser.add_argument(
        "--sample-condition",
        default=None,
        help="Condition to impose on the samples, excluding those which do not "
        "satisfy the condition from plotting.",
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
        "--chain-expression",
        default=[],
        action="append",
        help="Add an expression to include in the corner plot. If at least one "
        " expression is specifeid, only these expressions are included in the "
        "corner plot. If no expressions are specified, all MCMC blob entries "
        "are shown. Eech expression is specified as <NAME>=<EXPRESSION>.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help="How many steps to discard from the beginning of the chains. Set "
        "to -1 to plot starting positions (in this case --thin is ignored).",
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
    parser.add_argument(
        "--data-on-top",
        action="store_true",
        help="By default model is plotted on top of the data. If this flag is "
        "passed, the data is plotted on top of the model.",
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
    parser.add_argument(
        "--diagnostic-quantiles",
        type=float,
        nargs="+",
        default=list(numpy.linspace(0.1, 0.9, 9)),
        help="The quantiles at which to use for the Raftery-Lewis diagnostic to"
        " determine convergence.",
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


def get_pickler(filename):
    """Return a MultiPickle instance for the given filename."""

    return MultiPickle(
        filename,
        (
            "samples_fname_pattern",
            "samples_fname",
            "corner_plot_fname",
            "plot_expressions",
            "expression_movie",
            "histogram_movie",
            "plot_lightcurve",
            "show_model_with_lc",
            "show_lc_detrending",
            "eclipse_model_only",
            "plot_convergence",
            "sample_condition",
            "x_range",
            "y_range",
            "histogram_resolution",
            "histogram_range",
            "data_on_top",
        ),
    )


def get_chain_expressions(plot_data, chain_expressions):
    """Evaluate the chain expressions specified on the command line."""

    if "selected" in plot_data:
        plot_data = plot_data[plot_data["selected"]]
    if chain_expressions:
        evaluate = Interpreter(user_symbols=plot_data)
        plot_data = {}
        ranges = {}
        for expression in chain_expressions:
            if ":" in expression:
                value_expression, range_expression = expression.split(":", 1)
                plot_range = tuple(
                    evaluate(v) for v in range_expression.split(":")
                )
            else:
                value_expression = expression
                plot_range = 1.0
            name, value_expression = value_expression.split("=")
            plot_data[name] = evaluate(value_expression)
            ranges[name] = plot_range
    for value in plot_data.values():
        if numpy.atleast_1d(value).size > 1:
            print(f"Plot data: {plot_data!r}")
            plot_data = pandas.DataFrame(plot_data)
            ranges = [ranges[col] for col in plot_data.columns]
            return plot_data, ranges
    return plot_data


def include_in_axis(ax, x, y):
    """Ensure that the given x and y values are included in the given axis."""

    for val, direction in [(x, "x"), (y, "y")]:
        if val is None:
            continue
        ax_min, ax_max = getattr(ax, f"get_{direction}lim")()
        if val < ax_min:
            getattr(ax, f"set_{direction}lim")(val - 0.05 * (ax_max - val))
        elif val > ax_max:
            getattr(ax, f"set_{direction}lim")(
                None, val + 0.05 * (val - ax_min)
            )


def create_corner_plot(
    plot_data, config, _, lstsq_data
):  # pylint: disable=too-many-locals
    """Create and save a corner plot."""

    plot_data, ranges = get_chain_expressions(
        plot_data, config.chain_expression
    )
    print(f"Ranges: {ranges!r}")
    figure = make_corner_plot(
        plot_data,
        corner_plot_fname=None,
        plot_contours=True,
        bins=30,
        labelpad=0.08,
        range=ranges,
    )
    if lstsq_data:
        lstsq_values = get_chain_expressions(
            lstsq_data["lstsq_params"]._asdict(), config.chain_expression
        )
        maxlike_values = get_chain_expressions(
            lstsq_data["best_params"]._asdict(), config.chain_expression
        )
        print(f"Least Squares values: {lstsq_values!r}")
        ndim = len(plot_data.columns)
        axes = numpy.array(figure.axes).reshape((ndim, ndim))
        for yi, y_column in enumerate(plot_data.columns):
            ax = axes[yi, yi]
            mark_y = {
                "lstsq": lstsq_values[y_column],
                "maxlike": maxlike_values[y_column],
            }
            if mark_y["maxlike"] is not None:
                ax.axvline(mark_y["maxlike"], color="r")
            if mark_y["lstsq"] is not None:
                ax.axvline(mark_y["lstsq"], color="g")
                include_in_axis(ax, mark_y["lstsq"], None)

            for xi in range(yi):
                if lstsq_values[plot_data.columns[xi]] is None:
                    continue
                mark_x = {
                    "lstsq": lstsq_values[plot_data.columns[xi]],
                    "maxlike": maxlike_values[plot_data.columns[xi]],
                }
                ax = axes[yi, xi]
                for label, color in [("maxlike", "r"), ("lstsq", "g")]:
                    if mark_x[label] is None or mark_y[label] is None:
                        continue
                    ax.axvline(mark_x[label], color=color)
                    ax.axhline(mark_y[label], color=color)
                    ax.plot(mark_x[label], mark_y[label], "s" + color)
                    include_in_axis(ax, mark_x[label], mark_y[label])

    pyplot.savefig(config.corner_plot_fname)
    pyplot.cla()
    pyplot.clf()


def get_convergence_data(plot_data, config, num_walkers):
    """Prepare the data needed for the convergence plot."""

    num_steps = plot_data.shape[0] // num_walkers
    assert num_walkers * num_steps == plot_data.shape[0]

    if getattr(config, "check_pickled", False):
        pickler = get_pickler("convergence_data.pickle")
        config.num_steps = num_steps
        pickled = pickler.check_for_pickled(config)
        if pickled is not None:
            return pickled[0]

    num_entries = len(plot_data.columns) * len(config.diagnostic_quantiles)
    print(f"Initializing convergence data with {num_entries} entries")
    convergence_data = {
        "value": numpy.empty(num_entries, dtype=float),
        "burnin": numpy.empty(num_entries, dtype=int),
        "stdev": numpy.empty(num_entries, dtype=float),
        "thin": numpy.empty(num_entries, dtype=int),
        "num_steps": num_steps,
    }
    burnin = 0
    result_ind = 0
    for column in plot_data.columns:
        for cdf_value in config.diagnostic_quantiles:
            print(f"Processing quantile {cdf_value} for column {column}")
            convergence_data["burnin"][result_ind] = find_emcee_quantiles(
                plot_data[column].values.reshape(num_steps, num_walkers),
                cdf_value,
                config.burnin_tolerance,
                0,
                max(1, num_steps // 10),
            )[1]
            result_ind += 1
    burnin = convergence_data["burnin"].max()
    print(f"Burnin is {burnin} out of {num_steps} steps.")
    result_ind = 0
    for column in plot_data.columns:
        for cdf_value in config.diagnostic_quantiles:
            if burnin < num_steps:
                samples = plot_data[column].values.reshape(
                    num_steps, num_walkers
                )[burnin:]
            else:
                samples = plot_data[column].values.reshape(
                    num_steps, num_walkers
                )
            print(f"Finding quantile of {column} from {samples.size} samples.")
            quantile = numpy.quantile(samples.flatten(), cdf_value)
            convergence_data["value"][result_ind] = quantile
            print(
                f"Diagnosing {column} CDF({quantile}) = {cdf_value} quantile."
            )
            quantile_info = diagnose_emcee_quantile(
                samples,
                samples.shape[1],
                config.quantile_variance_realizations,
                quantile=quantile,
            )
            if quantile_info[2] is None:
                (
                    convergence_data["stdev"][result_ind],
                    convergence_data["thin"][result_ind],
                ) = (numpy.nan, 1)
            else:
                (
                    convergence_data["stdev"][result_ind],
                    convergence_data["thin"][result_ind],
                ) = quantile_info[1:]
            result_ind += 1

    if getattr(config, "check_pickled", False):
        pickler.add_result(config, convergence_data)

    return convergence_data


def create_convergence_plot(plot_data, config, num_walkers, _):
    """Create a figure to gauge convergence of the chain per Raftery-Lewis."""

    plot_data = get_chain_expressions(plot_data, config.chain_expression)[0]

    print("Creating convergence plot")
    quantile_height = 2 / 3 / len(config.diagnostic_quantiles)
    quantile_offset = 1 / 6

    y_pos = numpy.array(
        [
            quantile_ind + quantile_offset + quantile_height * sub_quantile_ind
            for quantile_ind in range(len(plot_data.columns))
            for sub_quantile_ind in range(len(config.diagnostic_quantiles))
        ]
    )
    convergence_data = get_convergence_data(plot_data, config, num_walkers)

    if len(config.diagnostic_quantiles) <= 10:
        cmap = colormaps["tab10"]
    else:
        cmap = colormaps["tab20"]

    mosaic_spec = Interpreter(
        user_symbols={"burnin": "burnin", "stdev": "stdev"}
    )(config.plot_convergence[1])
    for plot_type, axis in pyplot.subplot_mosaic(
        mosaic_spec,
        empty_sentinel="empty",
        gridspec_kw={"right": 0.95, "top": 0.95},
        figsize=(
            rcParams["figure.figsize"][0],
            rcParams["figure.figsize"][1]
            * len(mosaic_spec)
            / len(mosaic_spec[0]),
        ),
    )[1].items():
        pyplot.sca(axis)
        pyplot.xscale("log")
        if plot_type == "burnin":
            pyplot.axvspan(
                0,
                min(
                    convergence_data["num_steps"],
                    convergence_data["burnin"].max(),
                ),
                zorder=10,
                color="grey",
            )
            pyplot.axvline(
                x=3000,
                color="red",
                linestyle=":",
                zorder=25,
            )
            if convergence_data["burnin"].max() < convergence_data["num_steps"]:
                pyplot.axvspan(
                    min(
                        convergence_data["num_steps"],
                        convergence_data["burnin"].max(),
                    ),
                    convergence_data["num_steps"],
                    zorder=10,
                    color="black",
                )
        elif plot_type == "stdev":
            pyplot.axvline(
                x=0.01,
                color="red",
                linestyle=":",
                zorder=25,
            )

        print(
            f"Plotting bars with y_pos={y_pos!r}, "
            f"length={convergence_data[plot_type]}, "
            f"height={quantile_height}"
        )
        pyplot.barh(
            y_pos,
            convergence_data[plot_type],
            height=quantile_height,
            align="edge",
            zorder=20,
            color=[
                cmap(quantile_ind)
                for _ in plot_data.columns
                for quantile_ind in config.diagnostic_quantiles
            ],
        )
        pyplot.yticks(
            0.5 + numpy.arange(len(plot_data.columns)), plot_data.columns
        )
    pyplot.savefig(config.plot_convergence[0])


def get_lstsq_interpreters(lstsq_data):
    """Return interpreters for least-squares and max-likelihood samples."""

    return {
        "lstsq": Interpreter(
            {
                "logprob": lstsq_data["lstsq_logprob"],
                **lstsq_data["lstsq_params"]._asdict(),
            }
        ),
        "maxlike": Interpreter(
            {
                "logprob": lstsq_data["best_logprob"],
                **lstsq_data["best_params"]._asdict(),
            }
        ),
    }


def create_expressions_plot(
    plot_data, config, num_walkers=None, lstsq_data=None
):
    """Plot expressions inolving sampling vars vs common x."""

    def add_lstsq(colors):
        """Add least-squares fit and max likelihood points to the plot."""

        if not hasattr(add_lstsq, "data"):
            interp = get_lstsq_interpreters(lstsq_data)
            add_lstsq.data = {
                "lstsq_x": interp["lstsq"](config.plot_expressions[1]),
                "lstsq_y": [
                    interp["lstsq"](y_expression)
                    for y_expression in config.plot_expressions[2:]
                ],
                "maxlike_x": interp["maxlike"](config.plot_expressions[1]),
                "maxlike_y": [
                    interp["maxlike"](y_expression)
                    for y_expression in config.plot_expressions[2:]
                ],
            }

        for point, label in [
            ("lstsq", "Least-Squares"),
            ("maxlike", "Max Likelihood"),
        ]:
            plot_x = [add_lstsq.data[f"{point}_x"]] * len(
                add_lstsq.data[f"{point}_y"]
            )
            pyplot.scatter(
                plot_x,
                add_lstsq.data[f"{point}_y"],
                marker="s" if point == "lstsq" else "*",
                c=colors,
                edgecolors="black",
                s=150,
                label=label,
                zorder=100,
            )

    def fix_legend(handle, orig):
        handle.update_from(orig)
        handle.set_marker("o")

    assert len(config.plot_expressions) >= 3
    evaluate = Interpreter(user_symbols=plot_data)
    plot_x = evaluate(config.plot_expressions[1]).to_numpy()
    plot_y = [
        evaluate(y_expression).to_numpy()
        for y_expression in config.plot_expressions[2:]
    ]
    selected = (
        plot_data["selected"].to_numpy()
        if "selected" in plot_data
        else slice(None)
    )

    if config.expression_movie:
        assert num_walkers is not None
        shape = (plot_x.size // num_walkers, num_walkers)
        plot_x = plot_x.reshape(shape)
        plot_y = [y.reshape(shape) for y in plot_y]
        if "selected" in plot_data:
            selected = selected.reshape(shape)
        with MovieMaker(config.plot_expressions[0], plot_x.shape[0]) as movie:
            for frame_ind in range(plot_x.shape[0]):
                frame_selected = (
                    selected[frame_ind] if "selected" in plot_data else None
                )
                colors = [
                    pyplot.plot(
                        plot_x[frame_ind][frame_selected],
                        y[frame_ind][frame_selected],
                        ".",
                        label=label,
                    )[0].get_color()
                    for y, label in zip(plot_y, config.plot_expressions[2:])
                ]
                if lstsq_data:
                    add_lstsq(colors)

                pyplot.legend()
                pyplot.xlim(config.x_range)
                pyplot.ylim(config.y_range)
                movie.add_frame()
    else:
        colors = [
            pyplot.plot(plot_x[selected], y, ",", label=label)[0].get_color()
            for y, label in zip(plot_y[selected], config.plot_expressions[2:])
        ]
        if lstsq_data:
            add_lstsq(colors)

        pyplot.legend(
            handler_map={
                pyplot.Line2D: legend_handler.HandlerLine2D(
                    update_func=fix_legend
                )
            }
        )
        pyplot.xlim(config.x_range)
        pyplot.ylim(config.y_range)
        pyplot.savefig(config.plot_expressions[0])
        pyplot.cla()
        pyplot.clf()


def create_histogram_movie(plot_data, config, num_walkers, lstsq_data=None):
    """See `--histogram-movie` command line argument description."""

    values = Interpreter(user_symbols=plot_data)(
        config.histogram_movie[1]
    ).to_numpy()
    values = values.reshape(values.size // num_walkers, num_walkers)
    if lstsq_data:
        lstsq_interp = get_lstsq_interpreters(lstsq_data)
        lstsq_value = lstsq_interp["lstsq"](config.histogram_movie[1])
        maxlike_value = lstsq_interp["maxlike"](config.histogram_movie[1])

    selected = (
        plot_data["selected"].array.reshape(values.shape)
        if "selected" in plot_data
        else repeat(None)
    )
    with MovieMaker(config.histogram_movie[0], values.shape[0]) as movie:
        for iter_data, selected in zip(values, selected):
            pyplot.hist(
                iter_data[selected].flatten(),
                bins=config.histogram_resolution,
                range=config.histogram_range,
                density=True,
            )
            if lstsq_data:
                pyplot.axvline(
                    lstsq_value, color="g", linestyle="-", label="LSTSQ"
                )
                pyplot.axvline(
                    maxlike_value, color="r", linestyle="-", label="ML"
                )
                include_in_axis(pyplot.gca(), lstsq_value, None)
                include_in_axis(pyplot.gca(), maxlike_value, None)

            movie.add_frame()


def hex_color(color_tuple):
    """Return string of hex color give tuple of 0-1 float values."""

    return "#" + "".join(
        [f"{int(numpy.round(c * 255)):02x}" for c in color_tuple[:3]]
    )


def create_quantile_plot(
    plot_data, config, num_walkers, lstsq_data=None
):  # pylint: disable=too-many-locals
    """Make a plot showing the evolution of quantile(s) of chain expressions."""

    num_steps = plot_data.shape[0] // num_walkers
    plot_x = numpy.arange(num_steps)
    plot_data = get_chain_expressions(plot_data, config.chain_expression)[0]
    if lstsq_data:
        lstsq_data = {
            "lstsq": get_chain_expressions(
                {
                    "logprob": lstsq_data["lstsq_logprob"],
                    **lstsq_data["lstsq_params"]._asdict(),
                },
                config.chain_expression,
            ),
            "maxlike": get_chain_expressions(
                {
                    "logprob": lstsq_data["best_logprob"],
                    **lstsq_data["best_params"]._asdict(),
                },
                config.chain_expression,
            ),
        }
    pyplot.figure(
        figsize=(
            rcParams["figure.figsize"][0],
            rcParams["figure.figsize"][1] * len(plot_data.columns),
        )
    )
    plot_quantiles = [float(q) for q in config.plot_quantiles[1:]]
    if config.quantile_plot_show_burnin:
        orig_quantiles = config.diagnostic_quantiles
        config.diagnostic_quantiles = plot_quantiles
        convergence_data = get_convergence_data(plot_data, config, num_walkers)
        convergence_data = iter(
            [
                (
                    convergence_data["burnin"][i],
                    convergence_data["value"][i],
                )
                for i in range(len(convergence_data["value"]))
            ]
        )
        config.diagnostic_quantiles = orig_quantiles
    for column_i, column in enumerate(plot_data.columns):
        pyplot.subplot(len(plot_data.columns), 1, column_i + 1)
        pyplot.title(column)
        yrange = numpy.inf, -numpy.inf
        for cdf_value in plot_quantiles:
            assert 0.0 <= cdf_value <= 1.0
            plot_y = plot_data[column].to_numpy()
            keep_range = numpy.nanquantile(plot_y, (0.05, 0.95))
            keep_range[0] = max(
                numpy.nanmin(plot_y), 1.1 * keep_range[0] - 0.1 * keep_range[1]
            )
            keep_range[1] += min(
                numpy.nanmax(plot_y), 1.1 * keep_range[1] - 0.1 * keep_range[0]
            )
            plot_y[
                numpy.logical_or(plot_y < keep_range[0], plot_y > keep_range[1])
            ] = numpy.nan

            plot_y = numpy.nanquantile(
                plot_data[column].to_numpy().reshape(num_steps, num_walkers),
                cdf_value,
                axis=1,
            )
            if config.quantile_plot_show_burnin:
                burnin, quantile = next(convergence_data)
                print(
                    f"For {column!r} q={cdf_value} burnin: {burnin}, "
                    f"quantile: {quantile}"
                )
                color = pyplot.plot(
                    plot_x[:burnin],
                    plot_y[:burnin],
                    linewidth=1,
                )[0].get_color()
                pyplot.plot(
                    plot_x[burnin:],
                    plot_y[burnin:],
                    linewidth=3,
                    color=color,
                    label=f"q={cdf_value}",
                )
                pyplot.axhline(y=quantile, color=color, linestyle="--")
                yrange = (min(yrange[0], quantile), max(yrange[1], quantile))

            else:
                pyplot.plot(
                    plot_x,
                    plot_y,
                    label=f"q={cdf_value}",
                )
        if numpy.isfinite(yrange[0]) and numpy.isfinite(yrange[1]):
            pyplot.ylim(
                1.2 * yrange[0] - 0.2 * yrange[1],
                1.2 * yrange[1] - 0.2 * yrange[0],
            )
        if lstsq_data:
            print(
                f"Marking maxlike and lstsq for column {column} at "
                f"{lstsq_data['maxlike'][column]} and "
                f"{lstsq_data['lstsq'][column]} respectively"
            )

            pyplot.axhline(
                y=lstsq_data["maxlike"][column], color="r", linestyle=":"
            )
            pyplot.axhline(
                y=lstsq_data["lstsq"][column], color="g", linestyle=":"
            )
            include_in_axis(pyplot.gca(), None, lstsq_data["maxlike"][column])
            include_in_axis(pyplot.gca(), None, lstsq_data["lstsq"][column])

        pyplot.legend()
    pyplot.savefig(config.plot_quantiles[0])


def get_param_binaries(sample_params):
    """Convert the given sample parameters to binaries for plotting."""

    result = []
    for params in sample_params:
        try:
            result.append(Binary(from_mcmc=params))
        except ValueError:
            pass
    return result


def get_initial_positions(config, num_walkers):
    """Return the initial positions per the given config."""

    print(
        f"Reading initial positions from {config.samples_fname} with args:\n\t"
        + "\n\t".join(
            [
                f"chain_name={config.chain_name}",
                f"num_walkers={num_walkers}",
                f"num_params={len(SampleParams._fields)}",
                "blobs_dtype="
                + repr(
                    [("log_likelihood", float)]
                    + [
                        (f"s{i:02d}", float)
                        for i, _ in enumerate(SampleParams._fields)
                    ]
                ),
            ]
        )
    )

    result = load_initial_positions(
        config.samples_fname,
        chain_name=config.chain_name,
        num_walkers=num_walkers,
        num_params=len(SampleParams._fields),
        blobs_dtype=[("log_likelihood", float)]
        + [(f"s{i:02d}", float) for i, _ in enumerate(SampleParams._fields)],
    )
    return result[0][result[-1]]


def get_walker_step_params(
    raw_data, selection, config, log_likelihood, num_walkers
):
    """Return parameters for <STEP>,<WALKER> specifications."""

    selection = tuple(int(s) for s in selection.split(","))
    if selection[0] == -1:
        initial_positions = get_initial_positions(config, num_walkers)
        if len(selection) != 1:
            initial_positions = [initial_positions[selection[1]]]
        print(
            f"Filtering initial positions:\n{initial_positions!r} with "
            f"condition: {getattr(config, 'sample_condition', 'True')!r}"
        )
        sample_params = [
            log_likelihood.get_sample_params(sample)
            for sample in initial_positions
            if Interpreter(
                user_symbols=(
                    dict(zip(SampleParams._fields, sample))
                    | {
                        "bls_porb": log_likelihood.bls_porb,
                        "bls_duration": log_likelihood.best_fit_bls["duration"],
                    }
                )
            )(getattr(config, "sample_condition", "True"))
        ]
        print(f"Surviving params: {sample_params!r}")
    else:
        if len(selection) == 1:
            if include is not None:
                include = include[
                    selection[0]
                    * num_walkers : (selection[0] + 1)
                    * num_walkers
                ]
                selection = raw_data[selection][include]
            else:
                selection = raw_data[selection]
        else:
            assert selection[0] >= 0
            selection = [raw_data[selection]]

        sample_params = [SampleParams(*sample) for sample in selection]

    return sample_params


def get_lstsq(backend, log_likelihood, config):
    """Max likelihood mcmc sample, parameters, log prob and LSQ fit versions."""

    pickler = MultiPickle("lstsq_data.pickle")
    pickler_config = {
        "samples_fname": backend.filename,
        "num_steps": backend.iteration,
    }
    pickled = pickler.check_for_pickled(pickler_config)
    if pickled is not None:
        return pickled[0]

    log_prob = backend.get_log_prob(discard=config.burn_in, thin=config.thin)
    best_index = numpy.unravel_index(numpy.nanargmax(log_prob), log_prob.shape)
    best_mcmc = backend.get_chain(discard=config.burn_in, thin=config.thin)[
        best_index
    ]
    best_params = backend.get_blobs(discard=config.burn_in, thin=config.thin)[
        best_index
    ]
    lstsq_result = fit_least_squares(log_likelihood, best_mcmc)
    print(f"Least squares fit result: {lstsq_result!r}")

    min_log_prob = log_prob[numpy.isfinite(log_prob)].min()

    result = {
        "best_mcmc": best_mcmc,
        "best_params": SampleParams(*best_params),
        "best_logprob": log_prob[best_index] - min_log_prob,
        "lstsq_mcmc": lstsq_result.x,
        "lstsq_params": log_likelihood.get_sample_params(lstsq_result.x),
        "lstsq_logprob": log_likelihood(lstsq_result.x)[0] - min_log_prob,
    }
    print(
        "LSTSQ result: "
        + "\n\t".join([f"{key}: {value!r}" for key, value in result.items()])
    )
    pickler.add_result(pickler_config, result)
    return result


def get_model_binaries(
    config,
    raw_data,
    log_prob,
    include,
    log_likelihood,
    num_walkers,
    lstsq_data=None,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Return fully set-up binaries per ``--show-model-with-lc``."""

    if raw_data is not None:
        ordered = numpy.flip(
            numpy.unique(log_prob.flatten(), return_index=True)[1]
        )
        print(f"Include: {include!r}")
        print(f"Ordered: {ordered!r}")
        if include is not None:
            ordered = ordered[include.flatten()[ordered]]

        top_params = SampleParams(
            *raw_data[numpy.unravel_index(ordered[0], log_prob.shape)]
        )
    else:
        top_params = None

    result = (
        get_param_binaries([lstsq_data["lstsq_params"]])
        if getattr(config, "show_lstsq", False)
        else []
    )
    for selection in getattr(config, "show_model_with_lc", []):
        print(f"Plotting selection: {selection!r}")
        sample_params = None
        assert selection.strip().startswith("-1") or raw_data is not None

        config.highlight_first_model = top_params is not None

        if selection.startswith("top") or selection.startswith("random"):
            if selection.startswith("top"):
                selection = int(selection[3:])
                selection = ordered[:selection]
            else:
                selection = int(selection[6:])
                selection = numpy.random.choice(
                    numpy.nonzero(include)[0], selection
                )
            selection = raw_data[numpy.unravel_index(selection, log_prob.shape)]
            sample_params = [SampleParams(*sample) for sample in selection]
        elif selection != "bls":
            sample_params = get_walker_step_params(
                raw_data,
                selection,
                config,
                log_likelihood,
                num_walkers,
            )
            if selection.startswith("-1,") and top_params is not None:
                sample_params = [top_params] + sample_params
        result.extend(get_param_binaries(sample_params))

    return result


def get_plot_data(  # pylint: disable=too-many-statements
    config, log_likelihood
):
    """Return the data required to generate the plots spceified by config."""

    num_iterations = 0
    raw_data = None
    log_prob = None
    selected = None
    max_steps = getattr(config, "max_plot_steps", 100000)

    with h5py.File(config.samples_fname, "r") as samples_f:
        for chain_ind in [None] if config.chain_name else count():
            chain_name = f"prelim_mcmc_{chain_ind}"
            if chain_name not in samples_f:
                chain_name = "mcmc"
            backend = HDFBackend(
                config.samples_fname, name=chain_name, read_only=True
            )

            if backend.iteration > 0:
                num_iterations += min(
                    backend.iteration - config.burn_in, max_steps
                )
                burn_in = max(
                    config.burn_in, backend.iteration - num_iterations
                )
                if config.burn_in >= 0:
                    if raw_data is None:
                        raw_data = backend.get_blobs(
                            discard=burn_in, thin=config.thin
                        )
                        log_prob = backend.get_log_prob(
                            discard=burn_in, thin=config.thin
                        )

                    else:
                        raw_data = numpy.concatenate(
                            (
                                raw_data,
                                backend.get_blobs(
                                    discard=burn_in, thin=config.thin
                                ),
                            ),
                            axis=0,
                        )
                        log_prob = numpy.concatenate(
                            (
                                log_prob,
                                backend.get_log_prob(
                                    discard=burn_in, thin=config.thin
                                ),
                            ),
                            axis=0,
                        )
            print(
                f"Read chain {chain_name}. Now raw_data shape: "
                f"{raw_data.shape if raw_data is not None else None}, "
                "log_prob shape: "
                f"{log_prob.shape if log_prob is not None else None}"
            )
            if chain_name == "mcmc":
                break
        if raw_data is not None:
            print("Formatting plot data")
            plot_data = pandas.DataFrame(
                raw_data[: num_iterations // config.thin, :, :]
                .flatten()
                .reshape(
                    (num_iterations // config.thin)
                    * backend.shape[0],
                    backend.shape[1],
                ),
                columns=SampleParams._fields,
            )
            print(f"Calculating shifted log-prob from shape {log_prob.shape}")
            sub_log_prob = log_prob[
                : num_iterations // config.thin, :
            ].flatten()
            print(f"Applying shift on {sub_log_prob.shape} shaped array")
            min_log_prob = sub_log_prob[numpy.isfinite(sub_log_prob)].min()
            print(f"Subtracting minimum {min_log_prob}")
            sub_log_prob -= min_log_prob
            print("Inserting log-probability")
            plot_data.insert(
                0,
                "logprob",
                sub_log_prob,
            )
    if config.burn_in == -1:
        plot_data = get_initial_positions(config, backend.shape[0])
        plot_data = pandas.DataFrame(
            [log_likelihood.get_sample_params(sample) for sample in plot_data],
            columns=SampleParams._fields,
        )
        num_iterations = 1
        selected = None
        raw_data = None
        log_prob = None

    if getattr(config, "sample_condition", None) is not None:
        print(
            f"BLS: porb = {log_likelihood.bls_porb!r}, duration = "
            f"{log_likelihood.best_fit_bls['duration']!r}"
        )
        print(f"Data per: {plot_data.to_dict('series')['per']!r}")
        selected = Interpreter(
            user_symbols=(
                {col: plot_data[col].to_numpy() for col in plot_data.columns}
                | {
                    "bls_porb": log_likelihood.bls_porb,
                    "bls_duration": log_likelihood.best_fit_bls["duration"],
                }
            )
        )(config.sample_condition).reshape(num_iterations, backend.shape[0])

        plot_data.insert(
            0,
            "selected",
            selected.flatten(),
        )

    print("Returning plot and raw data")
    return plot_data, raw_data, log_prob, selected, backend


def main(config):
    """Avoid polluting global namespace."""

    config.highlight_first_model = False
    logging.basicConfig(level=logging.INFO)
    log_likelihood = None
    print(
        f"Reading plot data from {config.samples_fname}/"
        f"{getattr(config, 'chain_name', '')}"
    )
    lstsq_data = None
    if path.exists(config.samples_fname):
        print("Print found samples file. Loading data ...")
        if log_likelihood is None:
            log_likelihood = LogLikelihood(config.tic_id)
        print("Initializing HDF5 backend")
        print("Extracting data from HDF5 file")
        plot_data, raw_data, log_prob, selected, backend = get_plot_data(
            config,
            log_likelihood,
        )
        if getattr(config, "show_lstsq", False):
            lstsq_data = get_lstsq(backend, log_likelihood, config)
    else:
        plot_data = None
        raw_data = None
        log_prob = None
        selected = None
        backend = HDFBackend(
            config.samples_fname,
            name=config.chain_name or "mcmc",
            read_only=True,
        )

    print("Creating plots")

    if getattr(config, "plot_lightcurve", False):
        print("Plotting lightcurve")
        if (
            getattr(config, "show_model_with_lc", False)
            and plot_data is not None
        ):
            if log_likelihood is None:
                log_likelihood = LogLikelihood(config.tic_id)
            binaries = get_model_binaries(
                config,
                raw_data,
                log_prob,
                (
                    selected
                    if getattr(config, "sample_condition", None) is not None
                    else None
                ),
                log_likelihood,
                backend.shape[0],
                lstsq_data,
            )
        else:
            binaries = []

        if "bls" in getattr(config, "show_model_with_lc", []):
            binaries.append("bls")

        LightCurvePlotter(config)(config.tic_id, binaries)

    for attr, plot_func in (
        ("corner_plot_fname", create_corner_plot),
        ("plot_expressions", create_expressions_plot),
        ("histogram_movie", create_histogram_movie),
        ("plot_convergence", create_convergence_plot),
        ("plot_quantiles", create_quantile_plot),
    ):
        if getattr(config, attr, False):
            print(f"Creating {attr.replace('_', ' ')}")
            plot_func(plot_data, config, backend.shape[0], lstsq_data)


if __name__ == "__main__":
    main(parse_command_line())
