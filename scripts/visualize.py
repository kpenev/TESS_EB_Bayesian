#!/usr/bin/env python3

"""Create plots of emcee sampling results or lightcurves."""

from os import path, remove, makedirs
from subprocess import run
from glob import glob
import logging
from itertools import repeat

from matplotlib import pyplot, colormaps
import numpy
from configargparse import ArgumentParser, DefaultsFormatter
import pandas
from asteval import Interpreter

from general_purpose_python_modules.visuals import make_corner_plot
from general_purpose_python_modules.emcee_util import load_initial_positions
from general_purpose_python_modules.emcee_quantile_convergence import (
    find_emcee_quantiles,
)

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams
from log_likelihood import LogLikelihood
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
        help="Specify a model or models to show with the lightcurve for "
        "``--plot-lightcurve``. Possible values are:\n"
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
        "--plot-convergence",
        nargs=2,
        metavar=("FILENAME", "MOSAIC"),
        help="If specified, a figure is created showing comparison between the "
        "chain length to Raftery-Lewis burn-in estimate and/or quantile "
        "precision estimate for either the directly sampled quantities or those"
        " specified in ``--chain-expression``.",
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
        default=numpy.linspace(0.1, 0.9, 9),
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


def get_chain_expressions(plot_data, chain_expressions):
    """Evaluate the chain expressions specified on the command line."""

    if "selected" in plot_data:
        plot_data = plot_data[plot_data["selected"]]
    if chain_expressions:
        evaluate = Interpreter(user_symbols=plot_data)
        split_expressions = [
            expression.split("=") for expression in chain_expressions
        ]
        plot_data = pandas.DataFrame(
            {
                name: evaluate(expression)
                for name, expression in split_expressions
            }
        )
    return plot_data


def create_corner_plot(plot_data, config):
    """Create and save a corner plot."""

    plot_data = get_chain_expressions(plot_data, config.chain_expression)
    make_corner_plot(
        plot_data,
        corner_plot_fname=config.corner_plot_fname,
        plot_contours=False,
        bins=30,
        labelpad=0.08,
    )
    pyplot.cla()
    pyplot.clf()


def create_convergence_plot(plot_data, config, num_walkers):
    """Create a figure to gauge convergence of the chain per Raftery-Lewis."""

    plot_data = get_chain_expressions(plot_data, config.chain_expression)
    num_steps = plot_data.shape[0] // num_walkers
    assert num_walkers * num_steps == plot_data.shape[0]

    if len(config.diagnostic_quantiles) <= 10:
        cmap = colormaps["tab10"]
    else:
        cmap = colormaps["tab20"]

    quantile_height = 2 / 3 / len(config.diagnostic_quantiles)
    quantile_offset = 1 / 6
    y_pos = numpy.array(
        [
            quantile_ind + quantile_offset + quantile_height * sub_quantile_ind
            for quantile_ind in range(len(plot_data.columns))
            for sub_quantile_ind in range(len(config.diagnostic_quantiles))
        ]
    )
    burnin = numpy.empty(y_pos.size, dtype=float)
    burnin_ind = 0
    for column in plot_data.columns:
        for cdf_value in config.diagnostic_quantiles:
            quantile_info = find_emcee_quantiles(
                plot_data[column].values.reshape(num_steps, num_walkers),
                cdf_value,
                config.burnin_tolerance,
                config.quantile_variance_realizations,
                min(100, num_steps),
            )
            burnin[burnin_ind] = quantile_info[-1]
            burnin_ind += 1

    pyplot.xscale("log")
    pyplot.axvspan(0, num_steps, zorder=10, color="black")
    pyplot.barh(
        y_pos,
        burnin,
        height=quantile_height,
        align="edge",
        zorder=20,
        color=[
            cmap(quantile_ind)
            for _ in plot_data.columns
            for quantile_ind in config.diagnostic_quantiles
        ],
    )
    pyplot.yticks(0.5 + numpy.arange(len(plot_data.columns)), plot_data.columns)
    pyplot.savefig(config.plot_convergence[0])


def create_expressions_plot(plot_data, config, num_walkers=None):
    """Plot expressions inolving sampling vars vs common x."""

    assert len(config.plot_expressions) >= 3
    evaluate = Interpreter(user_symbols=plot_data)
    plot_x = evaluate(config.plot_expressions[1])
    plot_y = [
        evaluate(y_expression) for y_expression in config.plot_expressions[2:]
    ]
    selected = plot_data["selected"].array if "selected" in plot_data else None

    if config.expression_movie:
        assert num_walkers is not None
        shape = (plot_x.size // num_walkers, num_walkers)
        plot_x = plot_x.reshape(shape)
        plot_y = [y.reshape(shape) for y in plot_y]
        if selected:
            selected = selected.reshape(shape)
        with MovieMaker(config.plot_expressions[0], plot_x.shape[0]) as movie:
            for frame_ind in range(plot_x.shape[0]):
                frame_selected = (
                    None if selected is None else selected[frame_ind]
                )
                for y, label in zip(plot_y, config.plot_expressions[2:]):
                    pyplot.plot(
                        plot_x[frame_ind][frame_selected],
                        y[frame_ind][frame_selected],
                        ".",
                        label=label,
                    )
                pyplot.legend()
                pyplot.xlim(config.x_range)
                pyplot.ylim(config.y_range)
                movie.add_frame()
    else:
        for y, label in zip(plot_y[selected], config.plot_expressions[2:]):
            pyplot.plot(plot_x[selected], y, ",", label=label)
        pyplot.legend()
        pyplot.xlim(config.x_range)
        pyplot.ylim(config.y_range)
        pyplot.savefig(config.plot_expressions[0])
        pyplot.cla()
        pyplot.clf()


def create_histogram_movie(plot_data, config, num_walkers):
    """See `--histogram-movie` command line argument description."""

    values = Interpreter(user_symbols=plot_data)(config.histogram_movie[1])
    values = values.reshape(values.size // num_walkers, num_walkers)
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
            movie.add_frame()


def get_param_binaries(sample_params, config, **extra_condition_vars):
    """Convert the given sample parameters to binaries for plotting."""

    num_skipped = 0
    result = []
    for params in sample_params:
        if (
            config.show_model_with_lc.strip().startswith("-1")
            and getattr(config, "sample_condition", None) is not None
            and not Interpreter(
                user_symbols=(
                    dict(zip(SampleParams._fields, params))
                    | extra_condition_vars
                )
            )(config.sample_condition)
        ):
            num_skipped += 1
            continue
        try:
            result.append(Binary(from_mcmc=params))
        except ValueError:
            pass
    return result, num_skipped


def get_model_binaries(config, raw_data, log_prob, include, log_likelihood):
    """Return fully set-up binaries per ``--show-model-with-lc``."""

    selection = config.show_model_with_lc
    assert selection.strip().startswith("-1") or raw_data is not None
    sample_params = None
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
    else:
        selection = tuple(int(s) for s in selection.split(","))
        if selection[0] == -1:
            initial_positions = load_initial_positions(
                config.samples_fname, chain_name=config.chain_name
            )
            if len(selection) != 1:
                initial_positions = [initial_positions[selection[1]]]
            sample_params = [
                log_likelihood.get_sample_params(sample)
                for sample in initial_positions
            ]
            if top_params is not None:
                sample_params = [top_params] + sample_params
        elif len(selection) == 1:
            include = include[
                selection[0]
                * log_prob.shape[1] : (selection[0] + 1)
                * log_prob.shape[1]
            ]
            selection = raw_data[selection][include]
        else:
            assert selection[0] >= 0
            selection = [raw_data[selection]]

    if sample_params is None:
        sample_params = [SampleParams(*sample) for sample in selection]

    return get_param_binaries(
        sample_params,
        config,
        bls_porb=log_likelihood.bls_porb,
        bls_duration=log_likelihood.best_fit_bls["duration"],
    )


def get_plot_data(config, backend, log_likelihood):
    """Return the data required to generate the plots spceified by config."""

    num_iterations = backend.iteration
    raw_data = None
    log_prob = None
    selected = None

    if num_iterations > 0:
        raw_data = backend.get_blobs()
        log_prob = backend.get_log_prob()
        if config.burn_in >= 0:
            plot_data = pandas.DataFrame(
                raw_data[config.burn_in : num_iterations : config.thin, :, :]
                .flatten()
                .reshape(
                    (
                        (num_iterations - config.burn_in + config.thin - 1)
                        // config.thin
                    )
                    * backend.shape[0],
                    backend.shape[1],
                ),
                columns=SampleParams._fields,
            )
            sub_log_prob = log_prob[
                config.burn_in : num_iterations : config.thin, :
            ].flatten()
            sub_log_prob -= sub_log_prob[numpy.isfinite(sub_log_prob)].min()
            plot_data.insert(
                0,
                "logprob",
                sub_log_prob,
            )
    if config.burn_in == -1:
        plot_data = load_initial_positions(
            config.samples_fname, chain_name=config.chain_name
        )
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

        if config.burn_in >= 0:
            selected = selected[
                config.burn_in : num_iterations : config.thin, :
            ]

        plot_data.insert(
            0,
            "selected",
            selected.flatten(),
        )

    return plot_data, raw_data, log_prob, selected


def main(config):
    """Avoid polluting global namespace."""

    logging.basicConfig(level=logging.DEBUG)
    log_likelihood = LogLikelihood(config.tic_id)
    if path.exists(config.samples_fname):
        backend = HDFBackend(
            config.samples_fname, name=config.chain_name, read_only=True
        )
        plot_data, raw_data, log_prob, selected = get_plot_data(
            config,
            backend,
            log_likelihood,
        )
    else:
        plot_data = None
        raw_data = None
        log_prob = None
        selected = None

    if getattr(config, "plot_lightcurve", False):
        if config.show_model_with_lc and plot_data is not None:
            binaries, num_skipped = get_model_binaries(
                config,
                raw_data,
                log_prob,
                (
                    selected
                    if getattr(config, "sample_condition", None) is not None
                    else None
                ),
                log_likelihood,
            )
        else:
            num_skipped = 0
            binaries = None
        LightCurvePlotter(config)(
            config.tic_id,
            binaries,
            title_info=(
                f"{len(binaries)} shown, {num_skipped} skipped"
                if num_skipped
                else ""
            ),
        )

    if getattr(config, "corner_plot_fname", False):
        create_corner_plot(plot_data, config)

    if getattr(config, "plot_expressions", False):
        create_expressions_plot(plot_data, config, backend.shape[0])

    if getattr(config, "histogram_movie", False):
        create_histogram_movie(plot_data, config, backend.shape[0])

    if getattr(config, "plot_convergence", False):
        create_convergence_plot(plot_data, config, backend.shape[0])


if __name__ == "__main__":
    main(parse_command_line())
