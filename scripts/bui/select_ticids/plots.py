"""Plots that can be used for selecting TESS objects."""

from argparse import Namespace
from base64 import b64encode
from io import BytesIO
from os import path, makedirs
from functools import partial
from traceback import print_exc, format_exc
from time import sleep

from multiprocessing import Pool
import matplotlib
from sqlalchemy import select, update
from configargparse import ArgumentParser, DefaultsFormatter
import numpy
from numpy.random import randint, seed
import h5py

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)

from visualize import main as visualize
from mcmc_sampling import default_logging_format
from paths import (
    results_dir,
    samples as samples_fname_template,
    render_dir as default_render_dir,
)

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import get_ticid_select_table

matplotlib.use("Agg")


def plot_tic(config, fname):
    """Plot or encode lightcurve (and possibly models) with given config."""

    for plot_type in ["lightcurve", "convergence"]:
        plot_attr = getattr(config, f"plot_{plot_type}", None)
        if plot_attr is not None:
            break
    assert plot_attr is not None
    # False positive
    # pylint: disable=possibly-used-before-assignment
    if fname is None:
        png_stream = BytesIO()
        plot_attr[0] = (png_stream, "png")
    else:
        plot_attr[0] = fname
        if not path.exists(path.dirname(fname)):
            makedirs(path.dirname(fname))
    # pylint: enable=possibly-used-before-assignment

    if fname is None or not path.exists(fname):
        # pylint: enable=possibly-used-before-assignment

        visualize(config)

    if fname is None:
        return b64encode(png_stream.getvalue()).decode("utf-8")

    with open(fname, "rb") as plotf:
        return b64encode(plotf.read()).decode("utf-8")


def lightcurve(tic_id, fname=None):
    """Plot the lightcurves available for the given TIC."""

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            plot_lightcurve=[
                None,
                "[[full, full],"
                " [folded, folded],"
                " [zoom_odd, zoom_even],"
                " [sed, zoom_masked]]",
            ],
            remove_lc_trend="moving_median",
            data_on_top=False,
            samples_fname=samples_fname_template.format(tic_id=tic_id),
        ),
        fname,
    )


def starting(tic_id, fname=None):
    """Plot the initial walker models for the given TIC."""

    samples_fname = samples_fname_template.format(tic_id=tic_id)
    chain_name = None
    with h5py.File(samples_fname, "r") as samples_file:
        for candidate_name in ("prelim_mcmc_0", "mcmc"):
            if candidate_name in samples_file:
                chain_name = candidate_name
                break
    if chain_name is None:
        return None

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            plot_lightcurve=[
                None,
                "[[full, full],"
                " [folded, folded],"
                " [zoom_odd, zoom_even],"
                " [sed, zoom_masked]]",
            ],
            remove_lc_trend="moving_median",
            show_model_with_lc="-1",
            data_on_top=True,
            samples_fname=samples_fname,
            chain_name=chain_name,
            burn_in=-1,
            thin=1,
            sample_condition=(
                "(bls_porb - 5 * bls_duration < per) & "
                "(per < bls_porb + 5 * bls_duration)"
            ),
        ),
        fname,
    )


def best(tic_id, fname=None):
    """Plot the best fit model on top of the lightcurve and SED."""

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            plot_lightcurve=[
                None,
                "[[full,         full,        zoom_primary],"
                " [folded,       folded,      zoom_secondary],"
                " [folded_diff,  folded_diff, sed]]",
            ],
            show_model_with_lc="top1",
            data_on_top=False,
            samples_fname=samples_fname_template.format(tic_id=tic_id),
            chain_name="mcmc",
            burn_in=0,
            thin=1,
            sample_condition=None,
        ),
        fname,
    )


def convergence(tic_id, fname=None):
    """Create a plot to compare steps to burni-in."""

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            remove_lc_trend="moving_median",
            plot_convergence=[None, "[burnin]"],
            diagnostic_quantiles=numpy.linspace(0.1, 0.9, 9),
            burnin_tolerance=1e-4,
            quantile_variance_realizations=5,
            samples_fname=samples_fname_template.format(tic_id=tic_id),
            chain_expression=[
                "$M_1+M_2$=mtotal",
                "$M_2/M_1$=mratio",
                "Age=age_gyr",
                "$[M/H]$=meh",
                "$P_{orb}$=per",
                "e=ecc",
                r"$\omega$=w",
                "b=primary_impact_param",
                "$T_0$=eclipse_time",
            ],
            chain_name="mcmc",
            burn_in=0,
            thin=1,
            sample_condition=None,
        ),
        fname,
    )


def ooe_var_removal(tic_id, fname=None):
    """Create plot to show the detrending that removes all OOE variability."""

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            plot_lightcurve=[
                None,
                "[[full, full, full],"
                " [folded, folded, folded],"
                " [zoom_odd, zoom_even, zoom_masked]]",
            ],
            remove_lc_trend="ooe_variability",
            data_on_top=False,
            show_lc_detrending=True,
            highlight_first_model=False,
            samples_fname="",
        ),
        fname,
    )


def render_one(tic_id, tablename, plot_func, render_dir):
    """Render the lightcurve for a single TIC ID."""

    print(
        f"Rendering {plot_func.__name__} for {tic_id} from {tablename} "
        f"to {render_dir}"
    )
    seed()
    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(tablename)
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    try:
        while True:
            try:
                plot_func(tic_id, path.join(render_dir, f"tess{tic_id}.png"))
                break
            except (MemoryError, OSError):
                wait = randint(60)
                print(
                    "".join(
                        ["Plotting error:"]
                        + format_exc()
                        + ["Waiting {wait}s and retrying!"]
                    )
                )
                sleep(wait)
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            db_session.execute(
                update(SelectTICIDs).filter_by(id=tic_id).values(rendered=1)
            )
    # The point is to avoid crashes at all costs
    # pylint: disable=bare-except
    except:
        print_exc()
    # pylint: enable=bare-except
    print("Finished rendering ", tic_id)


def render_all_plots(config):
    """Render the lightcurves plots for a list of TIC IDs for faster review."""

    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(config.table_name)
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    selection = select(SelectTICIDs.id)
    if config.selected_only:
        selection = selection.where(SelectTICIDs.flag == 1)

    # False positive
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        tic_id_list = list(
            db_session.execute(selection.order_by(SelectTICIDs.id)).scalars()
        )[config.start :]

    if config.count is not None:
        tic_id_list = tic_id_list[: config.count]

    print(f"Rendering {len(tic_id_list)} plots for {config.table_name}.")

    render_func = partial(
        render_one,
        tablename=config.table_name,
        plot_func=globals()[config.plot_type],
        render_dir=config.plot_dir.format(config=config),
    )

    if config.num_parallel > 1:
        with Pool(
            min(config.num_parallel, len(tic_id_list)),
            initializer=setup_process_map,
            initargs=[vars(config)],
        ) as pool:
            pool.map(render_func, tic_id_list)
    else:
        for tic_id in tic_id_list:
            render_func(tic_id)


def parse_command_line():
    """Return th ecommand line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["render_plots.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument("plot_type", help="The type of plot to render")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="The starting index within the ordered TIC ID list to process.",
    )
    parser.add_argument(
        "--count", type=int, default=None, help="How many plots to generate."
    )
    parser.add_argument(
        "--table-name",
        default="prsa_ebs",
        help="The name of the database table to get TIC IDs to plot.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        help="The number of parallel processes to use.",
        default=16,
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        default=False,
        help="Only render objects that have been selected.",
    )
    parser.add_argument(
        "--plot-dir",
        default=path.join(
            default_render_dir, "{config.table_name}", "{config.plot_type}"
        ),
        help="Directory where to save the plot files.",
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
            results_dir,
            "logs",
            "render_{table_name}_{start}_{count}_{now!s}_{pid:d}.outerr",
        ),
        help="Filename to redirect worker process stdout and stderr to during "
        "multiprocessing. Should include at least `{pid:d}` (worker process "
        "id) substitution to avoid mangling, but may also include `{tic_id:d}`"
        " and `{now}` (approximate date and time the process started).",
    )
    parser.add_argument(
        "--logging-fname",
        default=path.join(
            results_dir,
            "logs",
            "render_{table_name}_{start}_{count}_{now!s}_{pid:d}.log",
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
        default=default_logging_format,
        help="How to format logging messages. See python logging module "
        "documentation for details.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    render_all_plots(parse_command_line())
