"""Plots that can be used for selecting TESS objects."""

from collections import namedtuple
from base64 import b64encode
from io import BytesIO
from os import path, makedirs
from functools import partial
from traceback import print_exc

from multiprocessing import Pool
import matplotlib
from matplotlib.pyplot import savefig
from sqlalchemy import select, update
from configargparse import ArgumentParser, DefaultsFormatter

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)

from light_curve_plotter import LightCurvePlotter
from mcmc_sampling import default_logging_format
from paths import results_dir

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import get_ticid_select_table

matplotlib.use("Agg")


def lightcurve(tic_id, fname=None):
    """Plot the lightcurves available for the given TIC."""

    # False positive
    # pylint: disable=possibly-used-before-assignment
    if fname is None:
        png_stream = BytesIO()
        destination = (png_stream, "png")
    else:
        destination = fname
        if not path.exists(path.dirname(fname)):
            makedirs(path.dirname(fname))
    # pylint: enable=possibly-used-before-assignment
    if fname is None or not path.exists(fname):
        # pylint: enable=possibly-used-before-assignment

        config = namedtuple("ConfigType", ["plot_lightcurve", "data_on_top"])(
            (
                destination,
                "[[full, full],"
                " [folded, folded],"
                " [zoom_odd, zoom_even],"
                " [sed, zoom_masked]]",
            ),
            False,
        )
        LightCurvePlotter(config)(tic_id)
        if fname is not None:
            savefig(fname)

    if fname is None:
        return b64encode(png_stream.getvalue()).decode("utf-8")

    with open(fname, "rb") as plotf:
        return b64encode(plotf.read()).decode("utf-8")


def render_one(tic_id, tablename, render_dir):
    """Render the lightcurve for a single TIC ID."""

    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(tablename)
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    try:
        lightcurve(tic_id, path.join(render_dir, f"tess{tic_id}.png"))
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


def render_all_plots(config):
    """Render the lightcurves plots for a list of TIC IDs for faster review."""

    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(config.table_name)
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    # False positive
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        tic_id_list = list(
            db_session.execute(
                select(SelectTICIDs.id).order_by(SelectTICIDs.id)
            ).scalars()
        )[config.start :]

    if config.count is not None:
        tic_id_list = tic_id_list[: config.count]

    with Pool(
        config.num_parallel,
        initializer=setup_process_map,
        initargs=[vars(config)],
    ) as pool:
        pool.map(
            partial(
                render_one,
                tablename=config.table_name,
                render_dir=config.plot_dir,
            ),
            tic_id_list,
        )


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
        "--plot-dir",
        default="/mnt/md2/TESS_EBs/prsa_ebs",
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
