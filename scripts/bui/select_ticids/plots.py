#!/usr/bin/env python3

"""Plots that can be used for selecting TESS objects."""

from argparse import Namespace
from base64 import b64encode
from io import BytesIO
from os import path, makedirs
from functools import partial
from traceback import print_exc, format_exc
from time import sleep
from itertools import count

from multiprocessing import Pool
import matplotlib
from sqlalchemy import select, or_, and_
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
from exclude_data import exclude_data

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import get_ticid_select_tables

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

        for retry in count():
            try:
                visualize(config)
                break
            except MemoryError:
                if retry == 10:
                    return ""
                wait = randint(60)
                print(
                    "Plotting error:\n"
                    + format_exc()
                    + f"\nWaiting {wait}s and retrying!"
                )
                sleep(wait)
            except:  # pylint: disable=bare-except
                print(f"Error while plotting {config.tic_id}:\n{format_exc()}")
                return ""

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
            show_model_with_lc=["-1"],
            data_on_top=True,
            samples_fname=samples_fname,
            chain_name=chain_name,
            burn_in=-1,
            thin=1,
            #            sample_condition=(
            #                "(bls_porb - 5 * bls_duration < per) & "
            #                "(per < bls_porb + 5 * bls_duration)"
            #            ),
        ),
        fname,
    )


def best(tic_id, fname=None):
    """Plot the best fit model on top of the lightcurve and SED."""

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            remove_lc_trend="moving_median",
            plot_lightcurve=[
                None,
                "[[full,         full,        zoom_primary],"
                " [folded,       folded,      zoom_secondary],"
                " [folded_diff,  folded_diff, sed]]",
            ],
            show_model_with_lc=["top1"],
            data_on_top=False,
            samples_fname=samples_fname_template.format(tic_id=tic_id),
            chain_name="mcmc",
            burn_in=0,
            thin=1,
            sample_condition=None,
            eclipse_model_only="OOE" in exclude_data.get(tic_id, []),
        ),
        fname,
    )


def convergence(tic_id, fname=None):
    """Create a plot to compare steps to burni-in."""

    return plot_tic(
        Namespace(
            tic_id=tic_id,
            remove_lc_trend="moving_median",
            plot_convergence=[None, "[[burnin], [stdev]]"],
            diagnostic_quantiles=numpy.linspace(0.1, 0.9, 9),
            burnin_tolerance=1e-3,
            quantile_variance_realizations=100,
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


def render_one(tic_id, tablename, plot_func, render_dir, samples_template):
    """Render the lightcurve for a single TIC ID."""

    globals()["samples_fname_template"] = samples_template
    print(
        f"Rendering {plot_func.__name__} for {tic_id} from {tablename} "
        f"to {render_dir}"
    )
    seed()
    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    Rendered = get_ticid_select_tables(tablename)[1]
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    try:
        if plot_func(tic_id, path.join(render_dir, f"tess{tic_id}.png")):
            # False positivie
            # pylint: disable=no-member
            with Session.begin() as db_session:
                # pylint: enable=no-member
                db_session.add(Rendered(id=tic_id, plot=plot_func.__name__))
    # The point is to avoid crashes at all costs
    # pylint: disable=bare-except
    except:
        print_exc()
    # pylint: enable=bare-except
    print("Finished rendering ", tic_id)


def get_tics_to_render(config):
    """Select from DB the TIC IDs for which plots should be generated."""

    def get_job_clause(job_str):
        """Return SQL condition to match the given job."""

        job_group, job_id = job_str.split(":")
        job_group = int(job_group)
        if job_id:
            job_id = int(job_id)
            return and_(
                SelectTICIDs.job_group  # pylint: disable=no-member
                == job_group,
                SelectTICIDs.job_id == job_id,  # pylint: disable=no-member
            )
        return SelectTICIDs.job_group == job_group  # pylint: disable=no-member

    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs, RenderedTable = get_ticid_select_tables(
        config.table_name,
        plot_dirs=(config.plot_dir.format(config=config),),
        refresh_rendered=True
    )
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    if config.manual_tics is not None:
        return config.manual_tics

    selection = select(SelectTICIDs.id).join(
        RenderedTable,
        and_(
            SelectTICIDs.id == RenderedTable.id,
            RenderedTable.plot == config.plot_type,
        ),
        isouter=True,
    )
    if config.limit_to_statuses:
        selection = selection.where(
            SelectTICIDs.status.in_(  # pylint: disable=no-member
                config.limit_to_statuses
            )
        )
    if config.limit_to_jobs:
        selection = selection.where(
            or_(*[get_job_clause(job_str) for job_str in config.limit_to_jobs])
        )
    if config.skip_rendered:
        selection = selection.where(RenderedTable.id == None)

    with Session.begin() as db_session:  # pylint: disable=no-member
        # pylint: enable=no-member
        tic_id_list = list(
            db_session.execute(
                selection.order_by(SelectTICIDs.id)  # pylint: disable=no-member
            ).scalars()
        )[config.start :]

    if config.plot_type in ["starting", "best", "convergence"]:
        print(
            f"Restricting {len(tic_id_list)} TIC IDs to existing samples files."
        )
        tic_id_list = [
            tic_id
            for tic_id in tic_id_list
            if path.exists(config.samples_fname_template.format(tic_id=tic_id))
        ]
        print(f"{len(tic_id_list)} surviving TICs")

    if config.count is not None:
        return tic_id_list[: config.count]

    return tic_id_list


def render_all_plots(config):
    """Render the lightcurves plots for a list of TIC IDs for faster review."""

    tic_id_list = get_tics_to_render(config)
    print(f"Rendering {len(tic_id_list)} plots for {config.table_name}.")

    render_func = partial(
        render_one,
        tablename=config.table_name,
        plot_func=globals()[config.plot_type],
        render_dir=config.plot_dir.format(config=config),
        samples_template=config.samples_fname_template,
    )

    if config.num_parallel > 1:
        with Pool(
            min(config.num_parallel, len(tic_id_list)),
            initializer=setup_process_map,
            initargs=[vars(config)],
            maxtasksperchild=1,
        ) as pool:
            pool.map(render_func, tic_id_list, chunksize=1)
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
        "--samples-fname-template",
        default=samples_fname_template,
        help="The template for the samples file name.",
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
        "--manual-tics",
        type=int,
        nargs="+",
        default=None,
        help="If given, render plots for exactly these TIC IDs.",
    )
    parser.add_argument(
        "--limit-to-statuses",
        nargs="+",
        default=False,
        help="Only render objects that have one of the specified statuses.",
    )
    parser.add_argument(
        "--limit-to-jobs",
        metavar="JOBGRP:[JOB]",
        nargs="+",
        default=False,
        help="Only render objects sampled in the given jobs specified as "
        "{job group}:{job}. Omit job to select all jobs from a group.",
    )
    parser.add_argument(
        "--skip-rendered",
        action="store_true",
        help="If passed, only attempts to plot TICs which are not flagged as "
        "rendered.",
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
