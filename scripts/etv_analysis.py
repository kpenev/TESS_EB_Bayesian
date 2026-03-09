#!/usr/bin/env python3
"""Perform ETV analysis on TESS EBs."""

import os

from configargparse import ArgumentParser, DefaultsFormatter
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from scipy.stats import norm as scipy_norm

from general_purpose_python_modules.multiprocessing_util import setup_process

from measure_etv import (
    MeasureETV,
    compute_and_save_etv_distros,
    load_etv_distros,
)
from paths import results_dir, samples as samples_fname_pattern


def plot_etv(  # pylint: disable=too-many-arguments, too-many-locals
    measure_etv,
    etv_distros,
    ax=None,
    *,
    primary_color="tab:blue",
    secondary_color="tab:orange",
    marker_sizes=None,
    sector_filter=None,
):
    """Plot ETV error bars for all loaded distributions.

    x coordinate: mean time of the contributing eclipses
    y center:     best-fit time shift (distribution mode)
    error bars:   1-sigma interval (CDF^{-1}(norm.cdf(±1)))
    marker size:  sector > segment > eclipse
    color:        primary vs secondary
    sector_filter: if given, only plot entries whose sector is in this set

    Returns the matplotlib Axes.
    """

    if marker_sizes is None:
        marker_sizes = {"sector": 10, "segment": 6, "eclipse": 3}
    if ax is None:
        _, ax = pyplot.subplots(layout="constrained")

    cdf_lo = scipy_norm.cdf(-1)
    cdf_hi = scipy_norm.cdf(1)
    num_primary = etv_distros["num_primary_tasks"]
    task_types = etv_distros["task_types"]
    labeled = set()
    zorder = {"sector": 100, "segment": 110, "eclipse": 120}
    task_sectors = etv_distros.get("task_sectors")

    for i, entry in enumerate(etv_distros["distributions"]):
        if (
            sector_filter is not None
            and task_sectors is not None
            and task_sectors[i] not in sector_filter
        ):
            continue
        is_primary = i < num_primary
        component = "Primary" if is_primary else "Secondary"
        task_type = task_types[i]
        color = primary_color if is_primary else secondary_color
        ms = marker_sizes[task_type]
        x = measure_etv.get_mean_time(entry["eclipse_indices"])
        distro = entry["distro"]
        y_errorbar = (
            numpy.array([distro.mode, distro.ppf(cdf_lo), distro.ppf(cdf_hi)])
            * 24
            * 60
        )
        label_key = (component, task_type)
        label = f"{component} {task_type}" if label_key not in labeled else None
        labeled.add(label_key)
        ax.errorbar(
            x,
            y_errorbar[0],
            yerr=[
                [y_errorbar[0] - y_errorbar[1]],
                [y_errorbar[2] - y_errorbar[0]],
            ],
            fmt="o",
            color=color,
            markersize=ms,
            capsize=ms,
            label=label,
            zorder=zorder[task_type],
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("ETV (min)")
    ax.figure.legend(loc="outside upper center", ncol=2)
    return ax


def _consecutive_sector_groups(sectors):
    """Return list of sets of consecutively-numbered sectors."""

    unique_sorted = sorted(set(sectors))
    groups = []
    current = [unique_sorted[0]]
    for sec in unique_sorted[1:]:
        if sec == current[-1] + 1:
            current.append(sec)
        else:
            groups.append(set(current))
            current = [sec]
    groups.append(set(current))
    return groups


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["mcmc_sampling.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "tic_id", type=int, help="The TIC identifier to sample."
    )
    parser.add_argument(
        "--samples-fname-pattern",
        default=samples_fname_pattern,
        help="The filename where to save samples. If the file already exists, "
        "sampling continues, adding more points to the existing chain.",
    )
    parser.add_argument(
        "--max-abs-etv",
        type=float,
        default=None,
        help="The maximum ETV to allow.",
    )
    parser.add_argument(
        "--disable-by-sector",
        action="store_true",
        default=False,
        help="If passed, ETVs are not extracted from entire TESS sectors.",
    )
    parser.add_argument(
        "--disable-by-segment",
        action="store_true",
        default=False,
        help="If passed, ETVs are not extracted from TESS lightcurve segments.",
    )
    parser.add_argument(
        "--disable-by-eclipse",
        action="store_true",
        default=False,
        help="If passed, ETVs are not extracted from individual eclipses.",
    )
    parser.add_argument(
        "--etv-distros-fname",
        default=os.path.join(results_dir, "tess{tic_id:d}_etv_distros.pkl"),
        help="Filename template for the pickle file of ETV distributions. "
        "May include `{tic_id:d}` substitution.",
    )

    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="The number of parallel processes to use.",
    )
    parser.add_argument(
        "--fname-datetime-format",
        default="%Y%m%d%H%M%S",
        help="How to format date and time as part of filenames (e.g. when "
        "creating output files for multiprocessing.",
    )
    parser.add_argument(
        "--std-out-err-fname",
        default=os.path.join(
            results_dir, "logs", "tess{tic_id:d}_{task}_{now!s}_{pid:d}.outerr"
        ),
        help="Filename to redirect worker process stdout and stderr to during "
        "multiprocessing. Should include at least `{pid:d}` (worker process "
        "id) substitution to avoid mangling, but may also include `{tic_id:d}`"
        " and `{now}` (approximate date and time the process started).",
    )
    parser.add_argument(
        "--logging-fname",
        default=os.path.join(
            results_dir, "logs", "tess{tic_id:d}_{task}_{now!s}_{pid:d}.log"
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
        default="%(levelname)s %(asctime)s %(name)s: %(message)s | "
        "%(pathname)s.%(funcName)s:%(lineno)d",
        help="How to format logging messages. See python logging module "
        "documentation for details.",
    )
    parser.add_argument(
        "--ignore-existing-pickle",
        action="store_true",
        help="If passed, even if a pickle file exists it is generated from "
        "scratch.",
    )

    return parser.parse_args()


def prepare_tasks(eclipse_indices, config):
    """Define the computing tasks that need to be carried out."""

    tasks = []
    task_types = []
    task_sectors = []
    num_primary_tasks = None
    for component_eclipse_indices in eclipse_indices:
        if not config.disable_by_sector:
            sector_tasks = [
                numpy.concatenate(sector_eclipses[1])
                for sector_eclipses in component_eclipse_indices
            ]
            tasks.extend(sector_tasks)
            task_types.extend("sector" for _ in sector_tasks)
            task_sectors.extend(sec for sec, _ in component_eclipse_indices)
        if not config.disable_by_segment:
            for sec, segs in component_eclipse_indices:
                tasks.extend(segs)
                task_types.extend("segment" for _ in segs)
                task_sectors.extend(sec for _ in segs)
        if not config.disable_by_eclipse:
            for sec, segs in component_eclipse_indices:
                for segment_eclipses in segs:
                    tasks.extend(numpy.array([e]) for e in segment_eclipses)
                    task_types.extend("eclipse" for _ in segment_eclipses)
                    task_sectors.extend(sec for _ in segment_eclipses)
        if num_primary_tasks is None:
            num_primary_tasks = len(tasks)
    return tasks, task_types, task_sectors, num_primary_tasks


def analyze_eb(config):
    """Perform ETV analysis for an eclipsing binary."""

    setup_process(task="etv_analysis", **vars(config))
    measure_etv = MeasureETV(
        config.tic_id, config.samples_fname_pattern, config.max_abs_etv
    )
    tasks, task_types, task_sectors, num_primary_tasks = prepare_tasks(
        measure_etv.get_eclipse_indices(0.5), config
    )

    # print("Tasks:\n\t" + "\n\t".join([str(t) for t in tasks]))

    output_fname = config.etv_distros_fname.format(tic_id=config.tic_id)
    if config.ignore_existing_pickle and os.path.exists(output_fname):
        os.remove(output_fname)
    if not os.path.exists(output_fname):
        compute_and_save_etv_distros(
            measure_etv,
            tasks,
            config.num_parallel,
            output_fname,
            samples_fname_pattern=config.samples_fname_pattern,
            max_abs_etv=config.max_abs_etv,
            num_primary_tasks=num_primary_tasks,
            task_types=task_types,
            task_sectors=task_sectors,
        )
    etv_distros = load_etv_distros(output_fname)
    plot_fname = output_fname.replace(".pkl", ".pdf")
    with PdfPages(plot_fname) as pdf:
        ax = plot_etv(measure_etv, etv_distros)
        pdf.savefig(ax.figure)
        pyplot.close(ax.figure)
        for sector_group in _consecutive_sector_groups(
            etv_distros["task_sectors"]
        ):
            ax = plot_etv(measure_etv, etv_distros, sector_filter=sector_group)
            sectors = sorted(sector_group)
            ax.set_title(f"Sectors {sectors[0]}\u2013{sectors[-1]}")
            pdf.savefig(ax.figure)
            pyplot.close(ax.figure)


if __name__ == "__main__":
    analyze_eb(parse_command_line())
