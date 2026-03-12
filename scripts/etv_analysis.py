#!/usr/bin/env python3
"""Perform ETV analysis on TESS EBs."""

import os

from configargparse import ArgumentParser, DefaultsFormatter
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy
from scipy.stats import norm as scipy_norm

from general_purpose_python_modules.multiprocessing_util import setup_process

from measure_etv import (
    MeasureETV,
    compute_and_save_etv_distros,
    load_etv_distros,
)
from paths import results_dir, samples as samples_fname_pattern
from exclude_data import exclude_data


def plot_etv(  # pylint: disable=too-many-arguments, too-many-locals
    measure_etv,
    etv_distros,
    ax=None,
    *,
    primary_color="tab:blue",
    secondary_color="tab:orange",
    marker="o",
    marker_sizes=None,
    sector_filter=None,
):
    """
    Plot ETV error bars for all loaded distributions.

    Args:
        x coordinate:    mean time of the contributing eclipses

        y center:    best-fit time shift (distribution mode)

        error bars:    1-sigma interval (CDF^{-1}(norm.cdf(±1)))

        marker:      marker symbol (use different symbols per detrending)

        marker size:    sector > segment > eclipse

        color:    primary vs secondary

        sector_filter:    if given, only plot entries whose sector is in this
            set

    Returns:
        The matplotlib Axes.
    """

    if marker_sizes is None:
        marker_sizes = {"sector": 10, "segment": 6, "eclipse": 3}
    if ax is None:
        _, ax = pyplot.subplots(layout="constrained")

    cdf_lo = scipy_norm.cdf(-1)
    cdf_hi = scipy_norm.cdf(1)
    num_primary = etv_distros["num_primary_tasks"]
    task_types = etv_distros["task_types"]
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
        ax.errorbar(
            x,
            y_errorbar[0],
            yerr=[
                [y_errorbar[0] - y_errorbar[1]],
                [y_errorbar[2] - y_errorbar[0]],
            ],
            fmt=marker,
            color=color,
            markersize=ms,
            capsize=ms,
            zorder=zorder[task_type],
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("ETV (min)")
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
        default=os.path.join(
            results_dir, "tess{tic_id:d}_etv_distros_{detrending}.pkl"
        ),
        help="Filename template for the pickle file of ETV distributions. "
        "May include `{tic_id:d}` and `{detrending:s}` substitution.",
    )
    parser.add_argument(
        "--plot-fname",
        default=os.path.join(results_dir, "tess{tic_id:d}_etv_distros.pdf"),
        help="Filename template for the plots. May include `{tic_id:d}` "
        "substitution.",
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
    parser.add_argument(
        "--disable-exclude-data",
        action="store_true",
        help="If passed, ETV is fit even for sectors that were not used for "
        "MCMC",
    )
    parser.add_argument(
        "--detrending",
        choices=["none", "spline", "both"],
        help="Specify the kind of detrending to apply to the out-of-eclipse "
        "lightcurve. If ``both``, comparison is shown between no detrending "
        "and spline detrending.",
    )
    parser.add_argument(
        "--eclipse-tweak-order",
        nargs="?",
        type=int,
        default=1,
        const=None,
        help="Specify the polynomial order of the tweaking allowed for the "
        "eclipse. Invoke with no argument to disable eclipse tweaking.",
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
            sector_tasks = [t for t in sector_tasks if t.size > 0]
            tasks.extend(sector_tasks)
            task_types.extend("sector" for _ in sector_tasks)
            task_sectors.extend(sec for sec, _ in component_eclipse_indices)
        if not config.disable_by_segment:
            for sec, segs in component_eclipse_indices:
                segs = [t for t in segs if t.size > 0]
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


def plot_etv_analysis_results(all_etv_distros, etv_comparisons, config):
    """Create plots showing the results of the ETV analysis."""

    detrending_list = list(all_etv_distros.keys())
    markers = ["o", "s", "^", "D"]
    primary_color = "tab:blue"
    secondary_color = "tab:orange"
    marker_sizes = {"sector": 10, "segment": 6, "eclipse": 3}

    all_sectors = set()
    for etv_distros in all_etv_distros.values():
        all_sectors.update(etv_distros["task_sectors"])
    sector_groups = _consecutive_sector_groups(sorted(all_sectors))

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=primary_color, markersize=8, label="Primary",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=secondary_color, markersize=8, label="Secondary",
        ),
    ] + [
        Line2D(
            [0], [0], marker=markers[i % len(markers)],
            color="black", linestyle="none", markersize=8, label=detrending,
        )
        for i, detrending in enumerate(detrending_list)
    ] + [
        Line2D(
            [0], [0], marker="o", color="black",
            linestyle="none", markersize=ms, label=task_type,
        )
        for task_type, ms in marker_sizes.items()
    ]

    plot_fname = config.plot_fname.format(tic_id=config.tic_id)
    with PdfPages(plot_fname) as pdf:
        for page_sector_filter, page_title in [(None, None)] + [
            (sg, f"Sectors {min(sg)}\u2013{max(sg)}") for sg in sector_groups
        ]:
            _, ax = pyplot.subplots(layout="constrained")
            for i, detrending in enumerate(detrending_list):
                plot_etv(
                    etv_comparisons[detrending],
                    all_etv_distros[detrending],
                    ax,
                    primary_color=primary_color,
                    secondary_color=secondary_color,
                    marker=markers[i % len(markers)],
                    marker_sizes=marker_sizes,
                    sector_filter=page_sector_filter,
                )
            ax.figure.legend(
                handles=legend_handles,
                loc="outside upper center",
                ncol=3,
            )
            if page_title is not None:
                ax.set_title(page_title)
            pdf.savefig(ax.figure)
            pyplot.close(ax.figure)


def analyze_eb(config):
    """Perform ETV analysis for an eclipsing binary."""

    if config.disable_exclude_data:
        exclude_data.disable_exclusions()
    setup_process(task="etv_analysis", **vars(config))
    etv_comparisons = {
        detrending: MeasureETV(
            config.tic_id,
            config.samples_fname_pattern,
            max_abs_etv=config.max_abs_etv,
            no_detrend=(detrending == "none"),
            eclipse_tweak_order=config.eclipse_tweak_order,
        )
        for detrending in (
            ["spline", "none"]
            if config.detrending == "both"
            else [config.detrending]
        )
    }
    tasks, task_types, task_sectors, num_primary_tasks = prepare_tasks(
        next(iter(etv_comparisons.values())).get_eclipse_indices(0.5), config
    )
    # print("Tasks:\n\t" + "\n\t".join([str(t) for t in tasks]))
    all_etv_distros = {}
    for detrending, measure_etv in etv_comparisons.items():
        output_fname = config.etv_distros_fname.format(
            tic_id=config.tic_id, detrending=detrending
        )
        if config.ignore_existing_pickle and os.path.exists(output_fname):
            os.remove(output_fname)
        if not os.path.exists(output_fname):
            compute_and_save_etv_distros(
                measure_etv,
                tasks,
                output_fname,
                config,
                samples_fname_pattern=config.samples_fname_pattern,
                max_abs_etv=config.max_abs_etv,
                num_primary_tasks=num_primary_tasks,
                task_types=task_types,
                task_sectors=task_sectors,
            )
        all_etv_distros[detrending] = load_etv_distros(output_fname)
    plot_etv_analysis_results(all_etv_distros, etv_comparisons, config)


if __name__ == "__main__":
    analyze_eb(parse_command_line())
