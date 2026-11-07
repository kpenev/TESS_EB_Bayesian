#!/usr/bin/env python3

"""
Corner plot of the minima found by sed_grid_search.py.

Points are colored by how much worse their SED cost is than the best. The
diagonal shows the histograms of all and of the good minima, the kernel density
estimate (KDE) of the good minima, and the slices at the KDE minima that define
the cells used to seed further fits (see sed_minima_cells.py). The cells and
their seeds are printed.
"""

from argparse import ArgumentParser

import numpy
from astropy.table import Table
from matplotlib import pyplot, colors

from sed_minima_cells import (
    find_slices,
    get_cell_seeds,
    get_kde,
    get_kde_grid,
)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("results", help="The .ecsv file of sed_grid_search.")
    parser.add_argument(
        "output", help="Output file name without extension (.pdf and .png)."
    )
    parser.add_argument("--title", default="", help="Plot title prefix.")
    parser.add_argument(
        "--max-delta-cost",
        type=float,
        default=3.0,
        help="Minima within this much of the best cost are the good ones.",
    )
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=0.1,
        help="Minimum KDE prominence of a slice, as a fraction of the KDE "
        "maximum.",
    )
    parser.add_argument(
        "--max-slice-width",
        type=float,
        default=9.0,
        help="Slices whose good minima spread more than this many kernel "
        "widths are cut again at their deepest KDE minimum (or median).",
    )
    parser.add_argument(
        "--kernel-scale",
        type=float,
        default=1.0,
        help="Factor to multiply all default KDE kernel widths by.",
    )
    parser.add_argument(
        "--show-delta-cost",
        type=float,
        default=None,
        help="Axis ranges cover minima within this much of the best cost "
        "(default: --max-delta-cost, i.e. the good minima). Minima outside "
        "the axes are still plotted.",
    )
    return parser.parse_args()


def get_quantities(table, kernel_scale=1.0, widths=None):
    """Return name -> (values, label, kernel width, sliced) for the plot."""

    if widths is None:
        # Chosen from TIC 101845679 (with max slice width 9); revisit with
        # other targets.
        widths = {
            "mtotal": 0.05,
            "mratio": 0.1 / 3,
            "meh": 0.3 / 3,
            "log_age": 0.3 / 3,
            "teff_primary": 500.0 / 3,
            "teff_secondary": 500.0 / 3,
        }
    widths = {name: kernel_scale * width for name, width in widths.items()}
    column = lambda name: numpy.asarray(table[name], dtype=float)
    return {
        "mtotal": (
            column("mtotal"),
            r"$M_{\rm tot}$ [$M_\odot$]",
            widths["mtotal"],
            True,
        ),
        "mratio": (column("mratio"), r"$q$", widths["mratio"], True),
        "meh": (column("meh"), "[M/H]", widths["meh"], True),
        "log_age": (
            numpy.log10(column("age_gyr")),
            r"$\log_{10}$ age/Gyr",
            widths["log_age"],
            True,
        ),
        "teff_primary": (
            column("teff_primary"),
            r"$T_{\rm eff,1}$ [K]",
            widths["teff_primary"],
            False,
        ),
        "teff_secondary": (
            column("teff_secondary"),
            r"$T_{\rm eff,2}$ [K]",
            widths["teff_secondary"],
            False,
        ),
    }


def main(config):
    """Make the plot and print the cells."""

    table = Table.read(config.results)
    cost = numpy.asarray(table["cost"], dtype=float)
    delta = cost - cost.min()
    good = delta < config.max_delta_cost
    if config.show_delta_cost is None:
        config.show_delta_cost = config.max_delta_cost
    shown = delta < config.show_delta_cost
    quantities = get_quantities(table, config.kernel_scale)
    names = list(quantities)

    slices = {
        name: find_slices(
            values[good], width, config.min_prominence, config.max_slice_width
        )
        for name, (values, _, width, _) in quantities.items()
    }
    limits = {}
    for name, (values, _, _, _) in quantities.items():
        low, high = values[shown].min(), values[shown].max()
        pad = 0.05 * (high - low)
        limits[name] = (low - pad, high + pad)

    order = numpy.argsort(-delta)
    norm = colors.LogNorm(vmin=0.1, vmax=1e3)
    cmap = pyplot.get_cmap("Blues_r")
    count = len(names)
    figure, axes = pyplot.subplots(count, count, figsize=(14, 14))
    for row, row_name in enumerate(names):
        for column, column_name in enumerate(names):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            x_values, x_label, x_width, x_sliced = quantities[column_name]
            if row == column:
                bins = numpy.linspace(*limits[column_name], 40)
                axis.hist(
                    x_values,
                    bins=bins,
                    color="#c6dbef",
                    label="all minima",
                )
                axis.hist(
                    x_values[good],
                    bins=bins,
                    color="#4292c6",
                    label=rf"$\Delta$cost < {config.max_delta_cost:g}",
                )
                grid = get_kde_grid(x_values[good], x_width)
                axis.plot(
                    grid,
                    get_kde(x_values[good], grid, x_width)
                    * (bins[1] - bins[0])
                    / (numpy.sqrt(2.0 * numpy.pi) * x_width),
                    color="#08306b",
                    linewidth=2,
                    label="KDE of good minima",
                )
                axis.set_yticks([])
            else:
                y_values, y_label, _, y_sliced = quantities[row_name]
                scatter = axis.scatter(
                    x_values[order],
                    y_values[order],
                    c=numpy.maximum(delta[order], 0.1),
                    cmap=cmap,
                    norm=norm,
                    s=10,
                    linewidths=0,
                )
                if y_sliced:
                    for position in slices[row_name]:
                        axis.axhline(
                            position, color="#d94801", linewidth=0.8, ls="--"
                        )
                axis.set_ylim(*limits[row_name])
                if column == 0:
                    axis.set_ylabel(y_label)
                else:
                    axis.set_yticklabels([])
            for position in slices[column_name]:
                axis.axvline(
                    position,
                    color="#d94801" if x_sliced else "#bdbdbd",
                    linewidth=0.8,
                    ls="--",
                )
            axis.set_xlim(*limits[column_name])
            if row == count - 1:
                axis.set_xlabel(x_label)
            else:
                axis.set_xticklabels([])
            axis.tick_params(labelsize=8)

    axes[0, 0].plot([], [], color="#d94801", ls="--", label="slices")
    axes[0, 0].legend(
        loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(1.05, 1.0)
    )
    colorbar_axis = figure.add_axes([0.70, 0.62, 0.02, 0.25])
    figure.colorbar(
        scatter,
        cax=colorbar_axis,
        label=r"$\Delta$cost = cost $-$ min (log scale)",
    )
    figure.suptitle(
        f"{config.title}{len(table)} SED-only minima, best cost "
        rf"{cost.min():.2f}; axes cover $\Delta$cost < "
        f"{config.show_delta_cost:g}",
        y=0.98,
    )
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.06, top=0.95, wspace=0.08, hspace=0.08
    )
    for extension in ("pdf", "png"):
        figure.savefig(f"{config.output}.{extension}", dpi=110)

    sliced = [name for name in names if quantities[name][3]]
    good_parameters = {name: quantities[name][0][good] for name in sliced}
    cell_slices, seeds = get_cell_seeds(
        good_parameters,
        cost[good],
        {name: quantities[name][2] for name in sliced},
        config.min_prominence,
        config.max_slice_width,
    )
    print(f"{good.sum()} good minima. Slices:")
    for name in sliced:
        print(f"  {name}: {numpy.round(cell_slices[name], 3)}")
    print(f"{len(seeds)} occupied cells (lowest cost first):")
    good_rows = table[good]
    for cell, best_index, num_minima in seeds:
        row = good_rows[best_index]
        print(
            f"  cell {cell}: {num_minima:4d} minima; best cost "
            f"{row['cost']:.3f}: Mtot {row['mtotal']:.3f} q {row['mratio']:.3f}"
            f" [M/H] {row['meh']:5.2f} age {row['age_gyr']:.4g} Gyr Teff "
            f"{row['teff_primary']:.0f}/{row['teff_secondary']:.0f}"
        )


if __name__ == "__main__":
    main(parse_command_line())
