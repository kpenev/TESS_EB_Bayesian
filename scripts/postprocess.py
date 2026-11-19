#!/usr/bin/env python3

"""
Post-process the MCMC chain of one target (TEC-142).

The weights of walker groups that never exchange are to be fixed after
sampling. So far: the coarse groups of the walker positions (step 1a), found
by recursive bisection.
"""

import logging

from configargparse import ArgumentParser, DefaultsFormatter
from corner import corner
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy
import pandas
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from chain_analysis import get_chain_expressions
from hacked_emcee_hdf5_backend import HDFBackend
from paths import samples as samples_fname
from sample_params import SampleParams

_logger = logging.getLogger(__name__)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["postprocess.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument("tic_id", type=int, help="The TIC ID to process.")
    parser.add_argument(
        "--samples-fname-pattern",
        default=samples_fname,
        help="The samples file, with ``{tic_id}`` substituted.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        required=True,
        help="The release burn-in cut: steps before it are not used.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=10,
        help="Use every this many steps after the burn-in. Moves between "
        "groups that return within this many steps are not counted, which "
        "only makes splits more likely (the safe direction).",
    )
    parser.add_argument(
        "--group-max-dip",
        type=float,
        default=0.2,
        help="A split is kept only if the density between the two halves, "
        "along the line joining their means, drops below this fraction of "
        "the lower of the two peaks.",
    )
    parser.add_argument(
        "--group-min-crossings",
        type=int,
        default=20,
        help="A split is kept only if walkers moved between the two halves "
        "fewer times than this (N_min of TEC-142).",
    )
    parser.add_argument(
        "--group-min-split-walkers",
        type=int,
        default=44,
        help="Groups with fewer walkers are not bisected: neither half could "
        "have the ndim + 1 = 22 walkers of a large group, and small groups "
        "are found by step 1b.",
    )
    parser.add_argument(
        "--group-histogram-bins",
        type=int,
        default=50,
        help="The number of histogram bins for the density dip, spanning "
        "from half the distance between the two means before the first to "
        "half of it after the second.",
    )
    parser.add_argument(
        "--group-kmeans-restarts",
        type=int,
        default=50,
        help="Each bisection tries this many 2-means runs and keeps the one "
        "with the fewest crossings per position of its smaller half. About "
        "0.35 s per run per 400 000 positions.",
    )
    parser.add_argument(
        "--group-min-large-walkers",
        type=int,
        default=22,
        help="Groups with fewer walkers than this (ndim + 1 by default) "
        "cannot sample their region; they are drawn with large, opaque "
        "markers in the corner plot.",
    )
    parser.add_argument(
        "--walker-link-factor",
        type=float,
        default=2.5,
        help="Within a coarse group, two walkers are linked if their closest "
        "approach (whitened by the group's covariance) is at most this times "
        "the group's median closest approach of a walker to its nearest "
        "other walker.",
    )
    parser.add_argument(
        "--walker-min-stay-fraction",
        type=float,
        default=0.5,
        help="An isolated set of walkers becomes a group only if they spend "
        "at least this fraction of the window there; otherwise it is logged "
        "as an excursion.",
    )
    parser.add_argument(
        "--walker-max-points",
        type=int,
        default=200,
        help="Each walker is represented by at most this many of its "
        "distinct positions (chosen at random) when finding closest "
        "approaches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the random number generator used by 2-means and by "
        "the choice of points to plot.",
    )
    parser.add_argument(
        "--corner-plot-fname",
        default=None,
        help="If given, a corner plot of the positions colored by group is "
        "saved here (``{tic_id}`` is substituted).",
    )
    parser.add_argument(
        "--chain-expression",
        default=[],
        action="append",
        help="A quantity to include in the corner plot, as "
        "<NAME>=<EXPRESSION>[:<MIN>:<MAX>] (as for visualize.py). Without "
        "any, all blob entries and logprob are shown.",
    )
    parser.add_argument(
        "--corner-max-points",
        type=int,
        default=20000,
        help="Draw at most this many points of each group in the corner "
        "plot (chosen at random), so small groups stay visible.",
    )
    return parser.parse_args()


def bisect(positions, selected, rng, num_restarts):
    """
    Split a group of walker positions in two, with the fewest crossings.

    Each 2-means run (in coordinates standardized to the group) gives a
    trial split. The one returned has the fewest walker moves between its
    halves per position in its smaller half: normalizing keeps lopsided
    splits, which see few crossings only because they cut off a thinly
    visited edge, from being favored.

    Args:
        positions(array):    All walker positions, shape (steps, walkers,
            parameters).

        selected(array):    Which positions are in the group to split,
            shape (steps, walkers).

        rng(numpy.random.Generator):    Seeds the 2-means runs.

        num_restarts(int):    How many 2-means runs to try.

    Returns:
        (array, array, int) or None:
            Whether each position of the group (in the order of
            ``positions[selected]``) is in the second half, those positions
            projected onto the line joining the means of the two halves
            (scaled so the means are at 0 and 1), and the number of
            crossings between the halves. None if every run left a half
            empty.
    """

    points = positions[selected]
    scale = points.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (points - points.mean(axis=0)) / scale
    halves = numpy.zeros(selected.shape, dtype=int)
    best_rate = numpy.inf
    best = None
    for _ in range(num_restarts):
        means, assignment = kmeans2(standardized, 2, minit="++", rng=rng)
        sizes = numpy.bincount(assignment, minlength=2)
        if sizes.min() == 0:
            continue
        halves[selected] = assignment + 1
        crossings = count_crossings(halves, 1, 2)
        if crossings / sizes.min() < best_rate:
            best_rate = crossings / sizes.min()
            best = (means, assignment, crossings)
    if best is None:
        return None
    means, assignment, crossings = best
    axis = means[1] - means[0]
    return (
        assignment == 1,
        (standardized - means[0]) @ axis / (axis @ axis),
        crossings,
    )


def get_dip(projection, num_bins):
    """
    Return how deep the density dips between the two halves of a bisection.

    Args:
        projection(array):    The points projected onto the line joining the
            means of the two halves, scaled so the means are at 0 and 1 (see
            `bisect()`).

        num_bins(int):    The number of histogram bins between -0.5 and 1.5.

    Returns:
        float:    The lowest (3-bin smoothed) histogram count between the
            peaks of the two halves, divided by the lower peak: near 0 for a
            gap, near 1 for none.
    """

    counts, edges = numpy.histogram(
        projection, bins=num_bins, range=(-0.5, 1.5)
    )
    smoothed = numpy.convolve(counts, numpy.ones(3) / 3, mode="same")
    in_first = 0.5 * (edges[1:] + edges[:-1]) < 0.5
    first_peak = numpy.argmax(numpy.where(in_first, smoothed, -1.0))
    second_peak = numpy.argmax(numpy.where(in_first, -1.0, smoothed))
    lowest = smoothed[first_peak : second_peak + 1].min()
    return lowest / min(smoothed[first_peak], smoothed[second_peak])


def count_crossings(labels, first, second):
    """Return how many times walkers moved between two groups."""

    before, after = labels[:-1], labels[1:]
    return int(
        numpy.sum((before == first) & (after == second))
        + numpy.sum((before == second) & (after == first))
    )


def find_coarse_groups(positions, config):
    """
    Return the coarse group of every walker position (step 1a of TEC-142).

    Groups are bisected recursively by 2-means. A split is kept only if the
    density dips between the two halves (`get_dip()` below
    ``config.group_max_dip``) and walkers moved between them fewer than
    ``config.group_min_crossings`` times. A group whose split is rejected is
    not tried again, so a group hiding behind a rejected split (e.g. the
    middle one of three in a row, if every 2-means run cut through it) is
    missed here; `find_isolated_walkers()` catches it.

    Args:
        positions(array):    The walker positions after the burn-in, shape
            (steps, walkers, parameters).

        config:    The command line configuration (the ``group_*`` options
            and ``seed``).

    Returns:
        (array, [dict]):
            The group of each position, shape (steps, walkers), and a
            record of each split tried: the group split, the new group of
            its second half, the walkers ever in each half, the dip, the
            crossings and whether the split was kept.
    """

    rng = numpy.random.default_rng(config.seed)
    labels = numpy.zeros(positions.shape[:2], dtype=int)
    num_groups = 1
    pending = [0]
    splits = []
    while pending:
        group = pending.pop()
        selected = labels == group
        if selected.any(axis=0).sum() < config.group_min_split_walkers:
            continue
        bisection = bisect(
            positions, selected, rng, config.group_kmeans_restarts
        )
        if bisection is None:
            continue
        in_second, projection, crossings = bisection
        trial = labels.copy()
        trial[selected] = numpy.where(in_second, num_groups, group)
        split = {
            "group": group,
            "new_group": num_groups,
            "walkers": tuple(
                int((trial == label).any(axis=0).sum())
                for label in (group, num_groups)
            ),
            "dip": get_dip(projection, config.group_histogram_bins),
            "crossings": crossings,
        }
        split["kept"] = (
            split["dip"] < config.group_max_dip
            and split["crossings"] < config.group_min_crossings
        )
        splits.append(split)
        if split["kept"]:
            labels = trial
            pending.extend((group, num_groups))
            num_groups += 1
    return labels, splits


def whiten(points, eigenvalue_floor=1e-6):
    """
    Return points in coordinates whitened by their own covariance.

    Each parameter is first standardized (so the parameters' very different
    scales do not matter), then the correlation matrix is whitened. Its
    eigenvalues are floored at ``eigenvalue_floor``, so constant or exactly
    degenerate parameters do not break the transformation; thin directions
    above the floor count fully, as they should (they are thin because the
    data constrain them).

    Args:
        points(array):    The positions, one per row.

    Returns:
        callable:    Transforms positions (one per row) to the whitened
            coordinates.
    """

    mean = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale == 0] = 1.0
    eigenvalues, eigenvectors = numpy.linalg.eigh(
        numpy.atleast_2d(numpy.cov(((points - mean) / scale).T))
    )
    transform = eigenvectors / numpy.sqrt(
        numpy.maximum(eigenvalues, eigenvalue_floor)
    )
    return lambda values: ((values - mean) / scale) @ transform


def get_walker_distances(walker_positions, rng, max_points):
    """
    Return the closest approach between every pair of walkers.

    Args:
        walker_positions([array]):    The positions of each walker, one per
            row.

        rng(numpy.random.Generator):    Chooses the positions used.

        max_points(int):    Each walker is represented by at most this many
            of its distinct positions, chosen at random.

    Returns:
        array:    The smallest distance between positions of each pair of
            walkers (infinite on the diagonal), shape (walkers, walkers), in
            coordinates whitened by the covariance of all the given positions
            (see `whiten()`).
    """

    transform = whiten(numpy.concatenate(walker_positions))
    subsets = []
    for points in walker_positions:
        points = numpy.unique(points, axis=0)
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        subsets.append(transform(points))
    starts = numpy.cumsum([0] + [len(points) for points in subsets[:-1]])
    pooled = numpy.concatenate(subsets)
    result = numpy.empty((len(subsets), len(subsets)))
    for walker, points in enumerate(subsets):
        result[walker] = numpy.minimum.reduceat(
            cdist(points, pooled).min(axis=0), starts
        )
    result = numpy.minimum(result, result.T)
    numpy.fill_diagonal(result, numpy.inf)
    return result


def find_isolated_sets(walker_positions, coarse_group, rng, config):
    """
    Return the linked sets among the walkers of one coarse group.

    See `find_isolated_walkers()` for how walkers are linked.

    Args:
        walker_positions([array]):    The positions in the coarse group of
            each of its walkers, one per row.

        coarse_group(int):    The coarse group (for the log).

        rng(numpy.random.Generator):    Chooses the positions used.

        config:    The command line configuration.

    Returns:
        [(array, float)]:
            For each set other than the largest: which of the walkers are in
            it, and their closest approach to the other walkers relative to
            the group's median.
    """

    distances = get_walker_distances(
        walker_positions, rng, config.walker_max_points
    )
    distances /= numpy.median(distances.min(axis=1))
    walker_set = connected_components(
        distances <= config.walker_link_factor, directed=False
    )[1]
    set_sizes = numpy.bincount(walker_set)
    if (set_sizes >= config.group_min_large_walkers).sum() > 1:
        _logger.info(
            "Coarse group %d splits into linked sets of %s walkers, "
            "including several large ones that bisection missed.",
            coarse_group,
            set_sizes.tolist(),
        )
    return [
        (
            walker_set == set_index,
            distances[
                numpy.ix_(walker_set == set_index, walker_set != set_index)
            ].min(),
        )
        for set_index in range(set_sizes.size)
        if set_index != numpy.argmax(set_sizes)
    ]


def find_isolated_walkers(positions, labels, config):
    """
    Give each set of walkers isolated within its coarse group its own group.

    Step 1b: within each coarse group (`find_coarse_groups()`), its walkers'
    positions in it are whitened by the group's own covariance, and two
    walkers are linked if their closest approach (see
    `get_walker_distances()`) is at most ``config.walker_link_factor`` times
    the group's median closest approach of a walker to its nearest other
    walker. Walkers that never exchanged never came close, so each linked
    set (single linkage) other than the group's largest becomes a new group,
    taking its walkers' positions in that coarse group, unless those walkers
    spend less than ``config.walker_min_stay_fraction`` of the window there:
    such a set is a brief excursion of walkers living elsewhere, and is only
    logged. This catches the one or few walkers far from everyone else that
    bisection cannot split off, and any large group bisection missed (e.g.
    the middle one of three in a row when every 2-means run cut through
    it).

    Args:
        positions(array):    The walker positions, shape (steps, walkers,
            parameters).

        labels(array):    The coarse group of each position, shape (steps,
            walkers).

        config:    The command line configuration (``walker_link_factor``,
            ``walker_max_points``, ``walker_min_stay_fraction``,
            ``group_min_large_walkers`` and ``seed``).

    Returns:
        (array, [dict]):
            The group of each position with the new groups added, and for
            each new group: its index, the coarse group it came from, its
            walkers and their closest approach to the coarse group's other
            walkers relative to the group's median.
    """

    rng = numpy.random.default_rng(config.seed)
    new_labels = labels.copy()
    num_groups = labels.max() + 1
    isolated = []
    for coarse_group in range(labels.max() + 1):
        in_group = labels == coarse_group
        walkers = numpy.flatnonzero(in_group.any(axis=0))
        if walkers.size < 2:
            continue
        for members, separation in find_isolated_sets(
            [positions[in_group[:, walker], walker] for walker in walkers],
            coarse_group,
            rng,
            config,
        ):
            members = walkers[members]
            stay = in_group[:, members].mean()
            if stay < config.walker_min_stay_fraction:
                _logger.info(
                    "Walkers %s only passed through an isolated part of "
                    "group %d (%.3f of their positions): an excursion, not "
                    "a group.",
                    members.tolist(),
                    coarse_group,
                    stay,
                )
                continue
            new_labels[:, members] = numpy.where(
                in_group[:, members], num_groups, new_labels[:, members]
            )
            isolated.append(
                {
                    "group": int(num_groups),
                    "coarse_group": coarse_group,
                    "walkers": members.tolist(),
                    "separation": separation,
                }
            )
            num_groups += 1
    return new_labels, isolated


def get_plot_ranges(plot_data, ranges):
    """
    Return explicit plot ranges, shared by all groups.

    Args:
        plot_data(pandas.DataFrame):    The quantities to plot (one column
            each) at every position.

        ranges([float or (float, float)]):    The range of each column, as
            `chain_analysis.get_chain_expressions()` returns them: (min, max)
            or the fraction of the values to include (centered quantiles).

    Returns:
        [(float, float)]:    The (min, max) of each column.
    """

    result = []
    for column, column_range in zip(plot_data.columns, ranges):
        if numpy.ndim(column_range) == 0:
            values = plot_data[column].to_numpy()
            low, high = numpy.quantile(
                values[numpy.isfinite(values)],
                [0.5 - 0.5 * column_range, 0.5 + 0.5 * column_range],
            )
        else:
            low, high = column_range
        if low == high:
            low, high = low - 0.5, high + 0.5
        result.append((low, high))
    return result


def rescale_histograms(figure, num_columns):
    """
    Scale every group's histogram in a corner plot to a maximum of 1.

    So small groups stay visible next to large ones (their fractions are in
    the legend). This also undoes corner fixing the histograms' y ranges to
    the group drawn last.

    Args:
        figure:    The corner plot.

        num_columns(int):    The number of quantities plotted.
    """

    for axis in numpy.diag(
        numpy.array(figure.axes).reshape(num_columns, num_columns)
    ):
        for patch in axis.patches:
            vertices = patch.get_xy()
            if vertices[:, 1].max() > 0:
                vertices[:, 1] /= vertices[:, 1].max()
                patch.set_xy(vertices)
        axis.set_ylim(0, 1.1)


def plot_groups(labels, plot_data, ranges, config):
    """
    Save a corner plot of the walker positions, colored by group.

    Each group gets its own color: its positions as points (at most
    ``config.corner_max_points`` of them; larger and opaque for groups with
    fewer than ``config.group_min_large_walkers`` walkers, so small groups
    stay visible) and histograms each scaled to a maximum of 1 (see
    `rescale_histograms()`); the legend gives each group's fraction of all
    positions.

    Args:
        labels(array):    The group of each position, shape (steps,
            walkers).

        plot_data(pandas.DataFrame):    The quantities to plot at each
            position, in the order of ``labels.flatten()``.

        ranges:    The range of each column (see `get_plot_ranges()`).

        config:    The command line configuration (``corner_plot_fname``,
            ``corner_max_points``, ``group_min_large_walkers``, ``seed`` and
            ``tic_id``).
    """

    flat_labels = labels.flatten()
    rng = numpy.random.default_rng(config.seed)
    plot_ranges = get_plot_ranges(plot_data, ranges)
    colors = pyplot.rcParams["axes.prop_cycle"].by_key()["color"]
    figure = None
    legend_handles = []
    for group in range(flat_labels.max() + 1):
        indices = numpy.flatnonzero(flat_labels == group)
        fraction = indices.size / flat_labels.size
        num_walkers = (labels == group).any(axis=0).sum()
        if indices.size > config.corner_max_points:
            indices = rng.choice(
                indices, config.corner_max_points, replace=False
            )
        # corner refuses fewer samples than dimensions; repeated positions
        # (weighted down accordingly) leave the plot unchanged.
        if indices.size <= len(plot_data.columns):
            indices = numpy.tile(
                indices, len(plot_data.columns) // indices.size + 1
            )
        color = colors[group % len(colors)]
        figure = corner(
            plot_data.iloc[indices].to_numpy(),
            fig=figure,
            range=plot_ranges,
            weights=numpy.full(indices.size, fraction / indices.size),
            color=color,
            labels=list(plot_data.columns),
            plot_density=False,
            plot_contours=False,
            data_kwargs=(
                {"alpha": 1.0, "ms": 3.0}
                if num_walkers < config.group_min_large_walkers
                else {"alpha": 0.2, "ms": 1.0}
            ),
            hist_kwargs={"color": color},
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                label=f"group {group}: {fraction:.3f} of positions, "
                f"{num_walkers} walkers",
            )
        )
    rescale_histograms(figure, len(plot_data.columns))
    figure.legend(handles=legend_handles, loc="upper right")
    pyplot.savefig(config.corner_plot_fname.format(tic_id=config.tic_id))
    pyplot.close(figure)


def main(config):
    """Find and report the coarse groups of one target's walkers."""

    backend = HDFBackend(
        config.samples_fname_pattern.format(tic_id=config.tic_id),
        read_only=True,
    )
    positions = backend.get_chain(discard=config.burn_in, thin=config.thin)
    _logger.info(
        "TIC %d: %d steps after burn-in %d (every %d), %d walkers.",
        config.tic_id,
        positions.shape[0],
        config.burn_in,
        config.thin,
        positions.shape[1],
    )
    labels, splits = find_coarse_groups(positions, config)
    for split in splits:
        _logger.info(
            "Split of group %d (%d | %d walkers ever): dip %.2f, %d "
            "crossings: %s.",
            split["group"],
            *split["walkers"],
            split["dip"],
            split["crossings"],
            (
                f"kept as {split['group']} | {split['new_group']}"
                if split["kept"]
                else "rejected"
            ),
        )
    labels, isolated = find_isolated_walkers(positions, labels, config)
    for group in isolated:
        _logger.info(
            "Isolated walkers %s of group %d -> group %d (closest approach "
            "to the group's other walkers %.1f times the group's typical).",
            group["walkers"],
            group["coarse_group"],
            group["group"],
            group["separation"],
        )
    for group in range(labels.max() + 1):
        here = labels == group
        _logger.info(
            "Group %d: %.3f of positions, %d walkers ever, %d always.",
            group,
            here.mean(),
            here.any(axis=0).sum(),
            here.all(axis=0).sum(),
        )
    if config.corner_plot_fname is None:
        return
    blobs = backend.get_blobs(discard=config.burn_in, thin=config.thin)
    log_prob = backend.get_log_prob(
        discard=config.burn_in, thin=config.thin
    ).flatten()
    plot_data = pandas.DataFrame(
        blobs.reshape(-1, blobs.shape[-1]), columns=SampleParams._fields
    )
    plot_data.insert(
        0, "logprob", log_prob - log_prob[numpy.isfinite(log_prob)].min()
    )
    plot_data, ranges = get_chain_expressions(
        plot_data, config.chain_expression
    )
    plot_groups(labels, plot_data, ranges, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
