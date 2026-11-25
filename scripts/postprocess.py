#!/usr/bin/env python3

"""
Post-process the MCMC chain of one target (TEC-142).

The weights of walker groups that never exchange are to be fixed after
sampling. So far: the groups of walkers that never exchange (steps 1a and
1b), discarding small groups whose walkers barely move, and local chains for
the other small groups (step 3, local_chain.py).
"""

from os import path
import logging

from configargparse import ArgumentParser, DefaultsFormatter
from corner import corner
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy
import pandas
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans

from chain_analysis import get_chain_expressions
from local_chain import resolve_small_groups
from main_chain import MainChain
from paths import results_dir, samples as samples_fname
from regions import build_region_rule
from sample_params import SampleParams
from walker_distances import get_walker_distances

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
        "have ndim + 1 = 22 walkers, and smaller groups are found by step "
        "1b. Deliberately below twice --group-min-large-walkers: a gap "
        "missed here gives a wrong weight silently, while splitting into "
        "two small halves only costs local chains.",
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
        "--group-kmeans-max-iterations",
        type=int,
        default=1000,
        help="Each 2-means run must converge strictly (no label changes) "
        "within this many iterations, otherwise post-processing fails.",
    )
    parser.add_argument(
        "--group-min-large-walkers",
        type=int,
        default=32,
        help="Groups with fewer walkers ever in them than this cannot be "
        "trusted to sample their region (they need at least ndim + 1 = 22 "
        "walkers that stay, and a kept split allows up to about 10 brief "
        "visitors; TEC-146 checks the cost); they get local chains and are "
        "drawn with large, opaque markers in the corner plot.",
    )
    parser.add_argument(
        "--walker-link-factor",
        type=float,
        default=1.5,
        help="Within a coarse group, two walkers are linked if their closest "
        "approach (whitened by the group's covariance) is at most this times "
        "the larger of their spreads (median distance of a walker's "
        "positions from their mean).",
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
        "--discard-move-fraction",
        type=float,
        default=0.1,
        help="Groups with fewer than --group-min-large-walkers walkers, all "
        "moving less often than this times the median move rate of all "
        "walkers, are discarded as walkers stuck in small holes.",
    )
    parser.add_argument(
        "--local-chains",
        action="store_true",
        help="Run (or resume) the local chains of the small groups that are "
        "not discarded (step 3; needs the target's likelihood, so run from "
        "the scripts directory).",
    )
    parser.add_argument(
        "--local-chain-walkers",
        type=int,
        default=64,
        help="Walkers per local chain: a multiple of twice --num-parallel, "
        "since emcee evaluates the two halves of the ensemble in turn.",
    )
    parser.add_argument(
        "--local-chain-block",
        type=int,
        default=500,
        help="Local chains run this many steps between leak and "
        "convergence checks.",
    )
    parser.add_argument(
        "--local-chain-max-steps",
        type=int,
        default=5000,
        help="At most this many steps are added to a local chain per run; "
        "an unconverged one is resumed by the next run.",
    )
    parser.add_argument(
        "--local-chain-autocorr-factor",
        type=float,
        default=50.0,
        help="A local chain has converged when its second half is at least "
        "this many integrated autocorrelation times long.",
    )
    parser.add_argument(
        "--leak-percentile",
        type=float,
        default=99.9,
        help="A local chain has leaked into a large group if any of its "
        "positions is within this percentile of the group's own squared "
        "distances from its mean (whitened by the group's covariance).",
    )
    parser.add_argument(
        "--bridge-max-components",
        type=int,
        default=8,
        help="Reference densities for bridge sampling are Gaussian mixtures "
        "of at most this many components (step 4).",
    )
    parser.add_argument(
        "--bridge-mixture-inits",
        type=int,
        default=5,
        help="Each mixture is the best of this many initializations.",
    )
    parser.add_argument(
        "--bridge-mixture-max-iterations",
        type=int,
        default=1000,
        help="Fitting a mixture fails if it has not converged after this many "
        "iterations.",
    )
    parser.add_argument(
        "--bridge-samples-per-parameter",
        type=float,
        default=10.0,
        help="Mixtures need at least this many effective samples per "
        "parameter of each component (252 parameters each in 21-D), which "
        "caps the number of components.",
    )
    parser.add_argument(
        "--bridge-noise-factor",
        type=float,
        default=2.0,
        help="The mixture used has the fewest components whose held-out score "
        "is within this many standard errors of the best one's.",
    )
    parser.add_argument(
        "--bridge-draws",
        type=int,
        default=5000,
        help="Draws from each cross-fitted reference density (two per "
        "region), each costing one likelihood evaluation.",
    )
    parser.add_argument(
        "--local-chain-fname-pattern",
        default=path.join(
            results_dir, "local_chains", "tess{tic_id:d}_walkers{walkers}.h5"
        ),
        help="The local chain files, with ``{tic_id}`` and ``{walkers}`` "
        "(the group's walkers joined by '-') substituted.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="The number of processes evaluating the likelihood for local "
        "chains.",
    )
    parser.add_argument(
        "--worker-max-tasks",
        type=int,
        default=25600,
        help="Likelihood evaluations per worker process before it is "
        "replaced (as in mcmc_sampling.py).",
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


def bisect(  # pylint: disable=too-many-locals
    positions, selected, rng, num_restarts, max_iterations
):
    """
    Split a group of walker positions in two, with the fewest crossings.

    Each 2-means run (in coordinates standardized to the group) gives a
    trial split. The one returned has the fewest walker moves between its
    halves per position in its smaller half: normalizing keeps lopsided
    splits, which see few crossings only because they cut off a thinly
    visited edge, from being favored.

    Every run must converge strictly (an iteration that changes no label),
    so the positions are split exactly as ``KMeans.predict()`` with the
    final centers splits any other point (bridge sampling draws).

    Args:
        positions(array):    All walker positions, shape (steps, walkers,
            parameters).

        selected(array):    Which positions are in the group to split,
            shape (steps, walkers).

        rng(numpy.random.Generator):    Seeds the 2-means runs.

        num_restarts(int):    How many 2-means runs to try.

        max_iterations(int):    The iteration limit of each run. Reaching
            it raises an error: the run was still converging.

    Returns:
        dict or None:
            ``in_second``: whether each position of the group (in the order
            of ``positions[selected]``) is in the second half;
            ``projection``: those positions projected onto the line joining
            the means of the two halves (scaled so the means are at 0 and
            1); ``crossings``: the number of crossings between the halves;
            ``classifier``: the fitted ``KMeans`` (``kmeans``) and the
            ``mean`` and ``scale`` that standardize positions for it. None
            if every run left a half empty.
    """

    points = positions[selected]
    mean = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (points - mean) / scale
    halves = numpy.zeros(selected.shape, dtype=int)
    best_rate = numpy.inf
    best = None
    for _ in range(num_restarts):
        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=1,
            max_iter=max_iterations,
            tol=0.0,
            random_state=int(rng.integers(2**32)),
        )
        assignment = kmeans.fit_predict(standardized)
        if kmeans.n_iter_ >= max_iterations:
            raise RuntimeError(
                f"2-means did not converge within {max_iterations:d} "
                "iterations (--group-kmeans-max-iterations)."
            )
        sizes = numpy.bincount(assignment, minlength=2)
        if sizes.min() == 0:
            continue
        halves[selected] = assignment + 1
        crossings = count_crossings(halves, 1, 2)
        if crossings / sizes.min() < best_rate:
            best_rate = crossings / sizes.min()
            best = (kmeans, assignment, crossings)
    if best is None:
        return None
    kmeans, assignment, crossings = best
    centers = kmeans.cluster_centers_
    axis = centers[1] - centers[0]
    return {
        "in_second": assignment == 1,
        "projection": (standardized - centers[0]) @ axis / (axis @ axis),
        "crossings": crossings,
        "classifier": {"kmeans": kmeans, "mean": mean, "scale": scale},
    }


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
            its second half, its classifier (see `bisect()`), the walkers
            ever in each half, the dip, the crossings and whether the split
            was kept. Replaying the kept splits in order, each classifying
            the points currently in its group, reproduces the labels.
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
            positions,
            selected,
            rng,
            config.group_kmeans_restarts,
            config.group_kmeans_max_iterations,
        )
        if bisection is None:
            continue
        trial = labels.copy()
        trial[selected] = numpy.where(bisection["in_second"], num_groups, group)
        split = {
            "group": group,
            "new_group": num_groups,
            "classifier": bisection["classifier"],
            "walkers": tuple(
                int((trial == label).any(axis=0).sum())
                for label in (group, num_groups)
            ),
            "dip": get_dip(
                bisection["projection"], config.group_histogram_bins
            ),
            "crossings": bisection["crossings"],
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
            the walkers' spreads (see `get_walker_distances()`).
    """

    distances = get_walker_distances(
        walker_positions, rng, config.walker_max_points
    )
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
    the larger of their spreads. Walkers that never exchanged never came
    within each other's range of motion, so each linked set (single linkage)
    other than the group's largest becomes a new group, taking its walkers'
    positions in that coarse group, unless those walkers spend less than
    ``config.walker_min_stay_fraction`` of the window there: such a set is a
    brief excursion of walkers living elsewhere, and is only logged. This
    catches the one or few walkers far from everyone else that bisection
    cannot split off, walkers each stuck in its own hole (even when many of
    them make up a coarse group), and any large group bisection missed (e.g.
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
            walkers relative to the walkers' spreads.
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


def get_move_rates(backend, burn_in, chunk_size=1000):
    """
    Return the fraction of steps after the burn-in in which each walker moved.

    Every step is used (not thinned), read ``chunk_size`` steps at a time.

    Args:
        backend(HDFBackend):    The samples file's chain.

        burn_in(int):    The steps to skip.

    Returns:
        array:    The move rate of each walker.
    """

    num_steps = backend.iteration
    moves = 0
    with backend.open() as samples_file:
        chain = samples_file[backend.name]["chain"]
        previous = chain[burn_in]
        for first in range(burn_in + 1, num_steps, chunk_size):
            positions = chain[first : min(first + chunk_size, num_steps)]
            moves = moves + numpy.sum(
                numpy.any(
                    numpy.concatenate([previous[None], positions[:-1]])
                    != positions,
                    axis=-1,
                ),
                axis=0,
            )
            previous = positions[-1]
    return moves / (num_steps - burn_in - 1)


def find_discarded_groups(labels, move_rates, config):
    """
    Return the small groups whose walkers barely move.

    A walker's move rate is roughly the size of the acceptable region around
    it (along the directions it proposes in) over its distance to the
    walkers it pairs with. A walker stuck in its own small hole accepts
    almost nothing, while a lone walker in a broad region moves about as
    often as the slow walkers of a large group. Groups with fewer than
    ``config.group_min_large_walkers`` walkers, all of which move less often
    than ``config.discard_move_fraction`` times the median move rate of all
    walkers, are discarded.

    Args:
        labels(array):    The group of each position, shape (steps,
            walkers).

        move_rates(array):    The move rate of each walker (see
            `get_move_rates()`).

        config:    The command line configuration.

    Returns:
        set:    The discarded groups.
    """

    threshold = config.discard_move_fraction * numpy.median(move_rates)
    discarded = set()
    for group in range(labels.max() + 1):
        walkers = numpy.flatnonzero((labels == group).any(axis=0))
        if (
            walkers.size < config.group_min_large_walkers
            and (move_rates[walkers] < threshold).all()
        ):
            discarded.add(group)
    return discarded


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


def plot_groups(labels, plot_data, ranges, discarded, config):
    """
    Save a corner plot of the walker positions, colored by group.

    Each group gets its own color: its positions as points (at most
    ``config.corner_max_points`` of them; larger and opaque for groups with
    fewer than ``config.group_min_large_walkers`` walkers, so small groups
    stay visible) and histograms each scaled to a maximum of 1 (see
    `rescale_histograms()`); the legend gives each group's fraction of all
    positions and marks discarded groups.

    Args:
        labels(array):    The group of each position, shape (steps,
            walkers).

        plot_data(pandas.DataFrame):    The quantities to plot at each
            position, in the order of ``labels.flatten()``.

        ranges:    The range of each column (see `get_plot_ranges()`).

        discarded(set):    The discarded groups (see
            `find_discarded_groups()`).

        config:    The command line configuration (``corner_plot_fname``,
            ``corner_max_points``, ``group_min_large_walkers``, ``seed`` and
            ``tic_id``).
    """

    flat_labels = labels.flatten()
    rng = numpy.random.default_rng(config.seed)
    plot_ranges = get_plot_ranges(plot_data, ranges)
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
        # The default color cycle (wraps around).
        color = f"C{group}"
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
                f"{num_walkers} walkers"
                + (" (discarded)" if group in discarded else ""),
            )
        )
    rescale_histograms(figure, len(plot_data.columns))
    figure.legend(handles=legend_handles, loc="upper right")
    pyplot.savefig(config.corner_plot_fname.format(tic_id=config.tic_id))
    pyplot.close(figure)


def plot_main_chain(main_chain, labels, discarded, config):
    """
    Make the corner plot of the main chain's positions, colored by group.

    Args:
        main_chain(MainChain):    The main chain.

        labels(array):    The group of each position, shape (steps,
            walkers).

        discarded(set):    The discarded groups.

        config:    The command line configuration (see `plot_groups()`).
    """

    blobs = main_chain.backend.get_blobs(
        discard=config.burn_in, thin=config.thin
    )
    log_prob = main_chain.log_prob.flatten()
    plot_data = pandas.DataFrame(
        blobs.reshape(-1, blobs.shape[-1]), columns=SampleParams._fields
    )
    plot_data.insert(
        0, "logprob", log_prob - log_prob[numpy.isfinite(log_prob)].min()
    )
    plot_data, ranges = get_chain_expressions(
        plot_data, config.chain_expression
    )
    plot_groups(labels, plot_data, ranges, discarded, config)


def main(config, main_chain=None):
    """
    Find and report the coarse groups of one target's walkers.

    Args:
        config:    The command line configuration.

        main_chain(MainChain or None):    The main chain to post-process
            (e.g. a toy posterior's, a subclass); None reads the target's
            samples file.

    Returns:
        RegionRule or None:    The rule assigning points to bridge-sampling
            regions, or None without ``config.local_chains`` (small groups
            are then unresolved).
    """

    if main_chain is None:
        main_chain = MainChain(config)
    positions = main_chain.positions
    _logger.info(
        "TIC %d: %d steps after burn-in %d (every %d), %d walkers.",
        config.tic_id,
        positions.shape[0],
        config.burn_in,
        config.thin,
        positions.shape[1],
    )
    coarse_labels, splits = find_coarse_groups(positions, config)
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
    labels, isolated = find_isolated_walkers(positions, coarse_labels, config)
    for group in isolated:
        _logger.info(
            "Isolated walkers %s of group %d -> group %d (closest approach "
            "to the group's other walkers %.1f times their spreads).",
            group["walkers"],
            group["coarse_group"],
            group["group"],
            group["separation"],
        )
    move_rates = get_move_rates(main_chain.backend, config.burn_in)
    discarded = find_discarded_groups(labels, move_rates, config)
    for group in range(labels.max() + 1):
        here = labels == group
        _logger.info(
            "Group %d: %.3f of positions, %d walkers ever, %d always, "
            "median move rate %.3f of all walkers' median%s.",
            group,
            here.mean(),
            here.any(axis=0).sum(),
            here.all(axis=0).sum(),
            numpy.median(move_rates[here.any(axis=0)])
            / numpy.median(move_rates),
            ": discarded" if group in discarded else "",
        )
    verdicts, small_samples = {}, {}
    if config.local_chains:
        labels, verdicts, small_samples = resolve_small_groups(
            main_chain, labels, discarded, config
        )
    if config.corner_plot_fname is not None:
        plot_main_chain(main_chain, labels, discarded, config)
    if not config.local_chains:
        return None
    return build_region_rule(
        main_chain,
        {
            "splits": splits,
            "coarse_labels": coarse_labels,
            "labels": labels,
            "discarded": discarded,
        },
        verdicts,
        small_samples,
        config,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # It logs every step it saves.
    logging.getLogger("hacked_emcee_hdf5_backend").setLevel(logging.WARNING)
    main(parse_command_line())
