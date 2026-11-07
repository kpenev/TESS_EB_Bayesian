"""
Group the minima found by the SED-only grid search into cells.

The distribution of each fitted parameter over the good minima is smoothed with
a Gaussian kernel of a fixed, physically chosen width. The parameter range is
sliced at the minima of that distribution, and the slices of all parameters
define cells. Each occupied cell provides one seed: its lowest cost minimum.
"""

import numpy
from scipy.signal import find_peaks


def get_kde(values, grid, width):
    """Return the (unnormalized) Gaussian kernel density of values on grid."""

    return numpy.exp(
        -0.5 * ((grid[:, None] - values[None, :]) / width) ** 2
    ).sum(axis=1)


def get_kde_grid(values, width, num_points=1000):
    """Return the points at which to evaluate the KDE of the given values."""

    return numpy.linspace(
        values.min() - 3.0 * width, values.max() + 3.0 * width, num_points
    )


def _split_interval(values, grid, density, max_spread):
    """
    Return cut positions that split values into pieces of limited spread.

    The interval is cut at the deepest local minimum of the KDE inside it, or
    at the median of the values if there is none (e.g. one broad peak), and
    each side is split again until its spread does not exceed ``max_spread``.
    """

    if values.size < 2 or values.max() - values.min() <= max_spread:
        return []
    inside = (grid > values.min()) & (grid < values.max())
    interior_minima = find_peaks(-density[inside])[0]
    if interior_minima.size:
        cut = grid[inside][
            interior_minima[numpy.argmin(density[inside][interior_minima])]
        ]
    else:
        cut = numpy.median(values)
    below, above = values[values <= cut], values[values > cut]
    if below.size == 0 or above.size == 0:
        return []
    return (
        _split_interval(below, grid, density, max_spread)
        + [cut]
        + _split_interval(above, grid, density, max_spread)
    )


def find_slices(values, width, min_prominence=0.1, max_slice_width=9.0):
    """
    Return the positions at which to slice the range of values.

    First the significant minima of the KDE are used. Then any slice whose
    values spread more than ``max_slice_width`` kernel widths is cut at its
    deepest KDE minimum (or median), recursively.

    Args:
        values(array):    The parameter values of the good minima.

        width(float):    The Gaussian kernel width.

        min_prominence(float):    A minimum of the KDE is kept only if it is
            lower than the smaller of the neighboring maxima by at least this
            fraction of the highest KDE value.

        max_slice_width(float or None):    The largest allowed spread of a
            slice, in kernel widths. None disables the recursive splitting.

    Returns:
        array:    The slice positions, sorted.
    """

    grid = get_kde_grid(values, width)
    density = get_kde(values, grid, width)
    minima = find_peaks(-density, prominence=min_prominence * density.max())[0]
    slices = list(grid[minima])
    if max_slice_width is not None:
        bounds = [-numpy.inf] + slices + [numpy.inf]
        for low, high in zip(bounds[:-1], bounds[1:]):
            slices.extend(
                _split_interval(
                    values[(values > low) & (values <= high)],
                    grid,
                    density,
                    max_slice_width * width,
                )
            )
    return numpy.sort(slices)


def get_cell_seeds(
    parameters, cost, widths, min_prominence=0.1, max_slice_width=9.0
):
    """
    Return the lowest cost minimum in every occupied cell.

    Args:
        parameters(dict):    Parameter name -> values for the good minima.

        cost(array):    The cost of each good minimum.

        widths(dict):    Parameter name -> kernel width.

        min_prominence(float):    See `find_slices()`.

        max_slice_width(float or None):    See `find_slices()`.

    Returns:
        (dict, list):
            The slice positions for each parameter, and for each occupied cell
            a tuple (cell index tuple, index of its lowest cost minimum,
            number of minima in the cell), lowest cost first.
    """

    slices = {
        name: find_slices(values, widths[name], min_prominence, max_slice_width)
        for name, values in parameters.items()
    }
    cell_indices = numpy.column_stack(
        [
            numpy.searchsorted(slices[name], values)
            for name, values in parameters.items()
        ]
    )
    best = {}
    counts = {}
    for point_index, cell in enumerate(map(tuple, cell_indices)):
        counts[cell] = counts.get(cell, 0) + 1
        if cell not in best or cost[point_index] < cost[best[cell]]:
            best[cell] = point_index
    seeds = sorted(
        ((cell, best[cell], counts[cell]) for cell in best),
        key=lambda seed: cost[seed[1]],
    )
    return slices, seeds


def get_cell_parameters(table):
    """Return name -> values of the parameters cells are built from."""

    return {
        "mtotal": numpy.asarray(table["mtotal"], dtype=float),
        "mratio": numpy.asarray(table["mratio"], dtype=float),
        "meh": numpy.asarray(table["meh"], dtype=float),
        "log_age": numpy.log10(numpy.asarray(table["age_gyr"], dtype=float)),
    }


# pylint: disable=too-many-arguments
def select_seeds(
    table,
    max_delta_cost=3.0,
    widths=None,
    min_prominence=0.1,
    max_slice_width=9.0,
):
    """
    Return one seed per occupied cell from the SED-only grid search results.

    Args:
        table(Table):    The results of sed_grid_search.py.

        max_delta_cost(float):    Only minima within this much of the best
            cost are used.

        widths(dict or None):    Parameter name (see `get_cell_parameters()`)
            -> kernel width. The defaults were chosen on TIC 101845679 and
            should be revisited on other targets.

        min_prominence(float):    See `find_slices()`.

        max_slice_width(float or None):    See `find_slices()`.

    Returns:
        (dict, list):
            The slice positions for each parameter, and for each occupied cell
            a tuple (cell index tuple, table row of its lowest cost minimum,
            number of minima in the cell), lowest cost first.
    """

    if widths is None:
        widths = {
            "mtotal": 0.1 / 3,
            "mratio": 0.1 / 3,
            "meh": 0.3 / 3,
            "log_age": 0.3 / 3,
        }
    cost = numpy.asarray(table["cost"], dtype=float)
    good = cost - cost.min() < max_delta_cost
    slices, seeds = get_cell_seeds(
        {
            name: values[good]
            for name, values in get_cell_parameters(table).items()
        },
        cost[good],
        widths,
        min_prominence,
        max_slice_width,
    )
    good_rows = table[good]
    return slices, [
        (cell, good_rows[best_index], count)
        for cell, best_index, count in seeds
    ]


# pylint: enable=too-many-arguments
