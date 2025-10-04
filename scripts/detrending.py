"""Function implementing LC detrending."""

from argparse import Namespace
import logging

import numpy
from numpy.polynomial import Polynomial
from numpy.lib.recfunctions import append_fields
from autowisp.iterative_rejection_util import (
    iterative_rej_smoothing_spline,
    iterative_rej_polynomial_fit,
)
from matplotlib import pyplot

from tess_target import get_bls_eclipse_mask

_logger = logging.getLogger(__name__)


def get_moving_median(lightcurve, half_porb=None, mask=None, min_points=20):
    """Each point is divided by the median of all points within +-Porb/2."""

    if half_porb is None:
        return numpy.full(
            lightcurve.shape, numpy.nanmedian(lightcurve["flux"]), dtype=float
        )
    print(
        f"Detrending time range: {lightcurve['time'][0]}, "
        f"{lightcurve['time'][-1]}"
    )
    result = numpy.empty(lightcurve.shape)
    if mask is None:
        mask = numpy.ones(lightcurve.size, dtype=bool)
    masked_time = lightcurve["time"][mask]
    masked_flux = lightcurve["flux"][mask]
    for i, time in enumerate(lightcurve["time"]):
        min_t = time - half_porb
        max_t = time + half_porb
        if min_t < lightcurve["time"][0]:
            left = 0
            right = numpy.searchsorted(
                masked_time, lightcurve["time"][0] + 2 * half_porb
            )
        else:
            if max_t > lightcurve["time"][-1]:
                right = masked_time.size
                left = numpy.searchsorted(
                    masked_time, lightcurve["time"][-1] - 2 * half_porb
                )
            else:
                left, right = numpy.searchsorted(masked_time, [min_t, max_t])
        window_flux = masked_flux[left:right]
        if window_flux.size < min_points:
            return numpy.nan
        result[i] = numpy.median(window_flux)
    return result


def get_ooe_spline_nodes(masked_time, half_porb):
    """Return the nodes to use for the out-of-eclipse smoothing spline."""

    timespan = masked_time[-1] - masked_time[0]
    node_times = numpy.linspace(
        masked_time[0],
        masked_time[-1],
        2 * int(numpy.ceil(timespan / half_porb)) + 6,
    )
    _logger.debug(
        "Guess node times (timespan=%s, P/2=%s): %s",
        repr(timespan),
        repr(half_porb),
        repr(node_times),
    )
    found_nodes = False
    while not found_nodes:
        found_nodes = True
        for t0, t1 in zip(node_times[:-1], node_times[1:]):
            if numpy.logical_and(masked_time > t0, masked_time < t1).sum() < 10:
                node_times = numpy.linspace(
                    masked_time[0],
                    masked_time[-1],
                    node_times.size - 1,
                )
                _logger.debug(
                    "Only %d between %s < t < %s, new suggested node times: %s",
                    numpy.logical_and(masked_time > t0, masked_time < t1).sum(),
                    repr(t0),
                    repr(t1),
                    repr(node_times),
                )
                found_nodes = False
                break
    return node_times


def ooe_ends_to_discard(mask, min_tail_points=10):
    """Check if the ends of an LC segment are sufficient far from eclipses."""

    if mask[:min_tail_points].all():
        left = 0
    else:
        left = numpy.nonzero(mask[min_tail_points:])[0][0] + min_tail_points

    if mask[-min_tail_points:].all():
        right = mask.size
    else:
        right = numpy.nonzero(mask[:-min_tail_points])[0][-1]

    return left, right


# Hiding argument names or putting in an object is worse for readability
# pylint: disable=too-many-arguments
def get_ooe_variability(
    lightcurve,
    half_porb=numpy.inf,
    mask=None,
    *,
    spline_rejection=(5.0, 3.0),
    eclipse_rejection=2.0,
    return_mask=False,
):
    """Remove the out-of-eclipse variability from the lightcurve."""

    _logger.debug(
        "Extracting OOE variability with mask %s and P/2 %s",
        repr(mask),
        repr(half_porb),
    )
    if mask is None:
        mask = numpy.ones(lightcurve.size, dtype=bool)

    masked_time = lightcurve["time"][mask]
    masked_flux = lightcurve["flux"][mask]

    if numpy.isfinite(half_porb) and masked_time[-1] - masked_time[0] < min(
        half_porb, 5
    ):
        _logger.warning(
            "Time span of LC portion too small (%s). Discarding",
            repr(masked_time[-1] - masked_time[0]),
        )
        return numpy.full(masked_time.shape, numpy.nan)

    ooe_mask = numpy.ones(masked_time.size, dtype=bool)
    while True:
        spline_nodes = get_ooe_spline_nodes(masked_time, half_porb)[1:-1]
        if spline_nodes.size > 3:
            _logger.debug(
                "Using %d nodes spline detrending for %s < t < %s",
                spline_nodes.size,
                lightcurve["time"][0],
                lightcurve["time"][-1],
            )

            ooe_model = iterative_rej_smoothing_spline(
                masked_time[ooe_mask],
                masked_flux[ooe_mask],
                spline_rejection,
                t=spline_nodes,
            )(lightcurve["time"])
        else:
            _logger.debug(
                "Not enough nodes, using cubic polynomial detrending for "
                "%s < t < %s",
                lightcurve["time"][0],
                lightcurve["time"][-1],
            )
            poly_coef = iterative_rej_polynomial_fit(
                masked_time[ooe_mask],
                masked_flux[ooe_mask],
                order=3,
                outlier_threshold=spline_rejection,
            )[0]
            _logger.debug("Polynomial coefficients: %s", repr(poly_coef))
            ooe_model = Polynomial(poly_coef)(lightcurve["time"])

        residuals = masked_flux - ooe_model[mask]
        new_ooe_mask = masked_flux > (
            ooe_model[mask]
            - eclipse_rejection * numpy.sqrt(numpy.mean(residuals**2))
        )
        if not ooe_mask[numpy.logical_not(new_ooe_mask)].any():
            break
        ooe_mask = numpy.logical_and(ooe_mask, new_ooe_mask)
        _logger.debug("Re-fitting trend on %d OOE points.", ooe_mask.sum())

    try:
        pyplot.plot(lightcurve["time"], lightcurve["flux"], ".k")
        pyplot.plot(masked_time, masked_flux, ".r")
        pyplot.plot(masked_time[ooe_mask], masked_flux[ooe_mask], ".g")
        pyplot.plot(lightcurve["time"], ooe_model, ".b")
        pyplot.title(
            ("Spline" if spline_nodes.size > 3 else "Polynomial")
            + " detrending"
        )
        pyplot.show()
    except:  # pylint: disable=bare-except
        pass

    discard_left, discard_right = ooe_ends_to_discard(mask)
    ooe_model[:discard_left] = numpy.nan
    ooe_model[discard_right:] = numpy.nan
    _logger.debug(
        "Discarding %s points (%s <= t < %s) at the left end and %s at right "
        "end (%s < t <= %s)",
        discard_left,
        lightcurve["time"][0],
        lightcurve["time"][discard_left],
        lightcurve.size - discard_right,
        lightcurve["time"][discard_right - 1],
        lightcurve["time"][-1],
    )

    if return_mask:
        print(f"OOE model size: {ooe_model.size}, mask size: {mask.size}")
        result_mask = numpy.copy(mask)
        result_mask[mask] = ooe_mask
        return ooe_model, result_mask
    return ooe_model


# pylint: enable=too-many-arguments


def detrend_with_gaps(
    lightcurve, min_gap, get_trend, full_output=False, **kwargs
):
    """
    Detrend each lightcurve segment between gaps.

    Args:
        lightcurve: The lightcurve to detrend.

        min_gap: The smallest time without observations to count as a gap.

        get_trend: Function to calculate the trend.

        full_output: If true, the returned lightcurve will also include fields
            ``"original"`` (the original, un-detrended lightcurve) and
            ``"trend"`` (the trend that was removed).

        kwargs: Additional arguments for get_trend. If any are arrays matching
            the length of the lightcurve, they will also be split at the LC gaps
            before passing to ``get_trend``.

    Returns:
        The detrended lightcurve.
    """

    print("Detrending with gaps" + ", saving detrending" if full_output else "")
    gap_indices = (
        numpy.nonzero(
            lightcurve["time"][1:] - lightcurve["time"][:-1] > min_gap
        )[0]
        + 1
    )
    gap_indices = numpy.append(gap_indices, lightcurve["time"].size)

    fixed_kwargs = {}
    segment_kwargs = {}
    for key, value in kwargs.items():
        try:
            if len(value) == lightcurve.size:
                segment_kwargs[key] = value
                continue
        except TypeError:
            pass
        fixed_kwargs[key] = value

    detrended = numpy.copy(lightcurve)
    if full_output:
        detrended = append_fields(
            detrended,
            ["original", "trend"],
            [
                lightcurve["flux"],
                numpy.full(detrended.size, numpy.nan, dtype=float),
            ],
            usemask=False,
        )
    start_index = 0
    good_mask = numpy.ones(detrended.size, dtype=bool)
    if fixed_kwargs.get("return_mask", False):
        eclipse_mask = numpy.zeros(detrended.size, dtype=bool)
    for end_index in gap_indices:
        if end_index - start_index < 20:
            good_mask[start_index:end_index] = False
            continue
        print("Getting trend.")
        scaling = get_trend(
            lightcurve[start_index:end_index],
            **fixed_kwargs,
            **{
                key: value[start_index:end_index]
                for key, value in segment_kwargs.items()
            },
        )
        if fixed_kwargs.get("return_mask", False):
            eclipse_mask[start_index:end_index] = scaling[1]
            scaling = scaling[0]

        good_mask[start_index:end_index] = numpy.isfinite(scaling)
        print(
            f"Scaling ({lightcurve['time'][start_index]} < t < "
            f"{lightcurve['time'][end_index-1]}): {scaling!r}"
        )
        detrended[start_index:end_index]["flux"] /= scaling
        detrended[start_index:end_index]["flux_err"] /= scaling

        if full_output:
            detrended[start_index:end_index]["trend"] = scaling
        start_index = end_index

    detrended = detrended[good_mask]
    if fixed_kwargs.get("return_mask", False):
        return detrended, eclipse_mask[good_mask]
    return detrended


def masked_detrend(
    lightcurve, exptime, log_likelihood, get_trend, full_output=False, **kwargs
):
    """Return the given lightcurve detrended after masking eclipses."""

    half_porb = log_likelihood.best_fit_bls["period"][0]
    if log_likelihood.masked_is_significant():
        mask = numpy.logical_not(
            numpy.logical_or(
                get_bls_eclipse_mask(
                    log_likelihood.best_fit_bls, lightcurve, "both"
                ),
                get_bls_eclipse_mask(
                    log_likelihood.best_fit_bls, lightcurve, "masked"
                ),
            )
        )
        half_porb /= 2
    else:
        mask = numpy.logical_not(
            get_bls_eclipse_mask(
                log_likelihood.best_fit_bls, lightcurve, "both"
            )
        )

    return detrend_with_gaps(
        lightcurve,
        max(2 * log_likelihood.best_fit_bls["duration"], 30 * exptime),
        get_trend,
        mask=mask,
        half_porb=half_porb,
        full_output=full_output,
        **kwargs,
    )


def test():
    """Avoid polluting global namespace."""

    # That is the idea
    # pylint: disable=import-outside-toplevel
    from light_curve_plotter import LightCurvePlotter

    # pylint: enable=import-outside-toplevel

    for tic in [101462]:  # [1045298, 1220444, 2020964, 22766107]:
        try:
            config = Namespace(
                tic_id=tic,
                plot_lightcurve=(
                    f"tess{tic}_detrending.pdf",
                    "[[full, full, full],"
                    " [folded, folded, folded],"
                    " [zoom_even, zoom_odd, zoom_masked]]",
                ),
                data_on_top=False,
                show_lc_detrending=True,
                highlight_first_model=False,
            )
            print(f"Plotting with configuration:\n {config}")
            LightCurvePlotter(config)(tic, get_trend=get_ooe_variability)
        except RuntimeError:
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test()
