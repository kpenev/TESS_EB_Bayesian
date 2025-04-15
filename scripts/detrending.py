"""Function implementing LC detrending."""

from collections import namedtuple
from functools import partial
import logging

import numpy

from tess_target import get_bls_eclipse_mask

def calc_moving_median(lightcurve, half_porb, mask, min_points=20):
    """Each point is divided by the median of all points within +-Porb/2."""

    print(
        f"Detrending time range: {lightcurve['time'][0]}, "
        f"{lightcurve['time'][-1]}"
    )
    result = numpy.empty(lightcurve.shape)
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


def masked_detrend(lightcurve, exptime, log_likelihood, get_trend):
    """Return the given lightcurve detrended after masking eclipses."""

    half_porb = log_likelihood.best_fit_bls["period"][0]
    gap_indices = (
        numpy.nonzero(
            lightcurve["time"][1:] - lightcurve["time"][:-1]
            > max(2 * log_likelihood.best_fit_bls["duration"],
                  30 * exptime)
        )[0]
        + 1
    )
    gap_indices = numpy.append(gap_indices, lightcurve["time"].size)
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

    detrended = numpy.copy(lightcurve)
    start_index = 0
    good_mask = numpy.ones(detrended.size, dtype=bool)
    for end_index in gap_indices:
        if end_index - start_index < 20:
            good_mask[start_index:end_index] = False
            continue
        scaling = get_trend(
            lightcurve[start_index:end_index],
            half_porb,
            mask[start_index:end_index],
        )
        good_mask[start_index:end_index] = numpy.isfinite(scaling)
        print(
            f"Scaling ({lightcurve['time'][start_index]} < t < "
            f"{lightcurve['time'][end_index-1]}): {scaling!r}"
        )
        detrended[start_index:end_index]["flux"] /= scaling
        detrended[start_index:end_index]["flux_err"] /= scaling
        start_index = end_index

    return detrended[good_mask]


def test():
    """Avoid polluting global namespace."""

    #That is the idea
    #pylint: disable=import-outside-toplevel
    from log_likelihood import LogLikelihood
    from visualize import create_lightcurve_plot
    #pylint: enable=import-outside-toplevel

    for tic in [1045298, 1129033, 1220444, 2020964]:
        try:
            log_likelihood = LogLikelihood(tic, detrend=None)
            if log_likelihood.masked_is_significant():
                zoom = "[zoom_default, zoom_masked]"
            else:
                zoom = "[zoom_even, zoom_odd]"
            config = namedtuple("ConfigType", ["tic_id", "plot_lightcurve"])(
                tic,
                (
                    f"tess{tic}_detrending.pdf",
                    f"[[full, full], [folded, folded], {zoom}]",
                ),
            )
            print(f'Plotting with configuration:\n {config}')
            create_lightcurve_plot(
                config,
                detrend=partial(masked_detrend, get_trend=calc_moving_median),
            )
        except RuntimeError:
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test()
