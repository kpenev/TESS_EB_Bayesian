"""Function implementing LC detrending."""

from matplotlib import pyplot
import numpy
from log_likelihood import LogLikelihood


def moving_median(lightcurve, half_porb, mask):
    """Each point is divided by the median of all points within +-Porb/2."""

    scaling = numpy.empty(lightcurve.shape)
    masked_time = lightcurve["time"][mask]
    masked_flux = lightcurve["flux"][mask]
    for i, time in enumerate(lightcurve["time"]):
        left, right = numpy.searchsorted(
            masked_time, [time - half_porb, time + half_porb]
        )
        scaling[i] = numpy.median(masked_flux[left:right])
    lightcurve["flux"] /= scaling


def test():
    """Avoid polluting global namespace."""

    log_likelihood = LogLikelihood(1045298)  # 1129033 #1220444 #2020964
    detrended = numpy.copy(log_likelihood.lcs[0][1])

    half_porb = log_likelihood.best_fit_bls["period"]
    if log_likelihood.masked_is_significant():
        mask = numpy.logical_and(
            log_likelihood.get_bls_eclipse_mask(detrended, "both"),
            log_likelihood.get_bls_eclipse_mask(detrended, "masked"),
        )
        half_porb /= 2
    else:
        mask = log_likelihood.get_bls_eclipse_mask(detrended, "both")
    moving_median(detrended, half_porb, mask)
    pyplot.plot(
        log_likelihood.lcs[0][1]["time"],
        log_likelihood.lcs[0][1]["flux"]
        / numpy.median(log_likelihood.lcs[0][1]["flux"]),
        ".",
        label="orig",
    )
    pyplot.plot(detrended["time"], detrended["flux"], ".", label="detrended")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    test()
