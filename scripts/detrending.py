"""Function implementing LC detrending."""

from matplotlib import pyplot
import numpy
from log_likelihood import LogLikelihood


def moving_median(lightcurve, half_porb):
    """Each point is divided by the median of all points within +-Porb/2."""

    scaling = numpy.empty(lightcurve.shape)
    for i, time in enumerate(lightcurve["time"]):
        left, right = numpy.searchsorted(
            lightcurve["time"], [time - half_porb, time + half_porb]
        )
        scaling[i] = numpy.median(lightcurve["flux"][left:right])
    lightcurve['flux'] /= scaling


if __name__ == "__main__":
    tess = LogLikelihood(1045298) #1129033 #1220444 #2020964
    detrended = numpy.copy(tess.lcs[0][1])
    moving_median(detrended, 1.4634139 / 2)
    pyplot.plot(
        tess.lcs[0][1]["time"],
        tess.lcs[0][1]["flux"] / numpy.median(tess.lcs[0][1]["flux"]),
        ".",
        label="orig",
    )
    pyplot.plot(detrended['time'], detrended['flux'], '.', label='detrended')
    pyplot.legend()
    pyplot.show()
