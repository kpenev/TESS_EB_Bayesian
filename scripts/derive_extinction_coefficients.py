#!/usr/bin/env python3

"""
Derive WISE extinction coefficients consistent with Green et al. (2019).

Green et al. (2019, arXiv:1905.02734, Table 1) give coefficients converting
Bayestar19 reddening to extinction in PS1 grizy and 2MASS JHKs, derived from
the Schlafly et al. (2016, arXiv:1602.03928, Table 2) mean reddening vector.
That vector has an arbitrary scale and zero point, so we fit the scale and
offset that reproduce the Green et al. (2019) coefficients and apply the same
transformation to the W1 and W2 entries.
"""

from matplotlib import pyplot
import numpy

bands = ["g", "r", "i", "z", "y", "J", "H", "Ks", "W1", "W2"]

# Schlafly et al. (2016) Table 2: mean reddening vector R0
schlafly16_reddening_vector = numpy.array(
    [
        0.6537,
        0.3906,
        0.2020,
        0.0787,
        -0.0048,
        -0.1421,
        -0.2366,
        -0.2852,
        -0.3213,
        -0.3350,
    ]
)

# Green et al. (2019) Table 1: extinction per unit Bayestar19 reddening
green19_coefficients = numpy.array(
    [3.518, 2.617, 1.971, 1.549, 1.263, 0.7927, 0.4690, 0.3026]
)


def main():
    """Fit scale and offset, then print coefficients for all bands."""

    num_green = green19_coefficients.size
    design_matrix = numpy.column_stack(
        [schlafly16_reddening_vector[:num_green], numpy.ones(num_green)]
    )
    (scale, offset), *_ = numpy.linalg.lstsq(
        design_matrix, green19_coefficients, rcond=None
    )
    derived = scale * schlafly16_reddening_vector + offset

    print(f"Scale: {scale:.4f}, offset: {offset:.4f}")
    print(f"{'Band':4s} {'Derived':>8s} {'Green19':>8s}")
    for band_index, band in enumerate(bands):
        green = (
            f"{green19_coefficients[band_index]:8.4f}"
            if band_index < num_green
            else f"{'-':>8s}"
        )
        print(f"{band:4s} {derived[band_index]:8.4f} {green}")
    print(
        "Max |derived - Green19|: "
        f"{numpy.abs(derived[:num_green] - green19_coefficients).max():.2e}"
    )
    yrange = numpy.array(
        [min(schlafly16_reddening_vector), max(schlafly16_reddening_vector)]
    )
    pyplot.plot(yrange * scale + offset, yrange, "-k", linewidth=3)
    pyplot.plot(
        green19_coefficients, schlafly16_reddening_vector[:num_green], "ok"
    )
    for band in ["W1", "W2"]:
        band_index = bands.index(band)
        pyplot.plot(
            derived[band_index],
            schlafly16_reddening_vector[band_index],
            "*",
            markersize=10,
        )
    pyplot.show()


if __name__ == "__main__":
    main()
