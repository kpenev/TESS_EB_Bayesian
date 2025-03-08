"""Tools for interpolating gravity darkening coefficients of stars."""

from matplotlib import pyplot
from astropy.io import fits
import numpy

from general_purpose_python_modules import grid_tracks_interpolate

import paths


class GravDarkInterpolator:
    """Implement callable interpolating the gravity darkening coefficients."""

    def __init__(self):
        """Prepare the instance for use."""

        with fits.open(paths.grav_dark["TESS"], "readonly") as grav_dark_f:
            raw_data = grav_dark_f[1].data

        raw_data = raw_data[raw_data["xi"] == 2]
        raw_data = raw_data[
            numpy.argsort(
                1e6 * raw_data["Z"].astype(float)
                + 1e3 * raw_data["logg"].astype(float)
                + raw_data["logTeff"].astype(float)
            )
        ]
        self._logteff_range = (
            raw_data["logTeff"].min(),
            raw_data["logTeff"].max(),
        )
        self._grid = tuple(
            (var, numpy.unique(raw_data[var])) for var in ["Z", "logg"]
        )
        self._data = []
        i = 0
        while i < raw_data.size:
            if i == 0:
                feh, logg = raw_data[0]["Z"], raw_data[0]["logg"]
            if raw_data[i]["Z"] != feh or raw_data[i]["logg"] != logg:
                assert (
                    feh
                    == self._grid[0][1][
                        len(self._data) // self._grid[1][1].size
                    ]
                )
                assert (
                    logg
                    == self._grid[1][1][len(self._data) % self._grid[1][1].size]
                )
                self._data.append(raw_data[:i])
                assert (
                    self._data[-1]["logTeff"][1:]
                    > self._data[-1]["logTeff"][:-1]
                ).all()
                raw_data = raw_data[i:]
                i = 0
            else:
                i += 1
        self._data.append(raw_data)

    def __call__(self, **interpolate_to):
        """Return the gravity darkening coefficient for given parameters."""

        return grid_tracks_interpolate(
            interpolate_to, ("y",), self._grid, self._data
        )[0]

    def get_range(self, quantity):
        """Return the available interpolation range of the given quantity."""

        if quantity == "logTeff":
            return self._logteff_range
        for name, values in self._grid:
            if name == quantity:
                return values[0], values[-1]
        raise ValueError(f"Unknown grid quantity {quantity}!")


def make_test_plots():
    """Create plots showing the interpolation in action."""

    with fits.open(paths.grav_dark["TESS"], "readonly") as grav_dark_f:
        raw_data = grav_dark_f[1].data
    raw_data = raw_data[raw_data["xi"] == 2]
    raw_data = raw_data[
        numpy.argsort(
            1e6 * raw_data["Z"] + 1e3 * raw_data["logg"] + raw_data["logTeff"]
        )
    ]

    interp = GravDarkInterpolator()
    logg = 4.5
    logteff = numpy.linspace(3.6, 4.6, 1000)
    gravdark = {
        feh: [interp(logg=logg, Z=feh, logTeff=lgt) for lgt in logteff]
        for feh in [-2.5, -numpy.pi / 2, 0.0, 0.99, 1.0]
    }
    for feh, plot_y in gravdark.items():
        pyplot.plot(logteff, plot_y, label=f"[Fe/H] = {feh}")
        if numpy.isclose(feh, raw_data["Z"]).any():
            raw_selection = raw_data[
                numpy.logical_and(raw_data["Z"] == feh, raw_data["logg"] == 4.5)
            ]
            pyplot.plot(
                raw_selection["logTeff"],
                raw_selection["y"],
                "o",
                label=f"raw [Fe/H]={feh}",
            )

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    make_test_plots()
