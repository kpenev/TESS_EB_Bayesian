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
                1e6 * raw_data["Z"]
                + 1e3 * raw_data["logg"]
                + raw_data["logTeff"]
            )
        ]
        self._grid = tuple(
            (var, numpy.unique(raw_data[var])) for var in ["Z", "logg"]
        )
        self._data = []
        i = 0
        while i < raw_data.size:
            if i == 0:
                feh, logg = raw_data[0]["Z"], raw_data[0]["logg"]
            if raw_data[i]["Z"] != feh or raw_data[i]["logg"] != logg:
                self._data.append(raw_data[:i])
                raw_data = raw_data[i:]
                i = 0
            else:
                i += 1
        self._data.append(raw_data)
        for track in self._data:
            print(
                "log10(Teff) range: "
                f"{track['logTeff'][0]} --- {track['logTeff'][-1]}"
            )

    def __call__(self, **interpolate_to):
        """Return the gravity darkening coefficient for given parameters."""

        return grid_tracks_interpolate(
            interpolate_to, ("y",), self._grid, self._data
        )[0]

    def get_range(self, quantity):
        """Return the available interpolation range of the given quantity."""

        for name, values in self._grid:
            if name == quantity:
                return values[0], values[-1]
        raise ValueError(f"Unknown grid quantity {quantity}!")


if __name__ == "__main__":
    interp = GravDarkInterpolator()
    logteff = numpy.linspace(3.5, 5.0, 100)
    gravdark = [interp(logg=4.5, Z=0.0, logTeff=lgt) for lgt in logteff]
    gravdark1 = [interp(logg=4.5, Z=0.01, logTeff=lgt) for lgt in logteff]

    pyplot.plot(logteff, gravdark, '-k')
    pyplot.plot(logteff, gravdark1, ':r')

    pyplot.show()
