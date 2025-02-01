#!/usr/bin/env python3

"""Utilities for calculating and predicting broadband magnitudes of EBs."""

from glob import glob
from os import path

import numpy
import h5py
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy import units
from healpy import pixelfunc

from paths import broadband_data_dir


def verify_monotonic():
    """Verify that Gaia IDs are monotonic in individual datasats."""

    for fname in sorted(glob(path.join(broadband_data_dir, "*.h5"))):
        print(fname)
        with h5py.File(fname, "r") as file:
            for dset_name, dset_data in file["metadata"].items():
                assert (
                    dset_data["obj_id"][1:] > dset_data["obj_id"][:-1]
                ).all()
                print(
                    f"\t{dset_name}: "
                    f"{dset_data['obj_id'][0]} - {dset_data['obj_id'][-1]}"
                )


def healpix_and_neighbors(galactic_coords):
    """Iterate over the given pixel and its nearest neighbors."""

#    for pixel in range(11285):
#        yield pixel
    pixel = pixelfunc.ang2pix(
        32, galactic_coords.l.deg, galactic_coords.b.deg, nest=True, lonlat=True
    )
    yield pixel
    for candidate in pixelfunc.get_all_neighbours(32, pixel, nest=True):
        if candidate > 0:
            yield candidate


def get_last_healpix(fname):
    """Return the last healpix in each of the Green et. al. (2019) files."""

    with h5py.File(fname, "r") as file:
        return max(map(int, file["metadata"].keys()))


class Green19Correction:
    """Querry Green et. al. (2019) broad band magnitudes and extinction"""

    def _find_tic(self, tic_entry):
        """Return opened file and index within it containing given TIC entry."""

        gal_coords = SkyCoord(
            ra=tic_entry["ra"] * units.deg,
            dec=tic_entry["dec"] * units.deg,
            frame="icrs",
        ).galactic
        for healpix in healpix_and_neighbors(gal_coords):
            file_index = numpy.searchsorted(self._last_healpix, healpix)
            healpix_key = str(healpix)
            with h5py.File(
                path.join(
                    broadband_data_dir,
                    f"stellar_params_{file_index:02d}.h5",
                ),
                "r",
            ) as extinction_f:
                if healpix_key not in extinction_f["gaia"]:
                    continue
                index = numpy.where(
                    extinction_f["gaia"][healpix_key]["gaia_id"]
                    == int(tic_entry["GAIA"]),
                )[0]
                if index.size > 0:
                    index = int(index)
                    return {
                        "mag": extinction_f["data"][healpix_key][index]["mag"],
                        "mag_err": extinction_f["data"][healpix_key][index][
                            "mag_err"
                        ],
                        "percentiles": extinction_f["percentiles"][healpix_key][
                            index
                        ],
                    }

        raise RuntimeError(
            f"TIC ID: {tic_entry['ID']} = Gaia ID: {tic_entry['GAIA']} "
            "not found in Green et. al. (2019) data."
        )

    def __init__(self):
        """Prepare to query the Green et. al. (2019) data."""

        self._last_healpix = numpy.array(
            [
                get_last_healpix(fname)
                for fname in sorted(glob(path.join(broadband_data_dir, "*.h5")))
            ]
        )
        self._extinction_coef = numpy.array(
            [3.518, 2.617, 1.971, 1.549, 1.263, 0.7927, 0.4690, 0.3026]
        )

    def get_map_data(self, tic_ids):
        """Return all info from Green et. al. (2019) for the given TIC."""

        return [
            self._find_tic(tic_entry)
            for tic_entry in Catalogs.query_criteria(catalog="Tic", ID=tic_ids)
        ]

    def get_absolute_magnitudes(self, tic_ids):
        """Return absolute mags and uncertainties per Green et. al. (2019)."""

        return [
            (
                (
                    map_data["mag"]
                    - map_data["percentiles"]["dm"][1]
                    - map_data["percentiles"]["E"][1] * self._extinction_coef
                ),
                numpy.sqrt(
                    map_data["mag_err"] ** 2
                    + (
                        map_data["percentiles"]["dm"][2]
                        - map_data["percentiles"]["dm"][0]
                    )
                    ** 2
                    / 4
                    + (
                        map_data["percentiles"]["E"][2]
                        - map_data["percentiles"]["E"][0]
                    )
                    ** 2
                    * self._extinction_coef**2
                    / 4
                ),
            )
            for map_data in self.get_map_data(tic_ids)
        ]


if __name__ == "__main__":
    print(repr(Green19Correction().get_absolute_magnitudes(18250189)))
