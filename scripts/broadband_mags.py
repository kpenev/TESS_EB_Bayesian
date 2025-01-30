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
        return max(file["metadata"].keys())


class BroadbandPhotometry:
    """Querry observed broad band magnitudes and predict extinction"""

    def _find_tic(self, tic_entry):
        """Return opened file and index within it containing given TIC entry."""

        gal_coords = SkyCoord(
            ra=tic_entry["ra"] * units.deg,
            dec=tic_entry["dec"] * units.deg,
            frame="icrs",
        ).galactic
        for healpix in healpix_and_neighbors(gal_coords):
            file_index = numpy.searchsorted(self._last_healpix, healpix)
            with h5py.File(
                path.join(
                    broadband_data_dir,
                    f"stellar_params_{file_index:02d}.h5",
                ),
                "r",
            ) as extinction_f:
                index = numpy.where(
                    extinction_f["gaia"][f"{healpix}"]["gaia_id"]
                    == int(tic_entry["GAIA"]),
                )[0]
                if index.size > 0:
                    return extinction_f["gaia"][f"{healpix}"][index]

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

    def __call__(self, tic_ids):
        """Return all info from Green et. al. (2019) for the given TIC."""

        return [
            self._find_tic(tic_entry)
            for tic_entry in Catalogs.query_criteria(catalog="Tic", ID=tic_ids)
        ]


if __name__ == "__main__":
    print(repr(BroadbandPhotometry()([282024596, 94322581])))
