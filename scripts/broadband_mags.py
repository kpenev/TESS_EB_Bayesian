#!/usr/bin/env python3

"""Utilities for calculating and predicting broadband magnitudes of EBs."""

from glob import glob
from os import path

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


def get_extinction(tic_ids):
    """Return the extinctions from Green et. al. (2019) for given TIC IDs."""

    for tic_entry in Catalogs.query_criteria(catalog="Tic", ID=tic_ids):
        print(
            f"Equatorial coords: RA={tic_entry['ra']}, Dec={tic_entry['dec']}"
        )
        gal_coords = SkyCoord(
            ra=tic_entry["ra"] * units.deg,
            dec=tic_entry["dec"] * units.deg,
            frame="icrs",
        ).galactic
        print(f"Galacting coords: {gal_coords}")
        healpix = pixelfunc.ang2pix(
            32, gal_coords.l.rad, gal_coords.b.rad, lonlat=True
        )
        print(f"Galactic Healpix: {healpix}")


if __name__ == "__main__":
    get_extinction([282024596, 94322581])
