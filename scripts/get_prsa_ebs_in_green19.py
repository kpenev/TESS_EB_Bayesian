"""Return a list of TIC IDs from the Prsa EB catalog in G19 stellar params."""

from os import path
from glob import glob

import numpy
import pandas
import h5py
from astropy.coordinates import SkyCoord
from astropy import units
from astroquery.mast import Catalogs

from paths import data_dir, broadband_data_dir
from extinction_correction import healpix_and_neighbors, get_last_healpix


def find_tic(tic_entry, last_healpix):
    """Return True iff the given TIC entry is in Green et. al. 19 params."""

    gal_coords = SkyCoord(
        ra=tic_entry["ra"] * units.deg,
        dec=tic_entry["dec"] * units.deg,
        frame="icrs",
    ).galactic
    for healpix in healpix_and_neighbors(gal_coords):
        if healpix > last_healpix.max():
            continue
        file_index = numpy.searchsorted(last_healpix, healpix)
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
            try:
                gaia_id = int(tic_entry["GAIA"])
            except numpy.ma.core.MaskError:
                continue
            index = numpy.where(
                extinction_f["gaia"][healpix_key]["gaia_id"] == gaia_id,
            )[0]
            if index.size > 0:
                return True
    return False


def main():
    """Return the TIC entries for all sources in the given Prsa EB catalog."""

    prsa_cat = pandas.read_csv(
        path.join(
            data_dir,
            "hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv",
        )
    )
    print(f"Prsa cat: {prsa_cat!r}\nColumns: {prsa_cat.columns}")
    last_healpix = numpy.array(
        [
            get_last_healpix(fname)
            for fname in sorted(glob(path.join(broadband_data_dir, "*.h5")))
        ]
    )

    for tic_id in prsa_cat["tess_id"]:
        tic_entry = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
        assert len(tic_entry) == 1
        found = find_tic(tic_entry[0], last_healpix)
        print(f"TIC {tic_id} {found!r}")


if __name__ == "__main__":
    main()
