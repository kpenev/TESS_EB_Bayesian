#!/usr/bin/env python3

"""Utilities for calculating and predicting broadband magnitudes of EBs."""

from glob import glob
from os import path
import logging

import numpy
import h5py
from astroquery.mast import Catalogs
from astroquery.gaia import GaiaClass
from astropy.coordinates import SkyCoord
from astropy import units
from healpy import pixelfunc
from dustmaps.bayestar import BayestarQuery
from scipy.stats import norm

from paths import broadband_data_dir
from panstarrs import get_panstarrs_mags

_logger = logging.getLogger(__name__)


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
        return max(map(int, file["metadata"].keys()))


def get_gaia_distance(gaia_id):
    """Return 50-th, 16-th, & 84-th pencentiles of the Gaia distance estimate"""

    gaia = GaiaClass()
    gaia_distance_entry = gaia.launch_job_async(
        "SELECT distance_gspphot_lower, distance_gspphot, "
        "distance_gspphot_upper FROM gaiadr3.gaia_source WHERE source_id = "
        f"{gaia_id};",
        verbose=True,
    ).get_results()
    print("Gaia distance entry: ", gaia_distance_entry)
    if len(gaia_distance_entry) == 0:
        return None
    quantiles = ["_lower", "", "_upper"]
    result = numpy.array(
        [
            float(gaia_distance_entry[f"distance_gspphot{quant}"])
            for quant in quantiles
        ]
    )
    if numpy.isfinite(result).all():
        return result
    return None


class Green19Correction:
    """Querry Green et. al. (2019) broad band magnitudes and extinction"""

    def _evaluate_bayestar_map(self, tic_entry):
        """Return the same information as direct stellar params but from map."""

        gaia_id = int(tic_entry["GAIA"])
        distance = get_gaia_distance(gaia_id)
        if distance is None:
            raise RuntimeError(
                "No distance found for TIC ID: "
                f"{tic_entry['ID']} (Gaia ID: {gaia_id})"
            )
        coords = SkyCoord(
            ra=tic_entry["ra"] * units.deg,
            dec=tic_entry["dec"] * units.deg,
            distance=distance * units.pc,
            frame="icrs",
        )
        reddening, flags = self._bayestar(
            coords.galactic,
            mode="percentile",
            pct=self._1sigma_pct,
            return_flags=True,
        )
        if not flags["converged"].all() or not flags["reliable_dist"].all():
            message = (
                "The extinction map is "
                f"{'' if flags['reliable_dist'].all() else 'not'} reliable and "
                f"is {'' if flags['converged'].all() else 'not'} converged for "
                f"TIC ID {tic_entry['ID']} (Gaia ID: {gaia_id})"
            )
            if self._ignore_flags:
                _logger.warning(message)
            else:
                raise RuntimeError(message)
        if numpy.isfinite(reddening[1, 1]) or not self._tic_fallback:
            reddening = (
                numpy.array(
                    [
                        -(
                            (
                                (reddening[0, 1] - reddening[1, 1]) ** 2
                                + (reddening[1, 0] - reddening[1, 1]) ** 2
                            )
                            ** 0.5
                        ),
                        0.0,
                        (
                            (reddening[2, 1] - reddening[1, 1]) ** 2
                            + (reddening[1, 2] - reddening[1, 1]) ** 2
                        )
                        ** 0.5,
                    ],
                )
                + reddening[1, 1]
            )
        else:
            reddening = (
                numpy.array([-tic_entry["e_ebv"], 0.0, tic_entry["e_ebv"]])
                + tic_entry["ebv"]
            )
        result = {
            "dm": (5.0 * numpy.log10(distance) - 5.0),
            "E": reddening,
        }
        return result

    def _get_magnitudes(self, tic_entry):
        """Return the magnitudes and uncertainties for the given TIC entry."""

        tic_filters = ["J", "H", "K", "w1", "w2"]
        ps1_mag, ps1_err = get_panstarrs_mags(tic_entry["GAIA"])
        result = {
            "mag": numpy.concatenate(
                [
                    ps1_mag,
                    [tic_entry[f"{fil}mag"] for fil in tic_filters],
                ]
            ),
            "mag_err": numpy.concatenate(
                [
                    ps1_err,
                    [tic_entry[f"e_{fil}mag"] for fil in tic_filters],
                ]
            ),
        }
        result["mag_err"][
            numpy.logical_not(numpy.isfinite(result["mag_err"]))
        ] = 0.1
        print(f"Result: {result!r}")
        return result

    def _get_tic_info(self, tic_entry):
        """Return stellar and extinction parameters for the given TIC entry."""

        result = self._get_magnitudes(tic_entry)
        result["percentiles"] = self._evaluate_bayestar_map(tic_entry)
        return result

        # pylint: disable=line-too-long
        # gal_coords = SkyCoord(
        #    ra=tic_entry["ra"] * units.deg,
        #    dec=tic_entry["dec"] * units.deg,
        #    frame="icrs",
        # ).galactic
        # for healpix in healpix_and_neighbors(gal_coords):
        #    file_index = numpy.searchsorted(self._last_healpix, healpix)
        #    healpix_key = str(healpix)
        #    with h5py.File(
        #        path.join(
        #            broadband_data_dir,
        #            f"stellar_params_{file_index:02d}.h5",
        #        ),
        #        "r",
        #    ) as extinction_f:
        #        if healpix_key not in extinction_f["gaia"]:
        #            continue
        #        index = numpy.where(
        #            extinction_f["gaia"][healpix_key]["gaia_id"]
        #            == int(tic_entry["GAIA"]),
        #        )[0]
        #        if index.size > 0:
        #            index = int(index)
        #            return {
        #                "mag": numpy.concatenate(
        #                    (
        #                        extinction_f["data"][healpix_key][index]["mag"],
        #                        [tic_entry["w1mag"], tic_entry["w2mag"]],
        #                    )
        #                ),
        #                "mag_err": numpy.concatenate(
        #                    (
        #                        extinction_f["data"][healpix_key][index][
        #                            "mag_err"
        #                        ],
        #                        [tic_entry["e_w1mag"], tic_entry["e_w2mag"]],
        #                    )
        #                ),
        #                "percentiles": extinction_f["percentiles"][healpix_key][
        #                    index
        #                ],
        #            }

        # raise RuntimeError(
        #    f"TIC ID: {tic_entry['ID']} = Gaia ID: {tic_entry['GAIA']} "
        #    "not found in Green et. al. (2019) data."
        # )
        # pylint: enable=line-too-long

    def __init__(self, ignore_flags=False, tic_fallback=True):
        """
        Prepare to query the Green et. al. (2019) data.

        Args:
            ignore_flags (bool): If True, warn instead of raising errors when
                convergence and reliability flags of the Bayestar map are False.

            tic_fallback (bool): If True, use TIC E(B-V) values to infer
                reddening when Bayestar map is undefined.
        """

        self._last_healpix = numpy.array(
            [
                get_last_healpix(fname)
                for fname in sorted(glob(path.join(broadband_data_dir, "*.h5")))
            ]
        )
        # Order is: PANSTARRS (g, r, i, z, y) 2MASS (J, H, Ks), WISE (W1, W2)
        self._extinction_coef = numpy.array(
            [
                3.518,
                2.617,
                1.971,
                1.549,
                1.263,
                0.7927,
                0.4690,
                0.3026,
                0.132,
                0.179,
            ]
        )

        self._1sigma_pct = 100.0 * norm.cdf([-1, 0, 1])
        self._bayestar = BayestarQuery(version="bayestar2019")
        self._ignore_flags = ignore_flags
        self._tic_fallback = tic_fallback

    def get_map_data(self, tic_ids):
        """Return all info from Green et. al. (2019) for the given TIC."""

        return [
            self._get_tic_info(tic_entry)
            # False positive
            # pylint: disable=no-member
            for tic_entry in Catalogs.query_criteria(catalog="Tic", ID=tic_ids)
            # pylint: enable=no-member
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
    print(repr(Green19Correction(True).get_absolute_magnitudes(189639080)))
