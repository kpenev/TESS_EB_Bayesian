#!/usr/bin/env python3

"""Utilities for calculating and predicting broadband magnitudes of EBs."""

from glob import glob
from os import path
import logging
from time import sleep

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

# Factor TIC-8 applied to SFD-scale E(B-V) (Stassun et al. 2019)
tic_ebv_recalibration = 0.884


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
    """
    Return 16-th, 50-th, & 84-th percentiles of the geometric distance.

    Uses the Bailer-Jones et al. (2021) geometric distances, which depend only
    on the parallax (identical in Gaia EDR3 and DR3) and a Galactic prior, not
    on single-star models of the photometry. Returns None if the distance is
    not available.
    """

    gaia = GaiaClass()
    gaia_distance_entry = gaia.launch_job_async(
        "SELECT r_lo_geo, r_med_geo, r_hi_geo FROM external.gaiaedr3_distance "
        f"WHERE source_id = {gaia_id};",
        verbose=True,
    ).get_results()
    print("Gaia distance entry: ", gaia_distance_entry)
    if len(gaia_distance_entry) == 0:
        return None
    result = numpy.array(
        [
            float(gaia_distance_entry[f"r_{quantile}_geo"])
            for quantile in ["lo", "med", "hi"]
        ]
    )
    if not numpy.isfinite(result).all():
        return None
    return result


class Green19Correction:
    """Querry Green et. al. (2019) broad band magnitudes and extinction"""

    # Extinction per unit Bayestar19 reddening. Order is: PANSTARRS (g, r, i,
    # z, y) 2MASS (J, H, Ks), WISE (W1, W2)
    extinction_coef = numpy.array(
        [
            3.518,
            2.617,
            1.971,
            1.549,
            1.263,
            0.7927,
            0.4690,
            0.3026,
            0.179,
            0.132,
        ]
    )

    def _evaluate_bayestar_map(self, tic_entry):
        """Return the same information as direct stellar params but from map."""

        gaia_id = int(tic_entry["GAIA"])
        distance = get_gaia_distance(gaia_id)
        if distance is None:
            raise ValueError(
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
        # reddening[i, j]: i indexes the distance (16th, 50th, 84th
        # percentile), j the percentile of the map at that distance.
        if numpy.isfinite(reddening[1, 1]) or not self._tic_fallback:
            result = {
                "E": reddening[1, 1],
                "E_distance_err": (reddening[2, 1] - reddening[0, 1]) / 2,
                "E_map_err": (reddening[1, 2] - reddening[1, 0]) / 2,
            }
        else:
            # TIC E(B-V) includes the Schlafly & Finkbeiner (2011)
            # recalibration (Stassun et al. 2019, Sec. 2.3.3), while
            # Bayestar19 units match the raw SFD scale (Green et al. 2019,
            # Sec. 2.1), so undo the recalibration.
            result = {
                "E": tic_entry["ebv"] / tic_ebv_recalibration,
                "E_distance_err": 0.0,
                "E_map_err": tic_entry["e_ebv"] / tic_ebv_recalibration,
            }
        result["dm"] = 5.0 * numpy.log10(distance) - 5.0
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

        for i in range(10):
            try:
                result = self._get_magnitudes(tic_entry)
                result["percentiles"] = self._evaluate_bayestar_map(tic_entry)
                return result
            except:  # pylint: disable=bare-except
                _logger.warning(
                    "Collecting SED information failed for for TIC ID %s "
                    "(Gaia ID: %s), attempt %s/10.",
                    tic_entry["ID"],
                    tic_entry["GAIA"],
                    i + 1,
                    exc_info=True,
                )
                sleep(120)
                if i == 9:
                    raise
        assert False

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
        """
        Return absolute mags and uncertainties per Green et. al. (2019).

        Returns:
            [(array, array, dict)]:
                For each TIC: the absolute magnitudes, their photometric-only
                uncertainties, and the uncertainties shared by all bands:
                ``distance_modulus_err``, ``reddening_distance_err`` (the
                change in reddening with a 1-sigma change in distance) and
                ``reddening_map_err`` (the map uncertainty at fixed
                distance). Band ``i`` shifts by ``extinction_coef[i]`` per unit
                reddening.
        """

        result = []
        for map_data in self.get_map_data(tic_ids):
            percentiles = map_data["percentiles"]
            result.append(
                (
                    map_data["mag"]
                    - percentiles["dm"][1]
                    - percentiles["E"] * self.extinction_coef,
                    map_data["mag_err"],
                    {
                        "distance_modulus_err": (
                            percentiles["dm"][2] - percentiles["dm"][0]
                        )
                        / 2,
                        "reddening_distance_err": percentiles["E_distance_err"],
                        "reddening_map_err": percentiles["E_map_err"],
                    },
                )
            )
        return result


if __name__ == "__main__":
    print(repr(Green19Correction(True).get_absolute_magnitudes(91961)))
