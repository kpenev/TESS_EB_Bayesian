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
from astropy.table import Table
from astropy import units
from healpy import pixelfunc
from dustmaps.bayestar import BayestarQuery
from scipy.stats import norm

from paths import broadband_data_dir, gaia_extinction_model

_logger = logging.getLogger(__name__)

# Factor TIC-8 applied to SFD-scale E(B-V) (Stassun et al. 2019)
tic_ebv_recalibration = 0.884


class GaiaExtinctionCoefficients:
    """
    Teff-dependent Gaia G, BP, RP extinction per unit Bayestar19 reddening.

    The values at the reference Teff are Wang & Chen (2019) shifted onto the
    Green et al. (2019) scale. The Teff dependence is k(Teff)/k(reference
    Teff) from the dustapprox polynomial model (Kurucz atmospheres, F99 law).
    See extinction_notes.md and data/dustapprox/README.md.
    """

    bands = ("G", "BP", "RP")
    _passbands = ("GAIA_GAIA3.G", "GAIA_GAIA3.Gbp", "GAIA_GAIA3.Grp")
    _variables = ("A0", "R0", "teffnorm")
    _non_term_columns = (
        "passband",
        "mae",
        "rmse",
        "mean_residuals",
        "std_residuals",
    )

    def __init__(
        self,
        model_fname=gaia_extinction_model,
        *,
        reference_coefficients=(2.457, 3.113, 1.880),
        reference_teff=4700.0,
        r0=3.1,
        a0_per_reddening=2.742,
    ):
        """
        Load the dustapprox polynomial model.

        Args:
            model_fname(str):    The dustapprox ecsv file to read.

            reference_coefficients((float, float, float)):    Extinction per
                unit Bayestar19 reddening in G, BP, RP at ``reference_teff``.

            reference_teff(float):    The Teff [K] for which
                ``reference_coefficients`` apply (red clump stars of Wang &
                Chen 2019).

            r0(float):    The R_V of the extinction law to evaluate.

            a0_per_reddening(float):    Extinction at 550 nm per unit
                Bayestar19 reddening, used only for the weak A_0 dependence
                (Landolt V per E(B-V)_SFD, Schlafly & Finkbeiner 2011).
        """

        table = Table.read(model_fname, format="ascii.ecsv")
        terms = [
            name
            for name in table.colnames
            if name not in self._non_term_columns
        ]
        self._powers = numpy.zeros(
            (len(terms), len(self._variables)), dtype=int
        )
        for term_index, term in enumerate(terms):
            if term == "1":
                continue
            for factor in term.split():
                variable, _, power = factor.partition("^")
                self._powers[term_index, self._variables.index(variable)] = (
                    int(power) if power else 1
                )
        passbands = list(table["passband"])
        self._polynomial_coefficients = numpy.array(
            [
                [
                    float(table[passbands.index(passband)][term])
                    for term in terms
                ]
                for passband in self._passbands
            ]
        )
        self.teff_range = tuple(table.meta["atmosphere"]["teff"])
        self._a0_range = tuple(table.meta["extinction"]["A0"])
        self._reference_coefficients = numpy.array(reference_coefficients)
        self._reference_teff = reference_teff
        self._r0 = r0
        self._a0_per_reddening = a0_per_reddening

    def _evaluate_polynomial(self, a0, teff):
        """Return k = A_X/A_0 for each band (rows) and Teff (columns)."""

        features = numpy.stack(
            numpy.broadcast_arrays(a0, self._r0, numpy.asarray(teff) / 5040.0)
        )
        terms = numpy.prod(
            features[None, :, :] ** self._powers[:, :, None], axis=1
        )
        return self._polynomial_coefficients @ terms

    def __call__(self, teff, reddening):
        """
        Return the extinction per unit reddening for the given temperatures.

        Args:
            teff(array):    Effective temperatures [K], e.g. one per binary
                component. Clamped to the validity range of the model.

            reddening(float):    The Bayestar19 reddening (sets A_0).

        Returns:
            array:    Shape (3, len(teff)), rows are G, BP, RP.
        """

        teff = numpy.clip(numpy.atleast_1d(teff), *self.teff_range)
        a0 = numpy.clip(self._a0_per_reddening * reddening, *self._a0_range)
        polynomial = self._evaluate_polynomial(
            a0, numpy.append(teff, self._reference_teff)
        )
        return (
            self._reference_coefficients[:, None]
            * polynomial[:, :-1]
            / polynomial[:, -1:]
        )


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


def correct_gaia_photometry(
    entry, *, calibration_floor=0.01, blend_threshold=3.0
):
    """
    Apply the Riello et al. (2021) corrections to Gaia DR3 photometry.

    Args:
        entry(dict):    The gaia_source columns phot_{g,bp,rp}_mean_mag,
            phot_{g,bp,rp}_mean_flux, phot_{g,bp,rp}_mean_flux_error, bp_rp
            and phot_bp_rp_excess_factor of one source, with NaN for missing
            values.

        calibration_floor(float):    Uncertainty [mag] added in quadrature to
            each band (absolute calibration of about 1%).

        blend_threshold(float):    How many sigma the corrected BP/RP flux
            excess C* must exceed for the source to be treated as blended.

    Returns:
        dict:
            ``mag``, ``mag_err``: G, BP, RP magnitudes and uncertainties.

            ``blend_err``: Extra uncertainty [mag] to add in quadrature to
            every SED band (0 unless the source is blended).
    """

    polyval = numpy.polynomial.polynomial.polyval
    bands = ("g", "bp", "rp")
    mag = numpy.array([entry[f"phot_{band}_mean_mag"] for band in bands])
    mag_err = (
        2.5
        / numpy.log(10.0)
        * numpy.array(
            [
                entry[f"phot_{band}_mean_flux_error"]
                / entry[f"phot_{band}_mean_flux"]
                for band in bands
            ]
        )
    )
    g_mag = entry["phot_g_mean_mag"]
    rp_mag = entry["phot_rp_mean_mag"]
    color = entry["bp_rp"]

    # The G correction for sources processed with a default nu_eff
    # (Riello et al. 2021, Sect. 8.3; astrometric_params_solved = 3 or 95)
    # is already included in Gaia DR3 phot_g_mean_mag, so it is not applied
    # here. This function is therefore only valid for DR3 photometry.

    # Saturation (Appendix C.1, Eqs. 24-26)
    if 2.0 < g_mag < 8.0:
        mag[0] += polyval(g_mag, (-0.09892, 0.059, -0.009775, 0.0004934))
    if 2.0 < g_mag < 3.94:
        mag[1] += polyval(g_mag, (-0.9921, -0.02598, 0.1833, -0.02862))
    if 2.0 < rp_mag < 3.45:
        mag[2] += polyval(rp_mag, (-14.94, 14.41, -4.657, 0.503))

    # Magnitude trend of G for blue and bright sources (Sect. 8.4)
    if 8.0 < g_mag < 13.0 and color < -0.1:
        mag_err[0] = numpy.hypot(mag_err[0], 0.054 - 0.0046 * g_mag)

    mag_err = numpy.hypot(mag_err, calibration_floor)

    # Blending, from the corrected BP/RP flux excess (Sect. 9.4, Table 2 and
    # Eq. 21). The excess factor is C = (I_BP + I_RP) / I_G, with I the mean
    # fluxes (phot_*_mean_flux), and C* = C - f(BP-RP). BP and RP are
    # measured in 3.5" x 2.1" windows while G is PSF/LSF fitted, so a
    # neighbour adding flux d mostly to BP and RP gives C* ~ d / I_G. Riello
    # et al. use C* only as a quality filter. Turning it into an uncertainty
    # is our heuristic: 2.5 log10(1 + C*) is roughly the brightening of BP/RP
    # by such a neighbour (overestimated by a factor of about C). It is
    # applied to all bands because a neighbour inside the BP/RP windows is
    # likely also inside the 2MASS and WISE beams, though how much it adds
    # there depends on its separation and color.
    if color < 0.5:
        excess_coefficients = (1.154360, 0.033772, 0.032277)
    elif color < 4.0:
        excess_coefficients = (1.162004, 0.011464, 0.049255, -0.005879)
    else:
        excess_coefficients = (1.057572, 0.140537)
    c_star = entry["phot_bp_rp_excess_factor"] - polyval(
        color, excess_coefficients
    )
    c_star_sigma = 0.0059898 + 8.817481e-12 * g_mag**7.618399
    blend_err = (
        2.5 * numpy.log10(1.0 + c_star)
        if c_star > blend_threshold * c_star_sigma
        else 0.0
    )
    return {"mag": mag, "mag_err": mag_err, "blend_err": blend_err}


def get_gaia_photometry(gaia_id, **correction_kwargs):
    """
    Return corrected Gaia DR3 G, BP, RP photometry for the given source.

    See `correct_gaia_photometry()` for the corrections, keyword arguments
    and return value. Returns None if the source is not found.
    """

    gaia_photometry_columns = [
        f"phot_{band}_mean_{quantity}"
        for band in ("g", "bp", "rp")
        for quantity in ("mag", "flux", "flux_error")
    ] + ["bp_rp", "phot_bp_rp_excess_factor"]
    gaia = GaiaClass()
    gaia_entry = gaia.launch_job_async(
        f"SELECT {', '.join(gaia_photometry_columns)} "
        f"FROM gaiadr3.gaia_source WHERE source_id = {gaia_id};",
        verbose=True,
    ).get_results()
    if len(gaia_entry) == 0:
        return None
    return correct_gaia_photometry(
        {
            column: float(numpy.ma.filled(gaia_entry[column][0], numpy.nan))
            for column in gaia_photometry_columns
        },
        **correction_kwargs,
    )


class Green19Correction:
    """Querry Green et. al. (2019) broad band magnitudes and extinction"""

    # Extinction per unit Bayestar19 reddening in 2MASS (J, H, Ks) and WISE
    # (W1, W2).
    infrared_extinction_coef = numpy.array(
        [0.7927, 0.4690, 0.3026, 0.179, 0.132]
    )
    # Uncertainty of the extinction law per unit reddening (see
    # extinction_notes.md): a gray offset shared by all bands (the
    # uncertainty of A_W2), and the 1-sigma sightline variation of the curve
    # shape (Schlafly et al. 2016) in G, BP, RP, J, H, Ks, W1, W2.
    extinction_gray_err = 0.03
    extinction_shape_err = numpy.array(
        [0.051, 0.021, 0.069, 0.042, 0.025, 0.016, 0.005, -0.003]
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
        gaia = get_gaia_photometry(tic_entry["GAIA"])
        if gaia is None:
            gaia = {
                "mag": numpy.full(3, numpy.nan),
                "mag_err": numpy.full(3, numpy.nan),
                "blend_err": 0.0,
            }
        result = {
            "mag": numpy.concatenate(
                [
                    gaia["mag"],
                    [tic_entry[f"{fil}mag"] for fil in tic_filters],
                ]
            ),
            "mag_err": numpy.concatenate(
                [
                    gaia["mag_err"],
                    [tic_entry[f"e_{fil}mag"] for fil in tic_filters],
                ]
            ),
        }
        result["mag_err"][
            numpy.logical_not(numpy.isfinite(result["mag_err"]))
        ] = 0.1
        result["mag_err"] = numpy.hypot(result["mag_err"], gaia["blend_err"])
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

    def get_distance_corrected_magnitudes(self, tic_ids):
        """
        Return magnitudes minus distance modulus and the shared quantities.

        Extinction is not subtracted, because for the Gaia bands it depends on
        the Teff of each star. It is applied to the model instead.

        Returns:
            [(array, array, dict)]:
                For each TIC: the magnitudes minus the distance modulus, their
                photometric-only uncertainties, and the quantities shared by
                all bands: ``reddening`` (Bayestar19 E),
                ``distance_modulus_err``, ``reddening_distance_err`` (the
                change in reddening with a 1-sigma change in distance) and
                ``reddening_map_err`` (the map uncertainty at fixed distance).
        """

        result = []
        for map_data in self.get_map_data(tic_ids):
            percentiles = map_data["percentiles"]
            result.append(
                (
                    map_data["mag"] - percentiles["dm"][1],
                    map_data["mag_err"],
                    {
                        "reddening": percentiles["E"],
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
    print(
        repr(Green19Correction(True).get_distance_corrected_magnitudes(91961))
    )
