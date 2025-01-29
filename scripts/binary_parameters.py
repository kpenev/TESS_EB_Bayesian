"""Unified interface for binary parameters needed by the likelihood function."""

from collections import namedtuple

import numpy
from astropy import units, constants
import batman
import phoebe

from general_purpose_python_modules.kepler_angles import (
    E_to_nu,
    M_to_E,
    nu_to_E,
    E_to_M,
)
from general_purpose_python_modules.cmd_utils import CMDInterpolator

from gravity_darkening import GravDarkInterpolator
from paths import cmd_data_fname

InputParams = namedtuple(
    "InputParams",
    [
        "mtotal",
        "mratio",
        "age_gyr",
        "feh",
        "per",
        "esinw",
        "ecosw",
        "inc",
        "perpass_phase",
        "primary_limb_dark_1",
        "primary_limb_dark_2",
        "secondary_limb_dark_1",
        "secondary_limb_dark_2",
        "primary_prot",
        "secondary_prot",
        "primary_reflection_coef",
        "secondary_reflection_coef",
        "primary_beaming_coef",
        "secondary_beaming_coef",
    ],
)


# This is set by BATMAN
# pylint: disable=too-many-instance-attributes
class BinaryParams(batman.TransitParams):
    """Extend the batman parameters with everything needed by model."""

    _cmd_interpolator = CMDInterpolator(cmd_data_fname)

    @staticmethod
    def _eclipse_phase_difference(esinw, ecosw, coti=0.0):
        """
        Calculate the phase difference between secondary and primary eclipse.

        Uses equation 31 (and correction for non-central transits) from
        Sterne 1940 (PNAS 26, 36):

        `https://ui.adsabs.harvard.edu/abs/1940PNAS...26...36S/abstract`_

        Note that for BATMAN purposes the inclination correction should not be
        applied since the parameter used is the time of conjunction, not
        mid-transit.

        Args:
            esinw(float):    Eccentricity times sin of longitude of periapsis.

            ecosw(float):    Eccentricity times cos of longitude of periapsis.

            coti(float):    1/tan of the inclination angle. Leave zero to
                            disable the correction to Eq. 31 in Sterne 1940

        Returns:
            float:
                The fraction of the orbital period that elapses between primary
                and secondary eclipses.
        """

        e_square = esinw**2 + ecosw**2

        central_transits_rhs = ecosw * (1.0 - e_square) ** 0.5 / (
            1.0 - esinw**2
        ) + numpy.arctan(ecosw / (1.0 - e_square) ** 0.5)

        if coti:
            coti2 = coti**2
            inclination_correction = (
                0.5
                * ecosw
                * coti2
                * (
                    1.0
                    / (
                        (1.0 + esinw) ** 3
                        + (1.0 + esinw)
                        * (esinw + esinw**2 + 3.0 * ecosw**2)
                        * coti2
                    )
                    + 1.0
                    / (
                        (1.0 - esinw) ** 3
                        + (1.0 - esinw)
                        * (-esinw + esinw**2 + 3.0 * ecosw**2)
                        * coti2
                    )
                )
            )
        else:
            inclination_correction = 0.0

        return (central_transits_rhs + inclination_correction) / numpy.pi + 0.5

    def _set_per_star(self):
        """Set the per-star attributes to match the primary."""

        for attr in self._per_star_attr:
            print(f"Setting {attr}")
            setattr(self, attr, getattr(self, f"_{attr}_both")["primary"])

    def set_from_phoebe(self, phoebe_binary):
        """Set BATMAN params independent of component given a PHOEBE binary."""

        orbit = phoebe_binary["orbit@component"]
        self._t0_both = {"primary": orbit["t0_supconj"].get_value(units.day)}
        self.t0_perpass = orbit["t0_perpass"].get_value(units.day)
        self.per = orbit["period"].get_value(units.day)
        self.rp = orbit["requivratio"].get_value("")
        self.a = (1.0 + self.rp) / orbit["requivsumfrac"].get_value("")
        self.inc = orbit["incl"].get_value(units.deg)
        self.ecc = orbit["ecc"].get_value("")
        self.w = orbit["per0"].get_value(units.deg)
        # (orbit['long_an'].get_value(units.deg)
        #                   +
        #                   orbit['per0'].get_value(units.deg)
        #                   +
        #                   90.0)

        self._t0_both["secondary"] = self._t0_both["primary"] + (
            self.per
            * self._eclipse_phase_difference(
                orbit["esinw"].get_value(""),
                orbit["ecosw"].get_value(""),
            )
        )

        self._linear_limbdark_both = {}
        self._gravdark_both = {}
        self._u_both = {}
        self._limb_dark_both = {}
        self.mtotal = 0.0

        self.teff_ratio = phoebe_binary["secondary@teff"].get_value(
            "K"
        ) / phoebe_binary["primary@teff"].get_value("K")

        for component in ["primary", "secondary"]:
            star = phoebe_binary[component]
            phoebe_coefs = phoebe_binary.compute_ld_coeffs(
                dataset="bol", component=component
            )
            self._limb_dark_both[component] = star["ld_func_bol"].get_value()
            assert len(phoebe_coefs) == 1
            for value in phoebe_coefs.values():
                self._u_both[component] = value
            self.mtotal += star["component@mass"].get_value(units.Msun)
            if component == "primary":
                self.mratio = 1.0 / self.mtotal
                self.rstar = star["requiv"].get_value(units.R_sun)
            else:
                self.mratio *= star["component@mass"].get_value(units.Msun)

            self._prot_both[component] = star["component@period"].get_value(
                units.day
            )
            self._linear_limbdark_both[component] = self._u_both[component]
            self._gravdark_both[component] = star["gravb_bol"].get_value("")
        self._set_per_star()

    def set_from_mcmc(self, sample_params: InputParams):
        """Set the binary parameters from an MCMC sample."""

        self.t0_perpass = sample_params.perpass_phase * sample_params.per
        for param in ["per", "inc"]:
            setattr(self, param, getattr(sample_params, param))
        self.ecc = (sample_params.ecosw**2 + sample_params.esinw**2) ** 0.5
        self.w = (
            numpy.arctan2(sample_params.esinw, sample_params.ecosw)
            * 180.0
            / numpy.pi
        )

        mprimary = sample_params.mtotal / (1.0 + sample_params.mratio)
        interp_kwargs = {
            "MH": sample_params.feh,
            "logAge": 9.0 + numpy.log10(sample_params.age_gyr),
        }

        interpolated = {
            "primary": self._cmd_interpolator(
                ("Mass", "logL", "logTe") + self._passbands,
                Mini=mprimary,
                **interp_kwargs,
            ),
            "secondary": self._cmd_interpolator(
                ("Mass", "logL", "logTe") + self._passbands,
                Mini=mprimary * sample_params.mratio,
                **interp_kwargs,
            ),
        }
        self.mtotal = interpolated["primary"][0] + interpolated["secondary"][0]
        self.mratio = interpolated["secondary"][0] / interpolated["primary"][0]
        radii = {}
        for component, comp_interp in interpolated.items():
            radii[component] = (
                10.0 ** (comp_interp[1] / 2.0 - 2.0 * comp_interp[2])
                / (2.0 * units.K**2)
                * numpy.sqrt(units.L_sun / (numpy.pi * constants.sigma_sb))
            )
            self._gravdark_both[component] = self._gravdark_interp(
                logg=numpy.log10(
                    (
                        constants.G
                        * comp_interp[0]
                        * units.M_sun
                        / radii[component] ** 2
                    ).to_value(units.cm / units.s**2)
                ),
                Z=sample_params.feh,
                logTeff=comp_interp[2],
            )
        self.a = (
            (
                (self.per * units.day) ** 2
                * (constants.G * self.mtotal * units.M_sun)
                / (4.0 * numpy.pi**2)
            )
            ** (1.0 / 3.0)
            / radii["primary"]
        ).to_value()
        self.rstar = radii["primary"].to_value(units.R_sun)
        self.rp = radii["secondary"] / radii["primary"]

        self._t0_both["primary"] = (
            self.t0_perpass
            + E_to_M(nu_to_E(numpy.pi / 2 - self.w, self.ecc), self.ecc)
            / (2.0 * numpy.pi)
            * self.per
        )
        self._t0_both["secondary"] = self._t0_both["primary"] + (
            self.per
            * self._eclipse_phase_difference(
                sample_params.esinw, sample_params.ecosw
            )
        )
        self.teff_ratio = 10.0 ** (
            interpolated["secondary"][2] - interpolated["primary"][2]
        )
        for component in ["primary", "secondary"]:
            self._limb_dark_both[component] = "quadratic"
            self._u_both[component] = [
                getattr(sample_params, f"{component}_limb_dark_1"),
                getattr(sample_params, f"{component}_limb_dark_2"),
            ]
            # From least squares diff between linear and quadratic profiles
            self._linear_limbdark_both[component] = (
                self._u_both[component][0] + 0.3 * self._u_both[component][1]
            )
            for param in ["prot", "reflection_coef", "beaming_coef"]:
                getattr(self, f"_{param}_both")[component] = getattr(
                    sample_params, f"{component}_{param}"
                )
        self._set_per_star()

    def __init__(self, *, from_phoebe=None, from_mcmc=None):
        """Set the model parameters either from PHOEBE binary or MCMC sample."""

        super().__init__()
        self._per_star_attr = [
            "t0",
            "u",
            "limb_dark",
            "prot",
            "linear_limbdark",
            "gravdark",
            "reflection_coef",
            "beaming_coef",
        ]
        self.per = None
        self.rp = None
        self.a = None
        self.inc = None
        self.ecc = None
        self.w = None
        self.mtotal = None
        self.mratio = None
        self.rstar = None
        self.teff_ratio = None
        self.t0_perpass = None
        self.inverted = False
        for attr in self._per_star_attr:
            setattr(self, f"_{attr}_both", {"primary": None, "secondary": None})

        self._passbands = tuple(b + "mag" for b in "UBVRIJHK")

        self._gravdark_interp = GravDarkInterpolator()

        if from_phoebe:
            self.set_from_phoebe(from_phoebe)
            assert not from_mcmc
        if from_mcmc:
            self.set_from_mcmc(from_mcmc)

    def swap_components(self):
        """Swap which star is considered primary vs secondary."""

        component = "primary" if self.inverted else "secondary"
        for attr in self._per_star_attr:
            setattr(self, attr, getattr(self, f"_{attr}_both")[component])

        self.rstar *= self.rp
        self.rp = 1.0 / self.rp
        self.mratio = 1.0 / self.mratio
        self.a *= self.rp
        self.w = (self.w + 180.0) % 360.0
        self.teff_ratio = 1.0 / self.teff_ratio
        self.inverted = not self.inverted

    def to_phoebe(self):
        """Return PHOEBE binary with parameters specified in this object."""

        result = phoebe.default_binary()
        result.add_dataset("lc", times=0, label="lc01")
        result.flip_constraint("mass@primary", solve_for="sma")
        result.flip_constraint(
            "requivratio", solve_for="requiv@secondary@component"
        )
        result.flip_constraint(
            "requivsumfrac", solve_for="requiv@primary@component"
        )

        orbit = result["orbit@component"]
        orbit["t0_perpass"].set_value(self.t0_perpass * units.day)
        orbit["period"].set_value(self.per * units.day)
        orbit["requivratio"].set_value(self.rp)
        orbit["requivsumfrac"].set_value((1.0 + self.rp) / self.a)
        orbit["incl"].set_value(self.inc * units.deg)
        orbit["ecc"].set_value(self.ecc)
        orbit["per0"].set_value(self.w * units.deg)

        stars = (
            result["primary"]["component"],
            result["secondary"]["component"],
        )
        stars[0]["mass"] = self.mtotal / (1.0 + self.mratio) * units.Msun
        stars[1]["mass"] = stars[0]["mass"] * self.mratio
        stars[1]["teff"].set_value(self.teff_ratio * stars[0]["teff"].quantity)
        for this_star, star_rank in zip(
            stars,
            (
                ["secondary", "primary"]
                if self.inverted
                else ["primary", "secondary"]
            ),
        ):
            this_star["gravb_bol"].set_value(self._gravdark_both[star_rank])
            this_star["ld_mode_bol"].set_value("manual")
            this_star["ld_func_bol"].set_value("linear")
            this_star["ld_coeffs_bol"].set_value(self._u_both[star_rank])
            this_star["period"].set_value(self._prot_both[star_rank])

    def secondary_flux_fraction(self):
        """Return the fraction of the flux coming from the secondary."""

        return self.teff_ratio**4 * (self.rp) ** 2

    def _set_ebeer_coefficients(self, coef, effect):
        """Set either the reflection or beaming (``effect``) coefficients."""

        target_coef = getattr(self, f"_{effect}_coef_both")

        if self.inverted:
            primary, secondary = "secondary", "primary"
        else:
            primary, secondary = "primary", "secondary"

        target_coef[primary] = coef[0]
        if len(coef) == 2:
            target_coef[secondary] = coef[1]
        else:
            target_coef[secondary] = 0.0
        # False positive
        # pylint: disable=attribute-defined-outside-init
        setattr(self, f"{effect}_coef", coef[0])
        # pylint: enable=attribute-defined-outside-init

    def set_reflection_coef(self, coef):
        """Set the reflection coefficients from the given 2-element iterable."""

        self._set_ebeer_coefficients(coef, "reflection")

    def set_beaming_coef(self, coef):
        """Set the reflection coefficients from the given 2-element iterable."""

        self._set_ebeer_coefficients(coef, "beaming")

    def calc_true_anomaly(self, times):
        """Return the true anomaly for the given binary and times."""

        mean_anom = 2 * numpy.pi * (times - self.t0_perpass) / self.per
        mean_anom = (mean_anom + numpy.pi) % (2 * numpy.pi) - numpy.pi
        true_anom = numpy.vectorize(E_to_nu)(
            numpy.vectorize(M_to_E)(mean_anom, self.ecc), self.ecc
        )

        return true_anom

    def __str__(self):
        """Human readable representation of the currently stored values."""

        # False positive
        # pylint: disable=no-member
        return (
            f"Binary: Mtot={self.mtotal}, q={self.mratio}, R1={self.rstar}, "
            f"R2/R1={self.rp}, Teff2/Teff1={self.teff_ratio} Porb={self.per}, "
            f"a={self.a}, e={self.ecc}, i={self.inc}, w={self.w}, "
            f"t0={self._t0_both}, t_perpass={self.t0_perpass}, "
            f"u={self._u_both}, LDmodel={self._limb_dark_both}, "
            f"Prot={self._prot_both}, LinLDcoef={self._linear_limbdark_both}, "
            f"GDcoef={self._gravdark_both}, reflect "
            f"coef={self._reflection_coef_both}, beaming "
            f"coef={self._beaming_coef_both}"
        )
        # pylint: enable=no-member


# pylint: enable=too-many-instance-attributes
