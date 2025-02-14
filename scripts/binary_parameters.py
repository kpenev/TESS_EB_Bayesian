"""Unified interface for binary parameters needed by the likelihood function."""

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


# This is set by BATMAN
# pylint: disable=too-many-instance-attributes
class BinaryParams(batman.TransitParams):
    """Extend the batman parameters with everything needed by model."""

    _cmd_interpolators = tuple(
        (photsys, CMDInterpolator(cmd_data_fname.format(photsys=photsys)))
        for photsys in ["panstarss1", "2mass_spitzer_wise"]
    )

    meh_range = _cmd_interpolators[0][1].get_range("MH")
    log_age_range = _cmd_interpolators[0][1].get_range("logAge")

    def _eclipse_phase_difference(self, mid_transit=False):
        """
        Calculate the phase difference between secondary and primary eclipse.

        Uses equation 31 (and correction for non-central transits) from
        Sterne 1940 (PNAS 26, 36):

        `https://ui.adsabs.harvard.edu/abs/1940PNAS...26...36S/abstract`_

        Note that for BATMAN purposes the inclination correction should not be
        applied since the parameter used is the time of conjunction, not
        mid-transit.

        Args:
            mid_transit(bool):    If True, the phase of mid-transit is used. If
                false, the time of conjunction.

        Returns:
            float:
                The fraction of the orbital period that elapses between primary
                and secondary eclipses.
        """

        e_square = self.ecc**2
        esinw = self.ecc * numpy.sin(self.w * numpy.pi / 180.0)
        ecosw = self.ecc * numpy.cos(self.w * numpy.pi / 180.0)

        central_transits_rhs = ecosw * (1.0 - e_square) ** 0.5 / (
            1.0 - esinw**2
        ) + numpy.arctan(ecosw / (1.0 - e_square) ** 0.5)

        if mid_transit:
            coti2 = numpy.tan(self.inc * numpy.pi / 180.0) ** (-2)
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
            self.per * self._eclipse_phase_difference()
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

            # False positive
            # pylint: disable=no-member
            self._prot_both[component] = star["component@period"].get_value(
                units.day
            )
            # pylint: enable=no-member
            self._linear_limbdark_both[component] = self._u_both[component]
            self._gravdark_both[component] = star["gravb_bol"].get_value("")
        self._set_per_star()

    def set_from_mcmc(self, sample_params):
        """
        Set the binary parameters from an MCMC sample.

        Args:
            sample_params:    Object with attributes specifying the phyisical
                parameters of the system being sampled. See `SampleParams` in
                `log_likelihood.py` for the attribute names.

        Returns:
            None
        """

        for param in ["per", "ecc", "w"]:
            setattr(self, param, getattr(sample_params, param))

        mprimary = sample_params.mtotal / (1.0 + sample_params.mratio)
        interp_kwargs = {
            "MH": sample_params.meh,
            "logAge": 9.0 + numpy.log10(sample_params.age_gyr),
        }

        interpolated = {}
        for photsys, interpolator in self._cmd_interpolators:
            for component, mass in [
                ("primary", mprimary),
                ("secondary", mprimary * sample_params.mratio),
            ]:
                if component not in interpolated:
                    interpolated[component] = interpolator(
                        ("Mass", "logL", "logTe") + self._passbands[photsys],
                        Mini=mass,
                        **interp_kwargs,
                    )
                else:
                    interpolated[component] = numpy.append(
                        interpolated[component],
                        interpolator(
                            self._passbands[photsys],
                            Mini=mprimary,
                            **interp_kwargs,
                        ),
                    )

        self.mtotal = interpolated["primary"][0] + interpolated["secondary"][0]
        self.mratio = interpolated["secondary"][0] / interpolated["primary"][0]
        radii = {}
        for component, comp_interp in interpolated.items():
            radii[component] = (
                10.0 ** (comp_interp[1] / 2.0 - 2.0 * comp_interp[2])
                / (2.0 * units.K**2)
                * numpy.sqrt(units.L_sun / (numpy.pi * constants.sigma_sb))
            )
            try:
                self._gravdark_both[component] = self._gravdark_interp(
                    logg=numpy.log10(
                        (
                            constants.G
                            * comp_interp[0]
                            * units.M_sun
                            / radii[component] ** 2
                        ).to_value(units.cm / units.s**2)
                    ),
                    Z=sample_params.meh,
                    logTeff=comp_interp[2],
                )
            except ValueError:
                self._gravdark_both[component] = numpy.nan
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

        self.inc = (
            numpy.arccos(sample_params.primary_impact_param / self.a)
            * 180.0
            / numpy.pi
        )
        # If we wish to use true impact parameter:
        # self.inc = numpy.arccos(
        #    sample_params.primary_impact_param
        #    / (
        #        self.a
        #        * numpy.sqrt(
        #            (numpy.cos(ecc_anom) - self.ecc) ** 2
        #            + ((1 - e) * numpy.sin(ecc_anom)) ** 2
        #        )
        #    )
        # )

        self._t0_both["primary"] = sample_params.eclipse_time
        self._t0_both["secondary"] = self._t0_both["primary"] + (
            self.per * self._eclipse_phase_difference()
        )
        self.t0_perpass = (
            self._t0_both["primary"]
            - E_to_M(
                nu_to_E(numpy.pi / 2 - self.w * numpy.pi / 180, self.ecc),
                self.ecc,
            )
            / (2.0 * numpy.pi)
            * self.per
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
        self.absmag = -2.5 * numpy.log10(
            10.0 ** (-interpolated["primary"][3:] / 2.5)
            + 10.0 ** (-interpolated["secondary"][3:] / 2.5)
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
        self.absmag = None
        for attr in self._per_star_attr:
            setattr(self, f"_{attr}_both", {"primary": None, "secondary": None})

        self._passbands = {
            "panstarss1": tuple(b + "P1mag" for b in "grizy"),
            "2mass_spitzer_wise": tuple(
                b + "mag" for b in ["J", "H", "Ks", "W1", "W2"]
            ),
        }

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
            # False positive
            # pylint: disable=no-member
            this_star["period"].set_value(self._prot_both[star_rank])
            # pylint: enable=no-member

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
            f"coef={self._reflection_coef_both}, "
            f"beaming coef={self._beaming_coef_both}, "
            f"Absolute magnitudes={self.absmag}"
        )
        # pylint: enable=no-member


# pylint: enable=too-many-instance-attributes
