"""Unified interface for binary parameters needed by the likelihood function."""

import numpy
from astropy import units
import batman
import phoebe


# This is set by BATMAN
# pylint: disable=too-many-instance-attributes
class BinaryParams(batman.TransitParams):
    """Extend the batman parameters with everything needed by model."""

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

    def _init_from_phoebe(self, phoebe_binary):
        """Set BATMAN params independent of component given a PHOEBE binary."""

        orbit = phoebe_binary["orbit@component"]
        self._t0_both = {"primary": orbit["t0_supconj"].get_value(units.day)}
        self.t0_perpass = orbit["t0_perpass"].get_value(units.day)
        self.per = orbit["period"].get_value(units.day)
        self.rp = orbit["requivratio"].get_value("")
        self.a = (1.0 + self.rp) / orbit["requivsumfrac"].get_value("")
        self.inc = orbit["incl"].get_value(units.deg)
        self.ecc = orbit["ecc"].get_value("")
        self.w = 90.0 if self.ecc == 0 else orbit["per0"].get_value(units.deg)
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

        self._prot_both = {}
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

        for attr in self._per_star_attr:
            setattr(self, attr, getattr(self, f"_{attr}_both")["primary"])

    def __init__(self, *, from_phoebe=None, from_mcmc=None):
        """Set the model parameters either from PHOEBE binary or MCMC sample."""

        super().__init__()
        assert from_phoebe or from_mcmc
        self._per_star_attr = [
            "t0",
            "u",
            "limb_dark",
            "prot",
            "linear_limbdark",
            "gravdark",
        ]
        self.mratio = 1.0
        self.rstar = 1.0
        if from_phoebe:
            self._init_from_phoebe(from_phoebe)
        self.teff_ratio = 1.0
        self.t0_perpass = 0.0
        self.inverted = False
        self.reflection_coef = numpy.zeros(2)
        self.beaming_coef = numpy.zeros(2)

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
        self.reflection_coef = numpy.flip(self.reflection_coef)
        self.beaming_coef = numpy.flip(self.beaming_coef)
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

    def __str__(self):
        """Human readable representation of the currently stored values."""

        # False positive
        # pylint: disable=no-member
        return (
            f"Binary: Mtot={self.mtotal}, q={self.mratio}, R1={self.rstar}, "
            f"R2/R1={self.rp}, Teff1/Teff2={self.teff_ratio} Porb={self.per}, "
            f"a={self.a}, e={self.ecc}, i={self.inc}, w={self.w}, "
            f"t0={self.t0}, t_perpass={self.t0_perpass}, u={self._u_both}, "
            f"LDcoef={self._limb_dark_both}, Prot={self._prot_both}, "
            f"LinLDcoef={self._linear_limbdark_both}, "
            f"GDcoef={self._gravdark_both}, reflect "
            f"coef={self.reflection_coef}, beaming coef={self.beaming_coef}"
        )
        # pylint: enable=no-member


# pylint: enable=too-many-instance-attributes
