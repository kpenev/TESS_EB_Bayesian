"""Unified interface for binary parameters needed by the likelihood function."""

import numpy
from astropy import units
import batman

class BinaryParms(batman.TransitParams):
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

        central_transits_rhs = (
            ecosw * (1.0 - e_square)**0.5 / (1.0 - esinw**2)
            +
            numpy.arctan(ecosw / (1.0 - e_square)**0.5)
        )

        if coti:
            coti2 = coti**2
            inclination_correction = 0.5 * ecosw * coti2 * (
                1.0
                /
                (
                    (1.0 + esinw)**3
                    +
                    (1.0 + esinw) * (esinw + esinw**2 + 3.0 * ecosw**2) * coti2
                )
                +
                1.0
                /
                (
                    (1.0 - esinw)**3
                    +
                    (1.0 - esinw) * (-esinw + esinw**2 + 3.0 * ecosw**2) * coti2
                )
            )
        else:
            inclination_correction = 0.0

        return (central_transits_rhs + inclination_correction) / numpy.pi + 0.5


    def _init_from_phoebe(self, phoebe_binary, dataset='lc01'):
        """Set BATMAN params independent of component given a PHOEBE binary."""

        orbit = phoebe_binary['orbit@component']
        self._t0_both = {'primary': orbit['t0_supconj'].get_value(units.day)}
        self.per = orbit['period'].get_value(units.day)
        self.rp = orbit['requivratio'].get_value('')
        self.a = ((1.0 + self.rp)
                  /
                  orbit['requivsumfrac'].get_value(''))
        self.inc = orbit['incl'].get_value(units.deg)
        self.ecc = orbit['ecc'].get_value('')
        self.w = (
            90.0 if self.ecc == 0 else orbit['per0'].get_value(units.deg)
        )
        #(orbit['long_an'].get_value(units.deg)
        #                   +
        #                   orbit['per0'].get_value(units.deg)
        #                   +
        #                   90.0)

        self._t0_both['secondary'] = self._t0_both['primary'] + (
            self.per
            *
            self._eclipse_phase_difference(
                orbit['esinw'].get_value(''),
                orbit['ecosw'].get_value(''),
            )
        )

        for component in ['primary', 'secondary']:
            self._limb_dark_both[
                component
            ] = phoebe_binary['ld_func'][dataset][component].get_value()
            phoebe_coefs = phoebe_binary.compute_ld_coeffs(dataset=dataset,
                                                           component=component)
            assert len(phoebe_coefs) == 1
            for value in phoebe_coefs.values():
                self._u_both[component] = value
        self.t0 = self._t0_both['primary']
        self.u = self._u_both['primary']
        self.limb_dark = self._limb_dark_both['primary']


    def __init__(self, *, from_phoebe=None, from_mcmc=None):
        """Set the model parameters either from PHOEBE binary or MCMC sample."""

        super().__init__()
        assert from_phoebe or from_mcmc
        self._limb_dark_both = {}
        self._u_both = {}
        if from_phoebe:
            self._init_from_phoebe(from_phoebe)
        self.inverted = False


    def swap_components(self):
        """Swap which star is considered primary vs secondary."""

        component = 'primary' if self.inverted else 'secondary'
        self.t0 = self._t0_both[component]
        self.u = self._u_both[component]
        self.limb_dark = self._limb_dark_both[component]
        self.rp = 1.0 / self.rp
        self.a *= self.rp
        self.w = (self.w + 180.0) % 360.0
