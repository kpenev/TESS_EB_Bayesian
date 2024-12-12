"""Define class for modeling out of eclipse variability."""

import numpy

from binary_parameters import BinaryParams

_deg = numpy.pi / 180.0


class EBEERBinary(BinaryParams):
    """Implement out of eclipse per Engel et al. 2020 (MNRAS, 497, 4884)."""

    def beaming(self, true_anomaly):
        """
        Beaming modulation of primary due to doppler effect.

        Using equation 2 from Engel et al. 2020 (MNRAS, 497, 4884):
        `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

        Args:
            true_anomaly(float or array):    The true anomaly at which to
                evaluate the beaming effect.

        Returns:
            A float or numpy array of the flux modulation due to the beaming
            effect of whichever star is currently set as primary.
        """

        # Full Calculation (Eq.2 from Engel et al. 2020)
        return (
            self.beaming_coef
            * (
                -2830
                * self.mratio
                / (1 + self.mratio)
                * self.mtotal ** (1 / 3)
                * self.per ** (-1 / 3)
                * numpy.sin(self.inc * _deg)
                * (
                    numpy.cos(self.w * _deg + true_anomaly)
                    + self.ecc * numpy.cos(self.w * _deg)
                )
                / numpy.sqrt(1 - self.ecc**2)
            )
            * 1e-6
        )

    # Trying to reduce variables leads to worse readability
    # pylint: disable=too-many-locals
    def ellipticity(self, true_anomaly):
        """
        Ellipsoidal modulation of primary from tidal distortion.

        Using equations 3-5 from Engel et al. 2020 (MNRAS, 497, 4884):
        `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

        Args:
            true_anomaly(float or array):    The true anomaly at which to
                evaluate the beaming effect.

        Returns:
            A float or numpy array of the flux modulation due to the ellipsoidal
            effect of whichever star is currently the primary.
        """

        mtotal = self.mtotal
        mratio = self.mratio
        star_term = (
            (1 + self.ecc * numpy.cos(true_anomaly))
            / (1 - self.ecc**2)
            * self.rstar
        ) ** (3) / (mtotal * self.per**2)

        # pylint: disable=invalid-name
        # pylint: disable=no-member
        u = self.linear_limbdark
        # pylint: enable=invalid-name
        tau = self.gravdark
        # pylint: enable=no-member
        orb_angle = self.w * _deg + true_anomaly

        # Eqs 4&5 from Engel et al. 2020
        alpha_e1 = (15 * u * (2 + tau)) / (32 * (3 - u))
        alpha_e2 = (3 * (15 + u) * (1 + tau)) / (20 * (3 - u))
        alpha_e2b = (15 * (1 - u) * (3 + tau)) / (64 * (3 - u))
        alpha_e0 = alpha_e2 / 9
        alpha_e0b = 3 * alpha_e2b / 20
        alpha_e3 = 5 * alpha_e1 / 3
        alpha_e4 = 7 * alpha_e2b / 4

        sin_inc = numpy.sin(self.inc * _deg)

        # Full Calculation (Eq.3 from Engel et al. 2020)
        return (
            (
                13435
                * 2
                * alpha_e0
                * (2 - 3 * sin_inc**2)
                * (1.0 + mratio)
                # pylint: disable=no-member
                / (mtotal * self.prot**2)
                # pylint: enable=no-member
                * self.rstar**3
            )
            + (13435 * 3 * alpha_e0 * (2 - 3 * sin_inc**2) * mratio * star_term)
            + (
                759
                * alpha_e0b
                * (8 - 40 * sin_inc**2 + 35 * sin_inc**4)
                * mratio
                * star_term ** (5 / 3)
            )
            + (
                3194
                * alpha_e1
                * (4 * sin_inc - 5 * sin_inc**3)
                * mratio
                * star_term ** (4 / 3)
                * numpy.sin(orb_angle)
            )
            + (
                13435
                * alpha_e2
                * sin_inc**2
                * mratio
                * star_term
                * numpy.cos(2 * orb_angle)
            )
            + (
                759
                * alpha_e2b
                * (6 * sin_inc**2 - 7 * sin_inc**4)
                * mratio
                * star_term ** (5 / 3)
                * numpy.cos(2 * orb_angle)
            )
            + (
                3194
                * alpha_e3
                * sin_inc**3
                * mratio
                * star_term ** (4 / 3)
                * numpy.sin(3 * orb_angle)
            )
            + (
                759
                * alpha_e4
                * sin_inc**4
                * mratio
                * star_term ** (5 / 3)
                * numpy.cos(4 * orb_angle)
            )
        ) * 1e-6

    # pylint: enable=too-many-locals

    def reflection(self, true_anomaly):
        """Modulation of primary light from reflected stellar radiation.

        Using equations 4 & 6 from Engel et al. 2020 (MNRAS, 497, 4884):
        `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract` but for
        primary instead of secondary for consistency with other effects.

        Args:
            true_anomaly(float or array):    The true anomaly at which to
                evaluate the beaming effect.

        Returns:
            A float or numpy array of the flux modulation due to the reflection
            effect
        """

        orb_angle = self.w * _deg + true_anomaly + 180.0 * _deg

        # Eq.4 from Engel et al. 2020
        beta = (1 + self.ecc * numpy.cos(true_anomaly)) / (1 - self.ecc**2)
        sin_inc = numpy.sin(self.inc * _deg)

        # Full Calculation (Eq.6 from Engel et al. 2020)
        return (
            self.reflection_coef
            * (
                56514
                * self.mtotal ** (-2 / 3)
                * self.per ** (-4 / 3)
                * (beta * self.rstar) ** 2
                * (
                    0.64
                    - sin_inc * numpy.sin(orb_angle)
                    + 0.18 * sin_inc**2 * (1 - numpy.cos(2 * orb_angle))
                )
            )
            * 1e-6
        )
