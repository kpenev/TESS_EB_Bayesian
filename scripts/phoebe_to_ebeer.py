"""Configure eBEER modulation to match a given PHOEBE binary."""

import numpy
from astropy import units
from poliastro.core.angles import E_to_nu, M_to_E

from binary_parameters import BinaryParams

_deg = numpy.pi / 180.0

def beaming(binary: BinaryParams, true_anomaly):
    """
    Beaming modulation due to doppler effect from radial velocity.

    Using equation 2 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    Args:
        binary(BinaryParams):    Data structure holding system parameters.

        true_anomaly(float or array):    The true anomaly at which to evaluate
            the beaming effect.

    Returns:
        A float or numpy array of the flux modulation due to the beaming effect
        of whichever star is currently set as primary in ``binary``.
    """

    # Full Calculation (Eq.2 from Engel et al. 2020)
    return (
        -2830
        *
        binary.mratio / (1 + binary.mratio) * binary.mtotal**(1/3)
        *
        binary.per**(-1/3)
        *
        numpy.sin(binary.inc * _deg)
        *
        (
            numpy.cos(binary.w * _deg + true_anomaly)
            /
            numpy.sqrt(1 - binary.ecc**2)
        )
    )


#Trying to reduce variables leads to worse readability
#pylint: disable=too-many-locals
def ellipticity(binary: BinaryParams, true_anomaly):
    """
    Ellipsoidal modulation from tidal distortion of stellar shape.

    Using equations 3-5 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    Args:
        binary(BinaryParams):    Data structure holding system parameters

        true_anomaly(float or array):    The true anomaly at which to evaluate
            the beaming effect.

    Returns:
        A float or numpy array of the flux modulation due to the ellipsoidal
        effect
    """

    mtotal = binary.mtotal
    mratio = binary.mratio
    star_term = (
        (1 + binary.ecc * numpy.cos(true_anomaly))
        /
        (1 - binary.ecc**2)
        *
        binary.rstar
    )**(3) / (mtotal * binary.per**2)

    #pylint: disable=invalid-name
    u = binary.linear_limbdark
    #pylint: enable=invalid-name
    tau = binary.gravdark
    orb_angle = binary.w * _deg + true_anomaly

    # Eqs 4&5 from Engel et al. 2020
    alpha_e1 = (15 * u * (2 + tau)) / (32 * (3 - u))
    alpha_e2 = (3 * (15 + u) * (1 + tau)) / (20 * (3 - u))
    alpha_e2b = (15 * (1 - u) * (3 + tau)) / (64 * (3 - u))
    alpha_e0 = alpha_e2 / 9
    alpha_e0b = 3 * alpha_e2b / 20
    alpha_e3 = 5 * alpha_e1 / 3
    alpha_e4 = 7 * alpha_e2b / 4

    sin_inc = numpy.sin(binary.inc * _deg)

    # Full Calculation (Eq.3 from Engel et al. 2020)
    return (
        (
            13435 * 2 * alpha_e0
            * (2 - 3 * sin_inc**2)
            * (1.0 + mratio) / (mtotal * binary.prot**2)
            * binary.rstar**3
        )
        +
        (
            13435 * 3 * alpha_e0
            * (2 - 3 * sin_inc**2)
            * mratio * star_term
        )
        +
        (
            759 * alpha_e0b
            * (8 - 40 * sin_inc**2 + 35 * sin_inc**4)
            * mratio * star_term**(5/3)
        )
        +
        (
            3194 * alpha_e1
            * (4 * sin_inc - 5 * sin_inc**3)
            * mratio * star_term**(4/3)
            * numpy.sin(orb_angle)
        )
        +
        (
            13435 * alpha_e2
            * sin_inc**2
            * mratio * star_term
            * numpy.cos(2 * orb_angle)
        )
        +
        (
            759 * alpha_e2b
            * (6 * sin_inc**2 - 7 * sin_inc**4)
            * mratio * star_term**(5/3)
            * numpy.cos(2 * orb_angle)
        )
        +
        (
            3194 * alpha_e3
            * sin_inc**3
            * mratio * star_term**(4/3)
            * numpy.sin(3 * orb_angle)
        )
        +
        (
            759 * alpha_e4
            * sin_inc**4
            * mratio * star_term**(5/3)
            * numpy.cos(4 * orb_angle)
        )
    )
#pylint: enable=too-many-locals


def reflection(binary: BinaryParams, true_anomaly):
    """Modulation from stellar radiation reflected by companion.

    Using equations 4 & 6 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    Args:
        binary(BinaryParams):    Data structure holding system parameters

        true_anomaly(float or array):    The true anomaly at which to evaluate
            the beaming effect.

    Returns:
        A float or numpy array of the flux modulation due to the reflection
        effect
    """

    orb_angle = binary.w * _deg + true_anomaly

    # Eq.4 from Engel et al. 2020
    beta = (1 + binary.ecc * numpy.cos(true_anomaly)) / (1 - binary.ecc**2)
    sin_inc = numpy.sin(binary.inc * _deg)

    # Full Calculation (Eq.6 from Engel et al. 2020)
    return (
        56514
        * binary.mtotal**(-2/3)
        * binary.per**(-4/3) * (beta * binary.rp * binary.rstar)**2
        * (
            0.64
            - sin_inc * numpy.sin(orb_angle)
            + 0.18 * sin_inc**2 * (1 - numpy.cos(2 * orb_angle))
        )
    )


def get_ebeer_lc(phoebe_binary,
                 times,
                 alpha_beam1: float,
                 alpha_beam2: float,
                 alpha_refl1: float,
                 alpha_refl2: float,
                 secondary_flux_fraction=None):
    """Total flux modulation from eBEER effects.

    Using equations 1 & 7 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    phoebe_binary must have ld_mode_bol set to 'manual' and ld_func_bol set
    to 'linear'. Assumes linear limb darkening (u) and gravity darkening
    (tau) coeffs are input into PHOEBE binary before eBEER model is computed.

    Args:
        phoebe_binary(PHOEBE bundle): phoebe system to generate lightcurve for
        times(float or numpy array): time(s) in DAYS which the model will
                                       be computed for
        alpha_beam1: beaming scaling factor coefficient of primary (to be tuned
                       using MCMC)
        alpha_beam2: beaming scaling factor coefficient of secondary (to be
                       tuned using MCMC)
        alpha_refl1: reflection scaling factor coefficient of primary (to be
                       tuned using MCMC)
        alpha_refl2: reflection scaling factor coefficient of secondary (to be
                       tuned using MCMC)
        secondary_flux_fraction(str or float or None):
            relative flux fraction of secondary compared to primary
            Options:
                        None (Default): Calculated based on component
                                          temperatures and masses
                        float: relative flux fraction entered manually
                        'split': returns primary and secondary
                                       components separately

    Returns:
        Total eBEER flux modulation. If secondary_flux_fraction is 'split'
        function return is a tuple (MeBEER1, MeBEER2) which are the calculated
        eBEER modulations (float or numpy array) from the primary and secondary
        respectively. Otherwise return is a single float or numpy array which
        is the combined eBEER flux modulation from both stars
    """

    binary = BinaryParams(from_phoebe=phoebe_binary)

    orbit = phoebe_binary['orbit@component']
    mean_anom = (
        2*numpy.pi * (times - orbit['t0_perpass'].get_value(units.day))
        /
        binary.per
    )
    mean_anom = (mean_anom + numpy.pi) % (2*numpy.pi) - numpy.pi
    true_anom = numpy.vectorize(E_to_nu)(
        numpy.vectorize(M_to_E)(mean_anom, binary.ecc),
        binary.ecc
    )

    primary_flux = (alpha_beam1 * beaming(binary, true_anom)
                    +
                    alpha_refl1 * reflection(binary, true_anom)
                    +
                    ellipticity(binary, true_anom))

    binary.swap_components()

    secondary_flux = (alpha_beam2 * beaming(binary, true_anom)
                      +
                      alpha_refl2 * reflection(binary, true_anom)
                      +
                      ellipticity(binary, true_anom))


    if secondary_flux_fraction is None:
        secondary_flux_fraction = (
            phoebe_binary['secondary@teff'].get_value('K')
            /
            phoebe_binary['primary@teff'].get_value('K')
        )**4 * (binary.rp)**2
    elif secondary_flux_fraction == 'split':
        return (primary_flux, secondary_flux)

    return (
        (primary_flux + secondary_flux_fraction * secondary_flux)
        /
        (1 + secondary_flux_fraction)
    )
