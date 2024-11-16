"""Configure eBEER modulation to match a given PHOEBE binary."""

import numpy
from scipy.linalg import lstsq
from scipy.optimize import minimize_scalar
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
        * binary.mratio
        / (1 + binary.mratio)
        * binary.mtotal ** (1 / 3)
        * binary.per ** (-1 / 3)
        * numpy.sin(binary.inc * _deg)
        * (
            numpy.cos(binary.w * _deg + true_anomaly)
            / numpy.sqrt(1 - binary.ecc**2)
        )
    )


# Trying to reduce variables leads to worse readability
# pylint: disable=too-many-locals
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
        / (1 - binary.ecc**2)
        * binary.rstar
    ) ** (3) / (mtotal * binary.per**2)

    # pylint: disable=invalid-name
    u = binary.linear_limbdark
    # pylint: enable=invalid-name
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
            13435
            * 2
            * alpha_e0
            * (2 - 3 * sin_inc**2)
            * (1.0 + mratio)
            / (mtotal * binary.prot**2)
            * binary.rstar**3
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
    )


# pylint: enable=too-many-locals


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
        * binary.mtotal ** (-2 / 3)
        * binary.per ** (-4 / 3)
        * (beta * binary.rp * binary.rstar) ** 2
        * (
            0.64
            - sin_inc * numpy.sin(orb_angle)
            + 0.18 * sin_inc**2 * (1 - numpy.cos(2 * orb_angle))
        )
    )


def calc_true_anomaly(binary, times):
    """Return the true anomaly for the given binary and times."""

    mean_anom = 2 * numpy.pi * (times - binary.t0_perpass) / binary.per
    mean_anom = (mean_anom + numpy.pi) % (2 * numpy.pi) - numpy.pi
    true_anom = numpy.vectorize(E_to_nu)(
        numpy.vectorize(M_to_E)(mean_anom, binary.ecc), binary.ecc
    )

    return true_anom


# Lots of cases to handle
# pylint: disable=too-many-branches
def fit_ebeer_coefficients(
    binary,
    true_anomaly,
    fluxdiff_to_fit,
    result="fluxdiff",
    *,
    include_beaming=True,
    include_ellipticity=True,
    include_reflection=True,
):
    """
    Find best fit beaming and reflection coefficients and return coefs and LC.

    Updates the parameters in ``binary`` to their best fit values.

    Args:
        binary(BinaryParams):    The parameters of the binary being modeled.

        true_anomaly(array):    The true anomaly values where the flux is known.

        fluxdiff_to_fit(array):    The flux difference at each entry of
            ``true_anomaly``.

        result(str):    What to return. Should be one of: ``'fluxdiff'``,
            ``'rms'``, ``'residues'``

        include_*:    For testing purposes it is sometimes useful to
            disable some effects.

    Returns:
        One of the following:
            array: The best fit eBEER lightcurve at ``true_anomaly`` if
            ``result`` is ``'fluxdiff'``.

            float: The RMS residuals between ``fluxdiff_to_fit`` and best fit
                lightcurve if ``result`` is ``'rms'``

            array: The residues if ``result`` is neither of the above.
    """

    secondary_flux_fraction = binary.teff_ratio**4 * (binary.rp) ** 2

    num_coef = 0
    if include_beaming:
        num_coef += 2
    if include_reflection:
        num_coef += 2

    lhs_matrix = numpy.empty((true_anomaly.size, num_coef))
    if include_ellipticity:
        ellip_fluxdiff = ellipticity(binary, true_anomaly)

    if include_reflection:
        lhs_matrix[:, 0] = reflection(binary, true_anomaly)
        beaming_ind = 2
    else:
        beaming_ind = 0
    if include_beaming:
        lhs_matrix[:, beaming_ind] = beaming(binary, true_anomaly)

    binary.swap_components()

    if include_reflection:
        lhs_matrix[:, 1] = secondary_flux_fraction * reflection(
            binary, true_anomaly
        )
    if include_beaming:
        lhs_matrix[:, beaming_ind + 1] = secondary_flux_fraction * beaming(
            binary, true_anomaly
        )

    if include_ellipticity:
        ellip_fluxdiff = (
            ellip_fluxdiff
            + secondary_flux_fraction * ellipticity(binary, true_anomaly)
        ) / (1.0 + secondary_flux_fraction)
        rhs = fluxdiff_to_fit - ellip_fluxdiff
    if num_coef == 0:
        if result == "fluxdiff":
            return ellip_fluxdiff
        if result == "rms":
            return numpy.mean(rhs**2) ** 0.5
        return rhs

    lhs_matrix /= 1.0 + secondary_flux_fraction
    fit_result = lstsq(lhs_matrix, rhs)
    if include_reflection:
        binary.reflection_coef = fit_result[0][:2]
    if include_beaming:
        binary.beaming_coef = fit_result[0][beaming_ind : beaming_ind + 2]
    print(
        f"Residues: {fit_result[1]!r}\nRank: {fit_result[2]}\nSingular "
        f"vals:{fit_result[3]!r}"
    )
    if result == "fluxdiff":
        return lhs_matrix.dot(fit_result[0])
    if result == "rms":
        result = numpy.mean(fit_result[1] ** 2) ** 0.5
        print(f"Returning: {result!r}")
        return result
    return fit_result[1]


# pylint: enable=too-many-branches


def fit_ebeer_time_and_coef(binary, times, fluxdiff_to_fit, **fit_coef_kwargs):
    """
    Fit periapsis time and eBEER coefficients to best match ``fluxdiff_to_fit``.

    Updates the parameters in ``binary`` to their best fit values.

    Args:
        binary(BinaryParams):    The parameters of the binary being modeled.

        fluxdiff_to_fit(array):    The flux difference at each entry of
            ``true_anomaly``.

        fit_coef_kwargs:    Keyword arguments to pass to
            `fit_ebeer_coefficients()`.

    Returns:
        See `fit_ebeer_coefficients()`.
    """

    def to_minimize(t0_perpass):
        """The function to minimize."""

        binary.t0_perpass = t0_perpass
        result = fit_ebeer_coefficients(
            binary,
            calc_true_anomaly(binary, times),
            fluxdiff_to_fit,
            result="rms",
            **fit_coef_kwargs,
        )
        print(f"To minimize result: {result!r}")
        return result

    orig_result = fit_coef_kwargs.pop("result", "fluxdiff")
    print(f"Minimizing with bounds: {(0, binary.per)!r}")
    fit_result = minimize_scalar(to_minimize, bounds=(0, binary.per))
    print("Minimize result: {fit_result!r}")
    binary.t0_perpass = fit_result.x
    fit_coef_kwargs["result"] = orig_result
    return fit_ebeer_coefficients(
        binary,
        calc_true_anomaly(binary, times),
        fluxdiff_to_fit,
        **fit_coef_kwargs,
    )


def get_ebeer_lc(
    binary, times, secondary_flux_fraction=None, disable_ellipticity=False
):
    """Total flux modulation from eBEER effects.

    Using equations 1 & 7 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    phoebe_binary must have ld_mode_bol set to 'manual' and ld_func_bol set
    to 'linear'. Assumes linear limb darkening (u) and gravity darkening
    (tau) coeffs are input into PHOEBE binary before eBEER model is computed.

    Args:
        binary(BinaryParams):    phoebe system to generate lightcurve for

        times(float or numpy array):    time(s) in DAYS which the model will
                                       be computed for

        alpha_beam1:    beaming scaling factor coefficient of primary (to be
            tuned using MCMC)

        alpha_beam2:    beaming scaling factor coefficient of secondary (to be
            tuned using MCMC)

        alpha_refl1:    reflection scaling factor coefficient of primary (to be
            tuned using MCMC)

        alpha_refl2:    reflection scaling factor coefficient of secondary (to
            be tuned using MCMC)

        secondary_flux_fraction(str or float or None):    relative flux fraction
            of secondary compared to primary
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

    true_anom = calc_true_anomaly(binary, times)
    alpha_ellip = 0 if disable_ellipticity else 1
    primary_flux = (
        binary.beaming_coef[0] * beaming(binary, true_anom)
        + binary.reflection_coef[0] * reflection(binary, true_anom)
        + alpha_ellip * ellipticity(binary, true_anom)
    )

    if secondary_flux_fraction is None:
        secondary_flux_fraction = binary.teff_ratio**4 * (binary.rp) ** 2

    binary.swap_components()

    secondary_flux = (
        binary.beaming_coef[1] * beaming(binary, true_anom)
        + binary.reflection_coef[1] * reflection(binary, true_anom)
        + alpha_ellip * ellipticity(binary, true_anom)
    )

    if secondary_flux_fraction == "split":
        return (primary_flux, secondary_flux)

    return (primary_flux + secondary_flux_fraction * secondary_flux) / (
        1 + secondary_flux_fraction
    )
