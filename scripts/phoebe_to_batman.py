"""Configure BATMAN transits to match a given PHOEBE binary."""

import numpy
import batman
from astropy import units

def get_common_batman_params(phoebe_binary):
    """Get BATMAN parameters independent of component given a PHOEBE binary ."""

    batman_params = batman.TransitParams()
    orbit = phoebe_binary['orbit@component']
    batman_params.t0 = orbit['t0_supconj'].get_value(units.day)
    batman_params.per = orbit['period'].get_value(units.day)
    batman_params.rp = orbit['requivratio'].get_value('')
    batman_params.a = ((1.0 + batman_params.rp)
                       /
                       orbit['requivsumfrac'].get_value(''))
    batman_params.inc = orbit['incl'].get_value(units.deg)
    batman_params.ecc = orbit['ecc'].get_value('')
    batman_params.w = (
        90.0 if batman_params.ecc == 0 else orbit['per0'].get_value(units.deg)
    )
#    (orbit['long_an'].get_value(units.deg)
#                       +
#                       orbit['per0'].get_value(units.deg)
#                       +
#                       90.0)

    return batman_params


def eclipse_phase_difference(esinw, ecosw, coti=0.0):
    """
    Calculate the phase difference between secondary and primary eclipse.

    Uses equation 31 (and correction for non-central transits) from Sterne 1940
    (PNAS 26, 36):

    `https://ui.adsabs.harvard.edu/abs/1940PNAS...26...36S/abstract`_

    Note that for BATMAN purposes the inclination correction should not be
    applied since the parameter used is the time of conjunction, not
    mid-transit.

    Args:
        esinw(float):    Eccentricity times sin of longitude of periapsis.

        ecosw(float):    Eccentricity times cos of longitude of periapsis.

        coti(float):    1/tan of the inclination angle. Leave zero to disable
                        the correction to Eq. 31 in Sterne 1940

    Returns:
        float:
            The fraction of the orbital period that elapses between primary and
            secondary eclipses.
    """

    e2 = esinw**2 + ecosw**2

    central_transits_rhs = (
        ecosw * (1.0 - e2)**0.5 / (1.0 - esinw**2)
        +
        numpy.arctan(ecosw / (1.0 - e2)**0.5)
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


def set_batman_component_params(batman_params,
                                phoebe_binary,
                                component,
                                dataset='lc01'):
    """
    Update the parameters of BATMAN model for one of the PHOEBE stars.

    Args:
        transit_params(batman.TransitParams):    The transit parameters to
            update.

        phoebe_binary:    The PHOEBE binary to get the configuration from.

        component(str):    Should be either ``'primary'`` or ``'secondary'``,
            indicating which star to get the limb darkening for.

    Returns:
        None
    """

    if component == 'secondary':
        batman_params.rp = 1.0 / batman_params.rp
        batman_params.a *= batman_params.rp

        orbit = phoebe_binary['orbit@component']
        batman_params.t0 += (
            batman_params.per
            *
            eclipse_phase_difference(
                orbit['esinw'].get_value(''),
                orbit['ecosw'].get_value(''),
            )
        )

        batman_params.w += 180.0

    batman_params.limb_dark = (
        phoebe_binary['ld_func'][dataset][component].get_value()
    )
    phoebe_coefs = phoebe_binary.compute_ld_coeffs(dataset=dataset,
                                                   component=component)
    assert len(phoebe_coefs) == 1
    for value in phoebe_coefs.values():
        batman_params.u = value


def get_batman_lc(phoebe_binary, dataset='lc01'):
    """Use BATMAN to approximate the LC of the given binary."""

    params = get_common_batman_params(phoebe_binary)
    set_batman_component_params(params, phoebe_binary, 'primary', dataset)

    model = batman.TransitModel(
        params,
        phoebe_binary.get('times@'
                          +
                          dataset
                          +
                          '@phoebe01@latest@lc@model').get_value()
    )

    flux = model.light_curve(params)

    set_batman_component_params(params, phoebe_binary, 'secondary', dataset)
    flux_ratio = (
        phoebe_binary['secondary@teff'].get_value('K')
        /
        phoebe_binary['primary@teff'].get_value('K')
    )**4 / params.rp**2
    flux += model.light_curve(params) * flux_ratio

    return flux
