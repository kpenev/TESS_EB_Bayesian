"""Configure BATMAN transits to match a given PHOEBE binary."""

import batman
import phoebe
from astropy import units

def get_batman_params_no_ld(phoebe_binary, dataset='lc01'):
    """Get BATMAN parameters given a PHOEBE binary except limb darkening."""

    batman_params = batman.TransitParams()
    orbit = phoebe_binary['orbit@component']
    batman_params.t0 = orbit['t0_ref'].get_value(units.day)
    batman_params.per = orbit['period'].get_value(units.day)
    batman_params.rp =  orbit['requivratio'].get_value('')
    batman_params.a = ((1.0 + batman_params.rp)
                       /
                       orbit['requivsumfrac'].get_value(''))
    batman_params.inc = orbit['incl'].get_value(units.deg)
    batman_params.ecc = orbit['ecc'].get_value('')
    batman_params.w = (orbit['long_an'].get_value(units.deg)
                +
                orbit['per0'].get_value(units.deg))
    return batman_params

def set_batman_ld_params(batman_params, phoebe_binary, component):
    """
    Set the limb darkering parameters of BATMAN model per a PHOEBE star.

    Args:
        transit_params(batman.TransitParams):    The transit parameters to
            update.

        phoebe_binary:    The PHOEBE binary to get the configuration from.

        component(str):    Should be either ``'primary'`` or ``'secondary'``,
            indicating which star to get the limb darkening for.
    """

    batman_params.limb_dark = phoebe_binary['ld_func@lc01'][component].get_value()
    phoebe_coefs = phoebe_binary.compute_ld_coeffs(dataset='lc01',
                                                   component=component)
    assert len(phoebe_coefs) == 1
    for key, value in coefs.items():
        batman_params.u = value
