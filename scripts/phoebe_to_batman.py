"""Configure BATMAN transits to match a given PHOEBE binary."""

import batman

from binary_parameters import BinaryParms

def get_batman_lc(phoebe_binary,
                  times,
                  dataset='lc01',
                  secondary_flux_fraction=None):
    """Use BATMAN to approximate the LC of the given binary."""

    assert dataset == 'lc01'
    params = BinaryParms(from_phoebe=phoebe_binary)

    print('\t\t\tRunning primary batman model for: ' + repr(vars(params)))
    model = batman.TransitModel(params, times)
    flux = model.light_curve(params)
    params.swap_components()
    print('\t\t\tRunning secondary batman model for: ' + repr(vars(params)))
    model = batman.TransitModel(params, times)
    if secondary_flux_fraction is None:
        secondary_flux_fraction = (
            phoebe_binary['secondary@teff'].get_value('K')
            /
            phoebe_binary['primary@teff'].get_value('K')
        )**4 / params.rp**2
    elif secondary_flux_fraction == 'split':
        print('\t\t\tReturning separate primary and secondary LCs')
        return flux, model.light_curve(params)

    print('\t\t\tReturning combined primary and secondary LCs')
    flux += model.light_curve(params) * secondary_flux_fraction
    print('\t\t\tResult is ready')
    return flux
