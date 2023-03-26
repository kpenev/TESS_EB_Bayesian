#!/usr/bin/env python3

"""Compare BATMAN and PHOEBE transit models."""

from matplotlib import pyplot
import numpy
from astropy import units

import phoebe

from phoebe_to_batman import get_batman_lc

if __name__ == '__main__':
    phoebe_binary = phoebe.default_binary()
    phoebe_binary.add_dataset('lc',
                              times=numpy.linspace(0.0, 1.0, 101),
                              dataset='lc01')
    orbit = phoebe_binary['orbit@component']
    orbit['ecc'].set_value(0.4)
    orbit['per0'].set_value(-80.0 * units.deg)
    orbit['incl'].set_value(60.0 * units.deg)


    for component in ['primary', 'secondary']:
        phoebe_binary['ld_mode@lc01@' + component].set_value('lookup')
        phoebe_binary['ld_func@lc01@' + component].set_value('quadratic')

    phoebe_binary.run_compute()
    plot_times = (
        phoebe_binary.get('times@lc01@phoebe01@latest@lc@model').get_value()
    )
    phoebe_lc = (
        phoebe_binary.get('fluxes@lc01@phoebe01@latest@lc@model').get_value()
    )
    batman_lc = get_batman_lc(phoebe_binary)

    pyplot.subplot(211)
    pyplot.plot(plot_times, phoebe_lc, 'og')
    pyplot.plot(plot_times, batman_lc, 'or')
    pyplot.subplot(212)
    pyplot.plot(plot_times, phoebe_lc - batman_lc, 'ok')

    pyplot.show()
