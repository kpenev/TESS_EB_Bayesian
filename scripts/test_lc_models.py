#!/usr/bin/env python3

"""Compare BATMAN and PHOEBE transit models."""

from matplotlib import pyplot
import numpy

import batman
import phoebe
from astropy import units

if __name__ == '__main__':
    phoebe_binary = phoebe.default_binary()
    phoebe_binary.add_dataset('lc',
                              times=numpy.linspace(-0.1, 0.1, 11),
                              dataset='lc01')
    for component in ['primary', 'secondary']:
        phoebe_binary['ld_mode@lc01@primary'].set_value('lookup')
        phoebe_binary['ld_func@lc01@' + component].set_value('quadratic')

    phoebe_binary.run_compute()
    pyplot.plot(
        phoebe_binary.get('times@lc01@phoebe01@latest@lc@model').get_value(),
        phoebe_binary.get('fluxes@lc01@phoebe01@latest@lc@model').get_value(),
        'ok'
    )
    pyplot.show()
