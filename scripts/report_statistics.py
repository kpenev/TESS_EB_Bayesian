#!/usr/bin/env python3


from os import path

import numpy
from astropy.io import fits
import pandas

def report_w19_lurie_crossmatch(data_dir):
    """Count how many KIC from W19 also have measured spin period by Lurie."""

    with fits.open(path.join(data_dir, 'lurie.fits'), 'readonly') as lurie_f:
        lurie_data = lurie_f[1].data[:]

    w19_data = pandas.read_csv(path.join(data_dir, 'w19_maxlike_pars.dat'),
                               sep='\s+',
                               index_col='#KIC')

    kic_with_spin = lurie_data['KIC'][lurie_data['Class'] == 'sp']
    print('%d W19 EBs also have spin'
          %
          w19_data.index.intersection(kic_with_spin).size)


def get_hemisphere(sectors):
    """Return hemisphere list of sectors belongs to (0-south, 1-north)."""

    print('Sectors: ' + repr(sectors))
    sectors = numpy.array([int(s) for s in sectors.rstrip(',').split(',')],
                          dtype=int)
    if sectors.min() <= 13 or sectors.min() > 26:
        if not (
                numpy.logical_or(
                    sectors <= 13,
                    numpy.logical_and(sectors > 26, sectors <= 39)
                ).all()
        ):
            raise ValueError('Not all sectors from same hemisphere'
                             +
                             repr(sectors))
        return 0
    else:
        if not numpy.logical_and(sectors > 13, sectors <= 26).all():
            raise ValueError('Not all sectors from same hemisphere'
                             +
                             repr(sectors))

        return 1


def report_prsa_statistics(data_dir):
    """Report how many Prsa binaries fall in various categories."""

    with fits.open(path.join(data_dir, 'prsa_ebs.fits'), 'readonly') as prsa_f:
        data = prsa_f[1].data[:]

    data = data[data['Sectors'] != '']
    hemispheres = numpy.array(
        [get_hemisphere(s) for s in data['Sectors']]
    )
    print('%d / %d Prsa EBs from S / N hemisphere'
          %
          ((1-hemispheres).sum(), hemispheres.sum()))
    detached = data['Morph'] < 0.5
    data = data[detached]
    hemispheres = hemispheres[detached]

    print('%d / %d Detached Prsa EBs from S / N hemisphere'
          %
          ((1-hemispheres).sum(), hemispheres.sum()))

    for mode in ['pf', '2g']:
        deep = numpy.logical_and(
            numpy.logical_and(
                numpy.isfinite(data['Dp-' + mode]),
                data['Dp-' + mode] > 0.05
            ),
            data['Ds-' + mode] > 0.01
        )
        print('From %s: %d / %d deep detached EBs from S / N hemisphere'
              %
              (mode, (1-hemispheres[deep]).sum(), hemispheres[deep].sum()))



def main():
    """Avoid polluting the global namespace."""

    data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                         'data')
    report_w19_lurie_crossmatch(data_dir)
    report_prsa_statistics(data_dir)


if __name__ == '__main__':
    main()
