#!/usr/bin/env python3

"""Report various statistics from past publications useful for the proposal."""


from os import path

import numpy
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy.table import Table
import pandas
from scipy import stats

def report_w19_lurie_crossmatch(lurie_fname, w19_fname):
    """Count how many KIC from W19 also have measured spin period by Lurie."""

    with fits.open(lurie_fname, 'readonly') as lurie_f:
        lurie_data = lurie_f[1].data[:]

    w19_data = pandas.read_csv(w19_fname, sep=r'\s+', index_col='#KIC')

    kic_with_spin = lurie_data['KIC'][lurie_data['Class'] == 'sp']
    kic_with_fast_spin = lurie_data['KIC'][
        numpy.logical_and(lurie_data['Class'] == 'sp',
                          lurie_data['P1min']<13.5)
    ]
    print(
        'Out of %d W19 EBs %d also have spin, %d with P1min < 13.5d'
        %
        (
            w19_data.index.size,
            w19_data.index.intersection(kic_with_spin).size,
            w19_data.index.intersection(kic_with_fast_spin).size,
        )
    )


def get_hemisphere(sectors):
    """Return hemisphere list of sectors belongs to (0-south, 1-north)."""

    if sectors.rstrip(',') == '':
        return -1
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

    if not numpy.logical_and(sectors > 13, sectors <= 26).all():
        raise ValueError('Not all sectors from same hemisphere'
                         +
                         repr(sectors))

    return 1


def report_prsa_statistics(prsa_fname, lurie_tic_info):
    """Report how many Prsa binaries fall in various categories."""

    with fits.open(prsa_fname, 'readonly') as prsa_f:
        data = prsa_f[1].data[:]

    hemispheres = numpy.array(
        [get_hemisphere(s) for s in data['Sectors']]
    )
    data = data[hemispheres >= 0]
    hemispheres = hemispheres[hemispheres >= 0]
    print('%d / %d Prsa EBs from S / N hemisphere'
          %
          ((1-hemispheres).sum(), hemispheres.sum()))
    detached = data['Morph'] < 0.5
    data = data[detached]
    hemispheres = hemispheres[detached]

    print('%d / %d Detached Prsa EBs from S / N hemisphere'
          %
          ((1-hemispheres).sum(), hemispheres.sum()))


    prsa_tics = numpy.array([int(tic) for tic in data['TIC']])

    for mode in ['pf', '2g']:
        deep = numpy.logical_and(
            numpy.logical_and(
                numpy.isfinite(data['Dp-' + mode]),
                data['Dp-' + mode] > 0.03
            ),
            data['Ds-' + mode] > 0.005
        )
        print('From %s: %d / %d deep detached EBs from S / N hemisphere'
              %
              (mode, (1-hemispheres[deep]).sum(), hemispheres[deep].sum()))

        prsa_tics_in_lurie = lurie_tic_info.index.intersection(prsa_tics[deep])
        print('From %s: %d deep detached Prsa TICs in Lurie: '
              %
              (mode, prsa_tics_in_lurie.size))


    prsa_tics_in_lurie = lurie_tic_info.index.intersection(prsa_tics)
    print('Total %d detached Prsa TICs in Lurie: ' % prsa_tics_in_lurie.size)



def get_raw_lurie_tic_info(lurie_fname, lurie_tic_fname):
    """Return a numpy record array of TIC info for all Lurie sources."""

    if path.exists(lurie_tic_fname):
        return numpy.load(lurie_tic_fname)

    with fits.open(lurie_fname, 'readonly') as lurie_f:
        lurie_kic = lurie_f[1].data['KIC']

    lurie_kic = lurie_kic[numpy.logical_and(lurie_kic != 9777987,
                                            lurie_kic != 10879213)]

    numpy_result = None
    for row_i, kic_id in enumerate(lurie_kic):
        tic_row = numpy.array(Catalogs.query_object('KIC' + str(kic_id),
                                                    catalog='TIC',
                                                    radius=1e-2)[0])
        if numpy_result is None:
            dtype = numpy.dtype([
                (name, ('<U10' if name =='KIC' else tic_row.dtype[name]))
                for name in tic_row.dtype.names
            ])
            numpy_result = numpy.empty(shape=lurie_kic.shape,
                                       dtype=dtype)
            print('Created result with dytpe: ' + repr(numpy_result.dtype))
        numpy_result[row_i] = tic_row
        if str(tic_row['KIC']) == '':
            numpy_result[row_i]['KIC'] = numpy.array(str(kic_id))
            print('Manually filling in KIC ID: %s -> %s'
                  %
                  (repr(kic_id), repr(numpy_result[row_i]['KIC'])))


        if int(numpy_result[row_i]['KIC']) != kic_id:
            print(
                'Bad catalog result for KIC %d: %s -> %s' %
                (
                    kic_id,
                    repr(tic_row['KIC']),
                    repr(numpy_result[row_i]['KIC'])
                )
            )
            exit(1)


        print('Progress: %d/%d' % (row_i, lurie_kic.size), end='\n')

    numpy.save(lurie_tic_fname, numpy_result)
    return numpy_result


def get_lurie_tic_info(*args, **kwargs):
    """Format the Lurie TIC data to pandas dataframe."""

    raw_data = get_raw_lurie_tic_info(*args, **kwargs)
    return pandas.DataFrame.from_records(
        raw_data,
        index=raw_data['ID'].astype(int)
    )


def report_expected_ffi_lurie_spins(lurie_fname, lurie_tic_info):
    """Report how many of the Lurie binaries are bright fast rotators."""

    with fits.open(lurie_fname, 'readonly') as lurie_f:
        lurie_data = Table(lurie_f[1].data).to_pandas().set_index('KIC')

    lurie_tic_info = lurie_tic_info.set_index(lurie_tic_info['KIC'].astype(int))
    print('Lurie data index: ' + repr(lurie_data.index))
    print('TIC info index: ' + repr(lurie_tic_info.index))
    lurie_data = lurie_data.merge(lurie_tic_info,
                                  left_index=True,
                                  right_index=True)
    print('Lurie data:\n' + repr(lurie_data))
    print(
        '%d Lurie EBs have T<13.5 and P1 < 13.5'
        %
        numpy.logical_and(
            lurie_data['Tmag'] < 13.5,
            lurie_data['P1min'] < 13.5
        ).sum()
    )




def report_w19_precision(w19_fname):
    """Report the 1-sigma confidence interval for each W19 parameter."""


def main():
    """Avoid polluting the global namespace."""

    data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                         'data')
    fnames = dict(
        lurie=path.join(data_dir, 'lurie.fits'),
        lurie_tic=path.join(data_dir, 'lurie_tic.npy'),
        w19=path.join(data_dir, 'w19_maxlike_pars.dat'),
        prsa=path.join(data_dir, 'prsa_ebs.fits')
    )
    lurie_tic_info = get_lurie_tic_info(fnames['lurie'], fnames['lurie_tic'])
    report_expected_ffi_lurie_spins(fnames['lurie'], lurie_tic_info)

    report_w19_lurie_crossmatch(fnames['lurie'], fnames['w19'])
    report_prsa_statistics(fnames['prsa'], lurie_tic_info)


if __name__ == '__main__':
    main()
