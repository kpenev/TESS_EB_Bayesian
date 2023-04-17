#!/usr/bin/env python3

"""Interface for dowloading LCs by TIC."""

from os import path
from glob import glob
import re

from matplotlib import pyplot
from astroquery.mast import Observations
#from astroquery.exceptions import InvalidQueryError
from astropy.io import fits
#import lightkurve
import numpy
from scipy.linalg import lstsq

from data_fnames import data_fnames

#def get_lightkurve(tic, plot=False):
#    """Return the lightcurve using lightkurve interface."""
#
#    lightcurve = lightkurve.search_lightcurve(
#        'TIC' + str(tic),
#        author='SPOC'
#    )[0].download()
#    print(type(lightcurve))
#    if plot:
#        pyplot.plot(
#            lightcurve['time'].value,
#            (
#                lightcurve['pdcsap_flux'].value
#                /
#                numpy.nanmax(lightcurve['pdcsap_flux'].value)
#            ),
#            'x',
#            label='LK'
#        )
#    return lightcurve


def get_astroquery(tic, sector, provenance='SPOC', plot=False):
    """Return the lightcurve using astroquery interface."""


    sector_tic_fname_part = 's{sector}-{tic:016d}'.format(
        sector=('*' if sector == 'all' else '{0:04d}'.format(sector)),
        tic=tic
    )

    if provenance == 'QLP':
        fits_list = glob(
            path.join(
                'mastDownload',
                'HLSP',
                'hlsp_qlp_tess_ffi_%s_tess_v01_llc' % sector_tic_fname_part,
                'hlsp_qlp_tess_ffi_%s_tess_v01_llc.fits' % sector_tic_fname_part,
            )
        )
        parse_sector_rex = re.compile(
            '.*hlsp_qlp_tess_ffi_s(?P<sector>[0-9]*)-[0-9]*_tess_v01_llc'
        )
    else:
        fits_list = glob(
            path.join(
                'mastDownload',
                'TESS',
                'tess*-%s-*-s' % sector_tic_fname_part,
                'tess*-%s-*-s_lc.fits' % sector_tic_fname_part
            )
        )
        parse_sector_rex = re.compile(
            '.*tess.*-s(?P<sector>[0-9]*)-[0-9]*-.*-s'
        )


    print('Fits list: ' + repr(fits_list))
    assert not fits_list or sector == 'all' or len(fits_list) == 1

    print('Fits path: ' + repr(fits_list))
    if fits_list:
        fits_list = [
            (fits_fname, int(parse_sector_rex.match(fits_fname)['sector']))
            for fits_fname in fits_list
        ]
    else:
        #False positive
        #pylint: disable=no-member
        objects = Observations.query_object('TIC' + str(tic), radius=0.001)
        #pylint: enable=no-member

        print('\tFound %d objects:\n' % len(objects) + repr(objects))

        selection = numpy.logical_and(
            objects['obs_collection'] == ('TESS' if provenance == 'SPOC'
                                          else 'HLSP'),
            objects['dataproduct_type'] == 'timeseries'
        )
        selection = numpy.logical_and(
            selection,
            objects['project'] == 'TESS'
        )
        selection = numpy.logical_and(
            selection,
            objects['provenance_name'] == provenance
        )

        print(
            '\tAvailable sectors: '
            +
            repr(objects['sequence_number'][selection])
        )

        if sector != 'all':
            selection = numpy.logical_and(
                selection,
                objects['sequence_number'] == sector
            )
            print('\tSelected %d objects from sectors:\n' % selection.sum()
                  +
                  repr(objects[selection]['sequence_number']))

        #False positive
        #pylint: disable=bad-string-format-type
        print('\tSelected %d/%d objects' % (selection.sum(), len(objects)))
        #pylint: enable=bad-string-format-type

        fits_list = []
        for get_object in objects[selection]:
            #False positive
            #pylint: disable=no-member
            products = Observations.get_product_list(get_object)
            #pylint: enable=no-member

            print('\tFound %d products:' + repr(products))

            product_selection = numpy.logical_and(
                products['productType'] == 'SCIENCE',
                products['description'] == (
                    'Light curves' if provenance == 'SPOC'
                    else 'FITS'
                )
            )
            if product_selection.sum() == 0:
                continue
            if provenance == 'SPOC' and sector != 'all':
                if product_selection.sum() != 1:
                    print('Ambiguous products:\n'
                          +
                          repr(products[product_selection]))
                assert product_selection.sum() == 1
            download = Observations.download_products(
                products[product_selection]
            )
            fits_list.append(
                (download['Local Path'], get_object['sequence_number'])
            )

    result = dict()
    for fits_path, fits_sector in fits_list:
        with fits.open(fits_path, 'readonly') as fits_f:
            lightcurve = fits_f[1].data[:]
        if plot:
            if plot is True:
                if provenance == 'SPOC':
                    plot = 'PDCSAP_FLUX'
                else:
                    plot = 'SAP_FLUX'

            pyplot.plot(lightcurve['TIME'],
                        lightcurve[plot] / numpy.nanmax(lightcurve[plot]),
                        '+',
                        label='AQ ' + provenance)
        result[fits_sector] = lightcurve

    if sector == 'all':
        print('Returning %d LCs' % len(result))
        return result
    return result[sector]


def get_eb_params(tic, catalog='ja21'):
    """Return the information for the given TIC in the J&A catalog."""

    with fits.open(data_fnames[catalog], 'readonly') as catalog_file:
        params = catalog_file[1].data[:]

    if catalog == 'ja21':
        return params[params['TIC'] == tic]
    prsa_tics = numpy.array([int(tic) for tic in params['TIC']])
    return params[prsa_tics == tic]


def bin_lightcurve(fluxes,
                   times,
                   bin_times,
                   exposure_shift=0,
                   average=numpy.mean):
    """Bin given LC to lower cadence (bin_times)."""

    assert fluxes.size == times.size

    split_indices = numpy.searchsorted(
        times,
        0.5 * (bin_times[1:] + bin_times[:-1])
    ) + exposure_shift

    assert split_indices.size == bin_times.size - 1
    binned_flux = numpy.empty(bin_times.size)
    start_i = 0
    for bin_i, end_i in enumerate(split_indices):
        binned_flux[bin_i] = average(
            fluxes[start_i : end_i]
        )
        start_i = end_i

    binned_flux[-1] = numpy.mean(fluxes[start_i :])
    return binned_flux


def match_lightcurves(qlp_lc, spoc_lc, spoc_flux, exposure_shift=0):
    """Bin, shift and scale the PDCSAP LC to match the QLP SAP."""

    binned_spoc_flux = numpy.empty((qlp_lc['TIME'].size, 2))

    binned_spoc_flux[:, 0] = bin_lightcurve(spoc_lc[spoc_flux],
                                            spoc_lc['TIME'],
                                            qlp_lc['TIME'],
                                            exposure_shift)
    keep_qlp = numpy.isfinite(binned_spoc_flux[:, 0])
    npoints = keep_qlp.sum()
    binned_spoc_flux[:, 1] = 1

    unbinned_spoc_flux = spoc_lc[spoc_flux]
    enter = True
    while enter or not keep_qlp.all():
        binned_spoc_flux = binned_spoc_flux[keep_qlp, :]
        qlp_lc = qlp_lc[keep_qlp]
        binned_gap_slices = numpy.append(
            numpy.argwhere(
                qlp_lc['TIME'][1:] - qlp_lc['TIME'][:-1]
                >
                0.3
            ).flatten() + 1,
            qlp_lc['TIME'].size
        )
        unbinned_gap_slices = numpy.append(
            numpy.argwhere(
                spoc_lc['TIME'][1:] - spoc_lc['TIME'][:-1]
                >
                0.3
            ).flatten() + 1,
            spoc_lc['TIME'].size
        )

        print('Gap slices:\n' + repr(binned_gap_slices) + '\n' +
              repr(unbinned_gap_slices))

        assert binned_gap_slices.size == unbinned_gap_slices.size

        binned_slice_start = unbinned_slice_start = 0
        for binned_slice_end, unbinned_slice_end in zip(binned_gap_slices,
                                                        unbinned_gap_slices):
            shift_scale = lstsq(
                binned_spoc_flux[binned_slice_start : binned_slice_end],
                qlp_lc['SAP_FLUX'][binned_slice_start : binned_slice_end]
            )[0]
            binned_spoc_flux[binned_slice_start : binned_slice_end, 0] = (
                binned_spoc_flux[binned_slice_start : binned_slice_end].dot(
                    shift_scale
                )
            )
            unbinned_spoc_flux[unbinned_slice_start : unbinned_slice_end] = (
                shift_scale[0] * unbinned_spoc_flux[unbinned_slice_start
                                                    :
                                                    unbinned_slice_end]
                +
                shift_scale[1]
            )
            binned_slice_start = binned_slice_end
            unbinned_slice_start = unbinned_slice_end

        keep_qlp = (
            numpy.abs(qlp_lc['SAP_FLUX'] - binned_spoc_flux[:, 0])
            <
            0.003
        )
        enter = False

    return qlp_lc, binned_spoc_flux, unbinned_spoc_flux


def main():
    """Avoid polluting global namespace."""

    plot_spoc_qlp_comparison(33419790, 6, (0.99, 1.015))
    plot_spoc_qlp_comparison(27767184,
                             14,
                             (0.985, 1.01),
                             exposure_shift=2,
                             bottom_left='zoomy')
#    checked_tic = [121022559, 122682776, 137549183, 137975907, 121124831,
#                   394179202,
#                   122224804, 122304494, 122606463, 184298625, 120684604,
#                   121866154,
#                   121604042, 122446960, 121598562, 378089068, 170344769,
#                   184008771,
#                   172422394, 121945407, 169467727, 137341354, 121016578,
#                   138639253,
#                   169819162, 170246850, 138430438, 63370066, 159573299,
#                   268289462,
#                   164413080, 123201406, 164552564, 158635832, 63449090,
#                   159720778,
#                   268305489, 268482699, 159047480, 63126950, 270700608,
#                   158988347,
#                   274129522, 63074282, 164458426, 272074664, 273042650,
#                   271773721, 158491288, 164781045, 158660631, 273131564,
#                   271040947, 275576176, 268380299, 269031095, 270517432,
#                   273373712, 272843045, 63071165, 271877841, 272598447,
#                   164557531, 63291675, 273376048, 268383780, 416635004,
#                   405685992, 27915909, 28449295, 48507019, 279918206,
#                   407000096, 27006880, 27397122, 27767184, 27845677,
#                   27843942,
#                   137975907,
#                   121604042,
#                   137341354,
#                   63126950,
#                   158988347,
#                   63291675,]

#    potential_example_tics = [27767184,
#                              27843942,
#                              137975907,
#                              121604042,
#                              137341354,
#                              63126950,
#                              158988347,
#                              63291675,]
#    synchronized_rotation = [273373712]
#
#    slightly_sub_synchronous_eb = [268289462]
#    eccentric_eb = [120684604]
#    wtf = [158660631]

#    for tic in potential_example_tics:
#        print('TIC: ' + repr(tic))
#        try:
#            plot_spoc_qlp_comparison(tic, 14, (0.97, 1.03))
#        except InvalidQueryError:
#            plot_spoc_qlp_comparison(tic, 15, (0.97, 1.03))

if __name__ == '__main__':
    main()
