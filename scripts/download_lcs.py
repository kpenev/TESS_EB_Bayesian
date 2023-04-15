#!/usr/bin/env python3

"""Interface for dowloading LCs by TIC."""

from os import path
from glob import glob

from matplotlib import pyplot
from astroquery.mast import Observations
from astroquery.exceptions import InvalidQueryError
from astropy.io import fits
#import lightkurve
import numpy
from scipy.linalg import lstsq

from data_fnames import plot_dir, data_fnames

def get_lightkurve(tic, plot=False):
    """Return the lightcurve using lightkurve interface."""

    lightcurve = lightkurve.search_lightcurve(
        'TIC' + str(tic),
        author='SPOC'
    )[0].download()
    print(type(lightcurve))
    if plot:
        pyplot.plot(
            lightcurve['time'].value,
            (
                lightcurve['pdcsap_flux'].value
                /
                numpy.nanmax(lightcurve['pdcsap_flux'].value)
            ),
            'x',
            label='LK'
        )
    return lightcurve


def get_astroquery(tic, sector, provenance='SPOC', plot=False):
    """Return the lightcurve using astroquery interface."""


    sector_tic_fname_part = 's{sector:04d}-{tic:016d}'.format(sector=sector,
                                                              tic=tic)

    if provenance == 'QLP':
        fits_path = path.join(
            'mastDownload',
            'HLSP',
            'hlsp_qlp_tess_ffi_%s_tess_v01_llc' % sector_tic_fname_part,
            'hlsp_qlp_tess_ffi_%s_tess_v01_llc.fits' % sector_tic_fname_part,
        )
    else:
        fits_path = glob(
            path.join(
                'mastDownload',
                'TESS',
                'tess*-%s-*-s' % sector_tic_fname_part,
                'tess*-%s-*-s_lc.fits' % sector_tic_fname_part
            )
        )
        if fits_path:
            assert len(fits_path) == 1
            fits_path = fits_path[0]
        else:
            fits_path = ''

    print('Fits path: ' + repr(fits_path))
    if not path.exists(fits_path):
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
        print('\tSelected %d objects:\n' % selection.sum()
              +
              repr(objects[selection]))

        selection = numpy.logical_and(
            selection,
            objects['project'] == 'TESS'
        )
        print('\tSelected %d objects:\n' % selection.sum()
              +
              repr(objects[selection]))

        selection = numpy.logical_and(
            selection,
            objects['provenance_name'] == provenance
        )
        print('\tSelected %d objects from sectors:\n' % selection.sum()
              +
              repr(objects[selection]['sequence_number']))

        selection = numpy.logical_and(
            selection,
            objects['sequence_number'] == sector
        )
        print('\tSelected %d objects:\n' % selection.sum()
              +
              repr(objects[selection]))

        #False positive
        #pylint: disable=bad-string-format-type
        print('\tSelected %d/%d objects' % (selection.sum(), len(objects)))
        #pylint: enable=bad-string-format-type

        #False positive
        #pylint: disable=no-member
        products = Observations.get_product_list(objects[selection])
        #pylint: enable=no-member

        print('\tFound %d products' % len(products))

        selection = numpy.logical_and(
            products['productType'] == 'SCIENCE',
            products['description'] == ('Light curves' if provenance == 'SPOC'
                                        else 'FITS')
        )
        if provenance == 'SPOC':
            assert selection.sum() == 1
        download = Observations.download_products(products[selection])
        fits_path = download['Local Path'][0]

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
    return lightcurve


def get_eb_params(tic, catalog='ja21'):
    """Return the information for the given TIC in the J&A catalog."""

    with fits.open(data_fnames[catalog], 'readonly') as catalog_file:
        params = catalog_file[1].data[:]

    if catalog == 'ja21':
        return params[params['TIC'] == tic]
    prsa_tics = numpy.array([int(tic) for tic in params['TIC']])
    return params[prsa_tics == tic]


def match_lightcurves(qlp_lc, spoc_lc, spoc_flux, exposure_shift=0):
    """Bin, shift and scale the PDCSAP LC to match the QLP SAP."""

    spoc_split_indices = numpy.searchsorted(
        spoc_lc['TIME'],
        0.5 * (qlp_lc['TIME'][1:] + qlp_lc['TIME'][:-1])
    ) + exposure_shift

    npoints = qlp_lc['TIME'].size
    assert spoc_split_indices.size == npoints - 1
    binned_spoc_flux = numpy.empty((npoints, 2))
    start_i = 0
    for bin_i, end_i in enumerate(spoc_split_indices):
        binned_spoc_flux[bin_i, 0] = numpy.mean(
            spoc_lc[spoc_flux][start_i : end_i]
        )
        start_i = end_i

    binned_spoc_flux[-1, 0] = numpy.mean(spoc_lc[spoc_flux][start_i :])
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


def plot_spoc_qlp_comparison(tic,
                             sector,
                             zoom_y,
                             *,
                             exposure_shift=0,
                             bottom_left='diff'):
    """Create a plot comparing SPOC PDCSAP LC to QLP un-detrended LC."""

    spoc_flux = 'PDCSAP_FLUX'

    qlp_lc = get_astroquery(tic, sector, 'QLP')
    spoc_lc = get_astroquery(tic, sector, 'SPOC')
    params = get_eb_params(tic)
    first_eclipse = params['t1']
    if len(params) == 0:
        params = get_eb_params(tic, 'prsa')
        first_eclipse = params['BJD0']

    print('Params: ' + repr(params))

    spoc_lc = spoc_lc[
        numpy.logical_and(
            numpy.isfinite(spoc_lc['TIME']),
            numpy.isfinite(spoc_lc[spoc_flux])
        )
    ]
    time_offset = 100 * (spoc_lc['TIME'][0] // 100)
    qlp_lc['TIME'] -= time_offset
    spoc_lc['TIME'] -= time_offset
    first_eclipse -= time_offset

    qlp_lc, binned_spoc_flux, unbinned_spoc_flux = match_lightcurves(
        qlp_lc,
        spoc_lc,
        spoc_flux,
        exposure_shift
    )

    figure, axes = pyplot.subplots(
        nrows=2,
        ncols=2,
        sharex='col',
        sharey=('row' if bottom_left == 'zoomy' else False),
        gridspec_kw=dict(hspace=0,
                         wspace=0.05,
                         height_ratios=[1.0, 1.0],
                         width_ratios=[1.0, 0.6],
                         left=0.1,
                         bottom=0.17,
                         top=0.89,
                         right=0.93),
        figsize=[6.4, 2.5]
    )
    label = True
    flux_axes = [axes[0, 0], axes[0, 1], axes[1, 1]]
    if bottom_left == 'zoomy':
        flux_axes.append(axes[1, 0])
    for plot_ax in flux_axes:
        plot_ax.plot(spoc_lc['TIME'],
                     unbinned_spoc_flux,
                     '.',
                     markeredgecolor='lightgrey',
                     markerfacecolor='lightgrey',
                     label=('SPOC ' + spoc_flux[:-5] if label else None))
        plot_ax.plot(qlp_lc['TIME'],
                     binned_spoc_flux[:, 0],
                     'xr',
                     label=('binned SPOC ' + spoc_flux[:-5] if label else None))
        plot_ax.plot(qlp_lc['TIME'],
                     qlp_lc['SAP_FLUX'],
                     '.g',
                     markersize=2,
                     label=('QLP SAP' if label else None))
        label = False
    axes[0, 0].set_ylabel('Flux')

    axes[0, 1].set_xlim(80.5, 80.5 + params['Per'])
#    axes[0, 1].set_yticks([])

    zoom_x_min = 0.4 * spoc_lc['TIME'][0] + 0.6 * spoc_lc['TIME'][-1]
    zoom_x_min = (
        ((zoom_x_min - first_eclipse) // params['Per'] + 0.25)
        *
        params['Per']
        +
        first_eclipse
    )
    axes[1, 1].set_xlim(zoom_x_min, zoom_x_min + params['Per'])
    axes[1, 1].set_ylim(zoom_y)
    axes[1, 1].tick_params(axis='y', left=False, right=True)

    axes[0, 1].yaxis.tick_right()
    axes[1, 1].yaxis.tick_right()

    pyplot.figlegend(ncol=3, loc='upper center', borderaxespad=0)

    if bottom_left == 'diff':
        axes[1, 0].plot(qlp_lc['TIME'],
                        binned_spoc_flux[:, 0] - qlp_lc['SAP_FLUX'],
                        '.k',
                        label='SPOC PDCSAP - QLP SAP')
#    pyplot.ylim(-1.5e-3, 2e-3)
    figure.suptitle('BJD - %d [days]' % (2457000 + time_offset),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    x=0.5,
                    y=0.0)
    axes[1, 0].legend(markerscale=0,
                      frameon=True,
                      facecolor='white',
                      edgecolor='none')

    pyplot.savefig(
        path.join(
            plot_dir,
            'tic%016d_s%06d_qlp_spoc_comparison.pdf' % (tic, sector)
        )
    )


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
