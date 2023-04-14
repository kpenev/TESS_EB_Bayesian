#!/usr/bin/env python3

"""Interface for dowloading LCs by TIC."""

from os import path

from matplotlib import pyplot
from astroquery.mast import Observations
from astropy.io import fits
#import lightkurve
import numpy
from scipy.linalg import lstsq

from data_fnames import plot_dir

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


def get_astroquery(tic, provenance='SPOC', plot=False):
    """Return the lightcurve using astroquery interface."""


    if provenance == 'QLP':
        fits_path = path.join(
            path.dirname(__file__),
            'mastDownload',
            'HLSP',
            'hlsp_qlp_tess_ffi_s0006-0000000033419790_tess_v01_llc',
            'hlsp_qlp_tess_ffi_s0006-0000000033419790_tess_v01_llc.fits'
        )
    else:
        fits_path = path.join(
            path.dirname(__file__),
            'mastDownload',
            'TESS',
            'tess2018349182500-s0006-0000000033419790-0126-s',
            'tess2018349182500-s0006-0000000033419790-0126-s_lc.fits'
        )
    print('Fits path: ' + repr(fits_path))
    if not path.exists(fits_path):
        #False positive
        #pylint: disable=no-member
        objects = Observations.query_object('TIC' + str(tic), radius=0.001)
        #pylint: enable=no-member

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

        #False positive
        #pylint: disable=no-member
        products = Observations.get_product_list(objects[selection])
        #pylint: enable=no-member

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


def plot_spoc_qlp_comparison(tic):
    """Create a plot comparing SPOC PDCSAP LC to QLP un-detrended LC."""

    spoc_flux = 'PDCSAP_FLUX'
    qlp_lc = get_astroquery(tic, 'QLP')
    pdcsap_lc = get_astroquery(tic, 'SPOC')
    pdcsap_lc = pdcsap_lc[
        numpy.logical_and(
            numpy.isfinite(pdcsap_lc['TIME']),
            numpy.isfinite(pdcsap_lc[spoc_flux])
        )
    ]
    print(
        'QLP times all_finite: %s, sorted: %s; fluxes all finite: %s'
        %
        (
            numpy.isfinite(qlp_lc['TIME']).all(),
            (qlp_lc['TIME'][1:] > qlp_lc['TIME'][:-1]).all(),
            numpy.isfinite(qlp_lc['SAP_FLUX']).all()
        )
    )

    print(
        'PDCSAP times all_finite: %s, sorted: %s; fluxes all finite: %s'
        %
        (
            numpy.isfinite(pdcsap_lc['TIME']).all(),
            (pdcsap_lc['TIME'][1:] > pdcsap_lc['TIME'][:-1]).all(),
            numpy.isfinite(pdcsap_lc[spoc_flux]).all()
        )
    )


    pdcsap_split_indices = numpy.searchsorted(
        pdcsap_lc['TIME'],
        0.5 * (qlp_lc['TIME'][1:] + qlp_lc['TIME'][:-1])
    )

    npoints = qlp_lc['TIME'].size
    assert pdcsap_split_indices.size == npoints - 1
    binned_pdcsap_flux = numpy.empty((npoints, 2))
    start_i = 0
    for bin_i, end_i in enumerate(pdcsap_split_indices):
        binned_pdcsap_flux[bin_i, 0] = numpy.mean(
            pdcsap_lc[spoc_flux][start_i:end_i]
        )
        start_i = end_i

    binned_pdcsap_flux[-1, 0] = numpy.mean(pdcsap_lc[spoc_flux][start_i:])
    keep_qlp = numpy.isfinite(binned_pdcsap_flux[:, 0])
    npoints = keep_qlp.sum()
    binned_pdcsap_flux[:, 1] = 1

    unbinned_pdcsap_flux = pdcsap_lc[spoc_flux]
    while not keep_qlp.all():
        print('Fitting')
        binned_pdcsap_flux = binned_pdcsap_flux[keep_qlp, :]
        qlp_lc = qlp_lc[keep_qlp]
        shift_scale = lstsq(binned_pdcsap_flux,
                            qlp_lc['SAP_FLUX'])[0]
        binned_pdcsap_flux[:, 0] = binned_pdcsap_flux.dot(shift_scale)
        unbinned_pdcsap_flux = (shift_scale[0] * unbinned_pdcsap_flux
                                +
                                shift_scale[1])
        keep_qlp = (
            numpy.abs(qlp_lc['SAP_FLUX'] - binned_pdcsap_flux[:, 0])
            <
            2e-3
        )

    pyplot.subplot(211)
    pyplot.plot(pdcsap_lc['TIME'],
                unbinned_pdcsap_flux,
                'x',
                markeredgecolor='lightgrey',
                label=spoc_flux[:-5])
    pyplot.plot(qlp_lc['TIME'],
                binned_pdcsap_flux[:, 0],
                'x',
                label='binned ' + spoc_flux[:-5])
    pyplot.plot(qlp_lc['TIME'], qlp_lc['SAP_FLUX'], '+', label='QLP SAP')
    pyplot.xlabel('BJD - 2457000 [days]')
    pyplot.ylabel('Flux')
    pyplot.legend()

    pyplot.subplot(212)
    pyplot.plot(qlp_lc['TIME'], binned_pdcsap_flux[:, 0] - qlp_lc['SAP_FLUX'])
#    pyplot.ylim(-1.5e-3, 2e-3)
    pyplot.xlabel('BJD - 2457000 [days]')
    pyplot.ylabel('SPOC PDCSAP - QLP SAP')

    pyplot.savefig(path.join(plot_dir, 'qlp_spoc_comparison.pdf'))


if __name__ == '__main__':
    plot_spoc_qlp_comparison(33419790)
