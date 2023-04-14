#!/usr/bin/env python3

from matplotlib import pyplot
from astroquery.mast import Observations
from astropy.io import fits
import lightkurve
import numpy

def plot_lightkurve(TIC):
    """Return the lightcurve using lightkurve interface."""

    lc = lightkurve.search_lightcurve(
        'TIC' + str(TIC),
        author='SPOC'
    )[0].download()
    pyplot.plot(lc['time'].value,
                lc['pdcsap_flux'].value / numpy.nanmax(lc['pdcsap_flux'].value),
                'x',
                label='LK')


def plot_astroquery(TIC, provenance='SPOC'):
    """Return the lightcurve using astroquery interface."""

    objects = Observations.query_object('TIC' + str(TIC), 0.00001)
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
    products = Observations.get_product_list(objects[selection])
    print('Products:\n' + repr(products))
    selection = numpy.logical_and(
        products['productType'] == 'SCIENCE',
        products['description'] == ('Light curves' if provenance == 'SPOC'
                                    else 'FITS')
    )
    print('Selection:\n' + repr(selection))
    if provenance == 'SPOC':
        assert selection.sum() == 1
        flux_column = 'PDCSAP_FLUX'
    else:
        flux_column = 'SAP_FLUX'
    download = Observations.download_products(products[selection])
    with fits.open(download['Local Path'][0], 'readonly') as fits_f:
        lc = fits_f[1].data[:]
    pyplot.plot(lc['TIME'],
                lc[flux_column] / numpy.nanmax(lc[flux_column]),
                '+',
                label='AQ ' + provenance)


if __name__ == '__main__':
    plot_astroquery(33419790, 'QLP')
    plot_astroquery(33419790, 'SPOC')
    plot_lightkurve(33419790)

    pyplot.legend()
    pyplot.show()
