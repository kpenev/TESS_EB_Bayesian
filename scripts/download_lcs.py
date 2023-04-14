#!/usr/bin/env python3

"""Interface for dowloading LCs by TIC."""

from matplotlib import pyplot
from astroquery.mast import Observations
from astropy.io import fits
#import lightkurve
import numpy

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
    with fits.open(download['Local Path'][0], 'readonly') as fits_f:
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


if __name__ == '__main__':
    get_astroquery(33419790, 'QLP', True)
    get_astroquery(33419790, 'SPOC', True)
    get_lightkurve(33419790)

    pyplot.legend()
    pyplot.show()
