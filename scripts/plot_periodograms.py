#!/usr/bin/env python3

"""Create and plot ACF, LSP, and wavelett map for a TIC."""

from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
from astropy.timeseries import LombScargle
import numpy
from scipy.linalg import lstsq

from download_lcs import get_astroquery as get_lc, get_eb_params, bin_lightcurve

def parse_command_line():
    """Return the command line configuration as namespace attributes."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['plot_periodograms.cfg'],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--tic-sector',
        type=int,
        nargs=2,
        default=(27767184, 14),
        help='The TIC identifier and sector of the LC to generate periodograms '
        'for.'
    )
    parser.add_argument(
        '--lightcurve-provenance',
        choices=['SPOC', 'QLP'],
        default='SPOC',
        help='The pipeline to get the lightcurve for.'
    )
    parser.add_argument(
        '--acf-fname', '--acf',
        default=None,
        help='The filename to save the autocorrelation plot as. If unspecified '
        'it is shown.'
    )
    parser.add_argument(
        '--lsp-fname', '--lsp',
        default=None,
        help='The filename to save the Lomb-Scargle periodogram plot as. If '
        'unspecified it is shown.'
    )
    parser.add_argument(
        '--wavelet-map-fname', '--wm',
        default=None,
        help='The filename to save the wavelet map as. If unspecified it is '
        'shown.'
    )
    parser.add_argument(
        '--global-wavelet-spectrum-fname', '--gws',
        default=None,
        help='The filename to save the global wavelet spectrum as. If '
        'unspecified it is shown.'

    )
    parser.add_argument(
        '--match-resolution',
        type=float,
        default=1.0 / 48.0,
        help='The resolution at which to bin the LC to match transits'
    )
    return parser.parse_args()


def get_eclipse_mask(times, params, safety=1.5):
    """Return a boolean array with True for LC points far from eclipse."""

    phase_from_primary = (
        ((times - params['BJD0']) / params['Per'])
        %
        1.0
    )
    phase_from_secondary = (
        (
            (
                times - params['BJD0']
                +
                (params['Phis-pf'] - params['Phip-pf']) * params['Per']
            )
            /
            params['Per']
        )
        %
        1.0
    )
    return numpy.logical_or(
        numpy.minimum(
            numpy.abs(phase_from_primary),
            numpy.abs(phase_from_primary - 1.0)
        ) < safety * params['Wp-pf'] / 2.0,
        numpy.minimum(
            numpy.abs(phase_from_secondary),
            numpy.abs(phase_from_secondary - 1.0)
        ) < safety * params['Ws-pf'] / 2.0
    )


def match_lc_segment(target_fluxes,
                     target_times,
                     match_mask,
                     reference_fluxes,
                     reference_times):
    """
    Return shifted & scaled target to match reference.


    Args:
        target_fluxes(array):    The flux measurements to shift and scale to
            match to the target.

        target_times(array):    The times at which the target fluxes are
            measured.

        match_mask(array):    Flags indicating which points from the target to
            use for matching.

        reference_fluxes(array):    The fluxes to match to.

        reference_fluxes(array):    The times at which reference fluxes to match
            are specified.

    Returns:
        array;
            The shifted and scaled ``target_fluxes`` such that the
            ``match_mask`` points are as close as possible to the reference
            fluxes at overlapping times.
    """

    binned_target_fluxes = numpy.empty((reference_times.size, 2))
    binned_target_fluxes[:, 0] = bin_lightcurve(target_fluxes[match_mask],
                                                target_times[match_mask],
                                                reference_times)
    binned_target_fluxes[:, 1] = 1.0
    fit_mask = numpy.isfinite(binned_target_fluxes[:, 0])
    shift_scale = lstsq(binned_target_fluxes[fit_mask, 0],
                        reference_fluxes[fit_mask])[0]
    return target_fluxes * shift_scale[0] + shift_scale[1]


def get_reference(first_lc, match_resolution, orbital_period, flux_column):
    """Return a full orbital period of flux to match eclipses to."""

    match_times = numpy.arange(0, orbital_period, match_resolution)
    pyplot.plot(first_lc['TIME'], first_lc[flux_column])
    pyplot.show()
    for norb in range(
            int((first_lc['TIME'][-1] - first_lc['TIME'][0]) // orbital_period)
    ):
        try_lc_portion = (
            (first_lc['TIME'] - first_lc['TIME'][0]) // orbital_period
            ==
            norb
        )

        print('Norb: ' + repr(norb))
        match_fluxes = bin_lightcurve(
            first_lc[flux_column][try_lc_portion],
            (
                first_lc['TIME'][try_lc_portion]
                - norb * orbital_period
                - first_lc['TIME'][0]
            ),
            match_times,
            average=numpy.nanmean
        )
        if numpy.isfinite(match_fluxes).all():
            break

    assert numpy.isfinite(match_fluxes).all()
    pyplot.plot(match_times, match_fluxes)
    pyplot.show()
    return match_times, match_fluxes


def get_folded_binned_lc(match_times, match_flixes, lightcurve_dict, params):
    """Match eclipses in each orbital period segment of each LC to reference."""

    first_obs_time = lightcurve_dict[min(lightcurve_dict.keys())]['TIME'][0]
    for lightcurve in lightcurve_dict.values():
        pass


def main(config):
    """Avoid polluting global namespace."""

    lightcurve_dict = get_lc(config.tic_sector[0], 'all',
                             provenance=config.lightcurve_provenance)
    print('LC dict: ' + repr(lightcurve_dict))
    params = get_eb_params(config.tic_sector[0], 'prsa')

    print('Params:\n' + '\n'.join(['%s: %s' % (p, repr(params[p]))
                                   for p in params.dtype.names]))
    flux_column = 'SAP_FLUX'
    if config.lightcurve_provenance == 'SPOC':
        flux_column = 'PDC' + flux_column
    all_fluxes = numpy.empty(0)
    all_times = numpy.empty(0)


    match_times, match_fluxes = get_reference(
        lightcurve_dict[min(lightcurve_dict.keys())],
        config.match_resolution,
        params['Per'],
        flux_column
    )

    for sector, lightcurve in lightcurve_dict.items():
        lightcurve = lightcurve[
            numpy.logical_and(
                numpy.isfinite(lightcurve['TIME']),
                numpy.isfinite(lightcurve[flux_column])
            )
        ]
        out_of_eclipse = numpy.logical_not(get_eclipse_mask(lightcurve['TIME'],
                                                            params))

        all_fluxes = numpy.append(all_fluxes, lightcurve[flux_column])
        all_times = numpy.append(all_times, lightcurve['TIME'])

        #pyplot.plot(lightcurve['TIME'], lightcurve[flux_column], '.k')
        pyplot.plot(lightcurve['TIME'][out_of_eclipse],
                    lightcurve[flux_column][out_of_eclipse], '.g')

        lomb_scargle = numpy.empty((3, 1000))
        lomb_scargle[0] = numpy.linspace(0.1, 10, 1000)
        lomb_scargle[1] = LombScargle(
            lightcurve['TIME'],
            lightcurve[flux_column]
        ).power(frequency=1.0 / lomb_scargle[0])
        lomb_scargle[2] = LombScargle(
            lightcurve['TIME'][out_of_eclipse],
            lightcurve[flux_column][out_of_eclipse]
        ).power(frequency=1.0 / lomb_scargle[0])

        #pyplot.plot(lomb_scargle[0], lomb_scargle[1], '-r', label='raw')
#        pyplot.plot(lomb_scargle[0],
#                    lomb_scargle[2],
#                    '-',
#                    label='Sector %d' % sector)
    pyplot.axvline(params['Per'], color='black')
    pyplot.axvline(params['Per'] / 2, color='black')
    pyplot.legend()

    pyplot.show()

    pyplot.plot(all_times % params['Per'], all_fluxes, ',k')
    pyplot.show()


if __name__ == '__main__':
    main(parse_command_line())
