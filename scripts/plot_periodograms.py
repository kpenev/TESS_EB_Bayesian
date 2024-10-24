#!/usr/bin/env python3

"""Create and plot ACF, LSP, and wavelett map for a TIC."""

from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
from astropy.timeseries import LombScargle
import numpy
from scipy.linalg import lstsq
from scipy import signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d

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
        '--tic',
        type=int,
        default=27767184,
        help='The TIC identifier and sector of the LC to generate periodograms '
        'for.'
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
        nargs=2,
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


def iterate_orbit_slice(lightcurve,
                        bin_times,
                        orbital_period,
                        flux_column,
                        reference_time,
                        *,
                        include_partial=True,
                        average=numpy.nanmean):
    """Split LC in orbits, iterate returning slice of LC and binned flux."""

    min_norb = int((numpy.floor if include_partial else numpy.ceil)(
        (lightcurve['TIME'][0] - reference_time) / orbital_period
    ))
    max_norb = int((numpy.ceil if include_partial else numpy.floor)(
        (lightcurve['TIME'][-1] - reference_time) / orbital_period
    ))
    print('Norb range: ' + repr((min_norb, max_norb)))
    orbit_splits = numpy.searchsorted(
        lightcurve['TIME'],
        (
            numpy.arange(min_norb, max_norb + 1, dtype=float) * orbital_period
            +
            reference_time
        )
    )
    for norb in range(max_norb-min_norb):
        if orbit_splits[norb] == orbit_splits[norb + 1]:
            continue

        binned_fluxes = bin_lightcurve(
            lightcurve[flux_column][orbit_splits[norb]:orbit_splits[norb + 1]],
            (
                lightcurve['TIME'][orbit_splits[norb]:orbit_splits[norb + 1]]
                - (norb + min_norb)* orbital_period
                - reference_time
            ),
            bin_times,
            average=average
        )
        yield (
            lightcurve[orbit_splits[norb]:orbit_splits[norb + 1]],
            binned_fluxes,
            (norb + min_norb) * orbital_period + reference_time
        )


def get_reference(first_lc, match_resolution, orbital_period, flux_column):
    """Return a full orbital period of flux to match eclipses to."""

    match_times = numpy.arange(0, orbital_period, match_resolution)
    for _, match_fluxes, _ in iterate_orbit_slice(first_lc,
                                                  match_times,
                                                  orbital_period,
                                                  flux_column,
                                                  first_lc['TIME'][0],
                                                  include_partial=False):
        if numpy.isfinite(match_fluxes).all():
            break

    #Assumption is that at least one period is available in first LC.
    #pylint: disable=undefined-loop-variable
    return match_times, match_fluxes
    #pylint: enable=undefined-loop-variable


def get_folded_matched_lc(lightcurve_dict,
                         params,
                         match_resolution,
                         flux_column):
    """Match eclipses in each orbital period segment of each LC to reference."""

    first_lc = lightcurve_dict[min(lightcurve_dict.keys())]
    match_times, match_fluxes = get_reference(first_lc,
                                              match_resolution,
                                              float(params['Per']),
                                              flux_column)

    eclipse_mask = get_eclipse_mask(match_times + first_lc['TIME'][0],
                                  params,
                                  1.0)
    print('Will use %d points to solve for scale and shift.'
          %
          eclipse_mask.sum())
    assert numpy.isfinite(match_fluxes).all()


    sum_binned_fluxes = numpy.zeros(match_fluxes.shape)
    num_summed_fluxes = numpy.zeros(match_fluxes.shape, dtype=int)
    while True:
        result_fluxes = numpy.empty(0)
        result_times = numpy.empty(0)
        for lightcurve in [lightcurve_dict[14]]:
            lightcurve = numpy.copy(lightcurve)
            time = numpy.copy(lightcurve['TIME'][:-2])
            lightcurve = lightcurve[2:]
            lightcurve['TIME'] = time
            lightcurve = lightcurve[
                numpy.logical_and(
                    numpy.isfinite(lightcurve['TIME']),
                    numpy.isfinite(lightcurve[flux_column])
                )
            ]

            match_lhs = numpy.empty((match_times.size, 2))
            match_lhs[:, 1] = 1.0
            for (
                    lc_portion,
                    match_lhs[:, 0],
                    slice_tstart
            ) in iterate_orbit_slice(
                    lightcurve,
                    match_times,
                    params['Per'],
                    flux_column,
                    first_lc['TIME'][0],
                    average=numpy.mean
            ):

                finite = numpy.isfinite(match_lhs[:, 0])
                match_mask = numpy.logical_and(eclipse_mask, finite)
                shift_scale = lstsq(match_lhs[match_mask],
                                    match_fluxes[match_mask])[0]
                sum_binned_fluxes[finite] += match_lhs.dot(shift_scale)[finite]
                num_summed_fluxes[finite] += 1

                result_times = numpy.append(
                    result_times,
                    lc_portion['TIME'] - slice_tstart
                )
                result_fluxes = numpy.append(
                    result_fluxes,
                    shift_scale[0] * lc_portion[flux_column] + shift_scale[1]
                )
#                pyplot.plot(
#                    lc_portion['TIME'] - slice_tstart,
#                    shift_scale[0] * lc_portion[flux_column] + shift_scale[1],
#                    zorder=10
#                )
#        pyplot.legend()
#        pyplot.show()
        rms_ref_change = numpy.sqrt(
            numpy.mean(
                numpy.square(
                    match_fluxes
                    -
                    sum_binned_fluxes / num_summed_fluxes
                )
            )
        ) / match_fluxes.mean()
        print('RMS ref change: ' + repr(rms_ref_change))
        if rms_ref_change < 1e-5:
            break
        match_fluxes = sum_binned_fluxes / num_summed_fluxes

    pyplot.plot(match_times,
                match_fluxes,
                '-k',
                label='reference',
                zorder=100)
    pyplot.show()

    sorter = numpy.argsort(result_times)
    return result_times[sorter], result_fluxes[sorter]


def get_periodogram_data(config, params):
    """Return versions of the available LCs ready for peroid analysis."""

    lightcurve_dict = get_lc(config.tic, 'all', 'SPOC')

    if params:
        print('Params:\n' + '\n'.join(['%s: %s' % (p, repr(params[p]))
                                       for p in params.dtype.names]))

        folded_spoc_times, folded_spoc_fluxes = get_folded_matched_lc(
            lightcurve_dict,
            params,
            config.match_resolution,
            'PDCSAP_FLUX'
        )

    reference_time = lightcurve_dict[min(lightcurve_dict.keys())]['TIME'][0]

    lightcurve_dict = get_lc(config.tic, 'all', 'QLP')
    print('Available sectiors: ' + repr(lightcurve_dict.keys()))

    result = dict()
    for sector, lightcurve in lightcurve_dict.items():
        lightcurve = lightcurve[
            numpy.logical_and(
                numpy.isfinite(lightcurve['TIME']),
                numpy.isfinite(lightcurve['SAP_FLUX'])
            )
        ]

        corrected_flux = numpy.copy(lightcurve['SAP_FLUX'])
        if params:
            min_norb = int(numpy.floor((lightcurve['TIME'][0] - reference_time)
                                       /
                                       params['Per']))
            max_norb = int(numpy.ceil((lightcurve['TIME'][-1] - reference_time)
                                      /
                                      params['Per']))
            orbit_splits = numpy.searchsorted(
                lightcurve['TIME'],
                (
                    numpy.arange(min_norb, max_norb + 1, dtype=float)
                    *
                    params['Per']
                    +
                    reference_time
                )
            )
            for norb in range(max_norb - min_norb):
                lightcurve_slice = lightcurve[orbit_splits[norb]
                                              :
                                              orbit_splits[norb + 1]]

                binned_folded_flux = numpy.ones(
                    (orbit_splits[norb + 1] - orbit_splits[norb], 2)
                )
                binned_folded_flux[:, 0] = bin_lightcurve(
                    folded_spoc_fluxes,
                    folded_spoc_times,
                    (
                        lightcurve_slice['TIME']
                        -
                        reference_time
                        -
                        (norb  + min_norb) * params['Per']
                    ),
                    average=numpy.nanmean
                )
                assert numpy.isfinite(binned_folded_flux).all()
                assert numpy.isfinite(lightcurve_slice['SAP_FLUX']).all()
                eclipse_mask = get_eclipse_mask(lightcurve_slice['TIME'],
                                                params,
                                                1.0)
                if eclipse_mask.sum():
                    shift_scale = lstsq(
                        binned_folded_flux[eclipse_mask],
                        lightcurve_slice['SAP_FLUX'][eclipse_mask]
                    )[0]
                    corrected_flux[
                        orbit_splits[norb]
                        :
                        orbit_splits[norb + 1]
                    ] /= binned_folded_flux.dot(shift_scale)
                else:
                    corrected_flux[
                        orbit_splits[norb]
                        :
                        orbit_splits[norb + 1]
                    ] = numpy.nan

            corrected_flux[numpy.abs(corrected_flux - 1) > 0.03] = numpy.nan
            remove_eclipse_mask = numpy.logical_not(
                get_eclipse_mask(lightcurve['TIME'], params)
            )
            usable = numpy.logical_and(remove_eclipse_mask,
                                       numpy.isfinite(corrected_flux))
        else:
            remove_eclipse_mask = numpy.ones(len(lightcurve), dtype=bool)
            usable = numpy.ones(len(lightcurve), dtype=bool)

        result[sector] = dict(
            times=lightcurve['TIME'],
            raw_flux=lightcurve['SAP_FLUX'],
            corrected_flux=corrected_flux,
            remove_eclipse_mask=remove_eclipse_mask,
            usable_corrected_mask=usable
        )

    return result


def resample_lc(times, fluxes):
    """Return given LC with uniform sampling using linear interpolation."""

    def time_sampling_residual(offset_step):
        """Rms between uniform sampled time and observations ignoring gaps."""

        if (
            offset_step[1] <= 0
            or
            offset_step[0] > times[0]
            or
            offset_step[0] < times[0] - offset_step[1]
        ):
            return numpy.nan

        nsteps = (int(numpy.ceil((times[-1] - offset_step[0]) / offset_step[1]))
                  +
                  1)
        uniform_times = (
            offset_step[0]
            +
            offset_step[1] * numpy.arange(nsteps, dtype=float)
        )
        neighbors = numpy.searchsorted(uniform_times, times)
        return numpy.sqrt(
            numpy.minimum(
                numpy.square(uniform_times[neighbors] - times),
                numpy.square(uniform_times[neighbors - 1] - times),
            ).sum()
        )

    assert numpy.isfinite(times).all()
    timestep_guess = numpy.median(times[1:] - times[:-1])
    time_offset, time_step = minimize(
        time_sampling_residual,
        (times[0] - timestep_guess, timestep_guess),
        method='Nelder-Mead'
    ).x

    nsteps = (int(numpy.ceil((times[-1] - time_offset) / time_step)) + 1)
    resampled_times = (
        time_offset
        +
        time_step * numpy.arange(nsteps, dtype=float)
    )
    resampled_times = resampled_times[
        numpy.logical_and(resampled_times >= times[0],
                          resampled_times <= times[-1])
    ]
    resampled_fluxes = interp1d(
        times,
        fluxes,
        assume_sorted=True,
    )(
        resampled_times
    )
    return resampled_times, resampled_fluxes, time_step


def make_lsp_figures(periodogram_data, params):
    """Create plots of Lomb-Sacrgle periodograms."""

    periodograms = dict()
    for sector, sector_data in periodogram_data.items():
#        pyplot.plot(lightcurve['TIME'], lightcurve['SAP_FLUX'], label='raw')
#        pyplot.plot(lightcurve['TIME'][usable],
#                    corrected_flux[usable],
#                    label='corrected')
#        pyplot.legend()
#        pyplot.suptitle('Sector: ' + repr(sector))
#        pyplot.show()

#        if sector <= 15:
#            continue

        lomb_scargle = numpy.empty((4, 1000))
        lomb_scargle[0] = numpy.linspace(0.1, 10, 1000)

        lomb_scargle[1] = LombScargle(
            sector_data['times'],
            sector_data['raw_flux'] - numpy.mean(sector_data['raw_flux'])
        ).power(frequency=1.0 / lomb_scargle[0])

        lomb_scargle[2] = LombScargle(
            sector_data['times'][sector_data['remove_eclipse_mask']],
            (
                sector_data['raw_flux'][sector_data['remove_eclipse_mask']]
                -
                numpy.mean(
                    sector_data['raw_flux'][sector_data['remove_eclipse_mask']]
                )
            )
        ).power(frequency=1.0 / lomb_scargle[0])

        lomb_scargle[3] = LombScargle(
            sector_data['times'][sector_data['usable_corrected_mask']],
            (
                sector_data[
                    'corrected_flux'
                ][
                    sector_data['usable_corrected_mask']
                ]
                -
                numpy.mean(
                    sector_data[
                        'corrected_flux'
                    ][
                        sector_data['usable_corrected_mask']
                    ]
                )
            )
        ).power(frequency=1.0 / lomb_scargle[0])

        periodograms[sector] = lomb_scargle


#        pyplot.plot(lomb_scargle[0], lomb_scargle[1], '-r', label='raw')
        pyplot.plot(lomb_scargle[0],
                    lomb_scargle[2],
                    '-',
                    label='no eclispe Sector ' + repr(sector))

        pyplot.plot(lomb_scargle[0],
                    lomb_scargle[3],
                    '-',
                    label='corrected Sector ' + repr(sector))

        if params:
            for i in range(1, 10):
                pyplot.axvline(params['Per'] / i,
                               color='black')
                pyplot.text(
                    x=params['Per'] / i,
                    y=pyplot.ylim()[1],
                    s=('Porb' + ('' if i==1 else ' / ' + str(i))),
                    verticalalignment='bottom',
                    horizontalalignment='center'
                )

        pyplot.legend()
        pyplot.show()


def plot_wavelet_map(times,
                     fluxes,
                     time_step,
                     orbital_period,
                     title,
                     *,
                     wavelet_order=6):
    """Calculate and display the wavelet map."""

    wavelet_periods = numpy.linspace(0.1, 10, 100)
    wavelet_widths = (wavelet_order
                      *
                      wavelet_periods
                      /
                      time_step
                      /
                      (2.0 * numpy.pi))
    wavelet_map = signal.cwt(fluxes - numpy.mean(fluxes),
                             signal.morlet2,
                             wavelet_widths,
                             w=wavelet_order)
    pyplot.pcolormesh(times,
                      wavelet_periods,
                      numpy.abs(wavelet_map),
                      cmap='viridis',
                      shading='gouraud')
    pyplot.axhline(y=orbital_period, color='black')
    pyplot.title(title)


def make_wavelet_figures(periodogram_data, params, wavelet_order=6):
    """Create plots of Wavelet maps."""

    for sector, sector_data in periodogram_data.items():
        print('Sector: ' + repr(sector))
        valid_points = numpy.logical_and(
            sector_data['remove_eclipse_mask'],
            sector_data['raw_flux'] > 0.98
        )
        times, fluxes, time_step = resample_lc(
            sector_data['times'][valid_points],
            sector_data['raw_flux'][valid_points]
        )
        pyplot.subplot(121)
        plot_wavelet_map(times,
                         fluxes,
                         time_step,
                         params['Per'],
                         'Transits masked',
                         wavelet_order=wavelet_order)
        times, fluxes, time_step = resample_lc(
            sector_data['times'][sector_data['usable_corrected_mask']],
            sector_data['corrected_flux'][sector_data['usable_corrected_mask']]
        )
        pyplot.subplot(122)
        plot_wavelet_map(times,
                         fluxes,
                         time_step,
                         params['Per'],
                         'Corrected',
                         wavelet_order=wavelet_order)
        pyplot.suptitle('Sector: ' + repr(sector))
        pyplot.show()


def main(config):
    """Avoid polluting global namespace."""

    params = get_eb_params(config.tic, 'prsa')
    periodogram_data = get_periodogram_data(config, params)
    make_lsp_figures(periodogram_data, params)
    make_wavelet_figures(periodogram_data, params)

if __name__ == '__main__':
    main(parse_command_line())
