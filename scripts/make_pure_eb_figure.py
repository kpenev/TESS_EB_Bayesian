#!/usr/bin/env python3

"""Create the figure showing a pure EB LC + model."""

from os import path

from matplotlib import pyplot
import numpy

from general_purpose_python_modules.cmd_utils import CMDInterpolator

from download_lcs import get_astroquery, match_lightcurves, get_eb_params
from data_fnames import plot_dir
from fit_lc import get_lc_model, run_phoebe
from data_fnames import data_fnames

def get_plot_data(tic, sector, exposure_shift, spoc_flux_column):
    """Return the SPOC and QLP lightcurvesl to plot filtered and matched."""

    qlp_lc = get_astroquery(tic, sector, 'QLP')
    spoc_lc = get_astroquery(tic, sector, 'SPOC')
    try:
        params = get_eb_params(tic)
        first_eclipse = params['t1']
        if len(params) == 0:
            params = get_eb_params(tic, 'prsa')
            first_eclipse = params['BJD0']

        print('Params: ' + repr(params))
    except:
        params = {
            'BJD0': qlp_lc['TIME'][0],
            'Per': 1.0 / 0.87897
        }

    print('QLP LC: ' + repr(qlp_lc))
    print('SPOC LC: ' + repr(spoc_lc))

    spoc_lc = spoc_lc[
        numpy.logical_and(
            numpy.isfinite(spoc_lc['TIME']),
            numpy.isfinite(spoc_lc[spoc_flux_column])
        )
    ]
    time_offset = 100 * (spoc_lc['TIME'][0] // 100)
    qlp_lc['TIME'] -= time_offset
    spoc_lc['TIME'] -= time_offset
    first_eclipse -= time_offset

    qlp_lc, binned_spoc_flux, unbinned_spoc_flux = match_lightcurves(
        qlp_lc,
        spoc_lc,
        spoc_flux_column,
        exposure_shift
    )
    return dict(
        params=params,
        qlp_lc=qlp_lc,
        spoc_binned_flux=binned_spoc_flux,
        spoc_unbinned_flux=unbinned_spoc_flux,
        spoc_times=spoc_lc['TIME'],
        time_offset=time_offset,
        first_eclipse=first_eclipse
    )


def plot_spoc_qlp_comparison(tic,
                             sector,
                             zoom_y,
                             *,
                             exposure_shift=0,
                             bottom_left='diff'):
    """Create a plot comparing SPOC PDCSAP LC to QLP un-detrended LC."""

    spoc_flux_column = 'PDCSAP_FLUX'
    plot_data = get_plot_data(tic,
                              sector,
                              exposure_shift,
                              spoc_flux_column)

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
    legend_artists = []
    for plot_ax in flux_axes:
        legend_artists.extend([
            plot_ax.plot(plot_data['spoc_times'],
                         plot_data['spoc_unbinned_flux'],
                         '.',
                         markeredgecolor='lightgrey',
                         markerfacecolor='lightgrey',
                         label=('SPOC ' + spoc_flux_column[:-5] if label
                                else None),
                         zorder=10)[0],
            plot_ax.plot(plot_data['qlp_lc']['TIME'],
                         plot_data['spoc_binned_flux'][:, 0],
                         'xr',
                         label=('binned SPOC ' + spoc_flux_column[:-5] if label
                                else None),
                         zorder=20)[0],
            plot_ax.plot(plot_data['qlp_lc']['TIME'],
                         plot_data['qlp_lc']['SAP_FLUX'],
                         '.g',
                         markersize=4,
                         label=('QLP SAP' if label else None),
                         zorder=30)[0]
        ])
        if not label:
            legend_artists = legend_artists[:-3]
        label = False
    axes[0, 0].set_ylabel('Flux')

    axes[0, 1].set_xlim(80.5, 80.5 + plot_data['params']['Per'])
#    axes[0, 1].set_yticks([])

    zoom_x_min = (0.4 * plot_data['spoc_times'][0]
                  +
                  0.6 * plot_data['spoc_times'][-1])
    zoom_x_min = (
        (
            (zoom_x_min - plot_data['first_eclipse'])
            //
            plot_data['params']['Per']
            +
            0.25
        )
        *
        plot_data['params']['Per']
        +
        plot_data['first_eclipse']
    )
    axes[1, 1].set_xlim(zoom_x_min, zoom_x_min + plot_data['params']['Per'])
    axes[1, 1].set_ylim(zoom_y)
    axes[1, 1].tick_params(axis='y', left=False, right=True)

    axes[0, 1].yaxis.tick_right()
    axes[1, 1].yaxis.tick_right()

    if bottom_left == 'diff':
        axes[1, 0].plot(
            plot_data['qlp_lc']['TIME'],
            (
                plot_data['spoc_binned_flux'][:, 0]
                -
                plot_data['qlp_lc']['SAP_FLUX']
            ),
            '.k',
            label='SPOC PDCSAP - QLP SAP'
        )
#    pyplot.ylim(-1.5e-3, 2e-3)
    figure.suptitle('BJD - %d [days]' % (2457000 + plot_data['time_offset']),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    x=0.5,
                    y=0.0)
    axes[1, 0].legend(markerscale=0,
                      frameon=True,
                      facecolor='white',
                      edgecolor='none',
                      framealpha=1)
    return axes, plot_data, legend_artists


def main(tic, sector):
    """Avoid polluting global namespace."""

    axes, plot_data, legend_artists = plot_spoc_qlp_comparison(tic,
                                                               sector,
                                                               (0.99, 1.015))
    print('Legend artists: ' + repr(legend_artists))

    if tic == 33419790:
        params = {'esinw': 1.8497156069500608e-09,
                  'ecosw': -7.325177690070932e-10,
                  'inc': 99.98818807712863,
                  'Per': 2.016328992,
                  't1': 68.611614,
                  'Teff1': 6733.0,
                  'Teff2': 6635.0,
                  'rp': 1.0427699231863137,
                  'a_R1': 5.348155406166791,
                  'fp': 'fit'}
        cmd_interpolator = CMDInterpolator(data_fnames['cmd_isochrone'])
        phoebe_model = run_phoebe(params,
                                  plot_data['spoc_times'],
                                  cmd_interpolator)
        lc_model, params['fp'] = get_lc_model(
            params,
            dict(
                TIME=plot_data['spoc_times'],
                PDCSAP_FLUX=plot_data['spoc_unbinned_flux']
            ),
            phoebe_model,
            cmd_interpolator
        )
        legend_artists.append(
            axes[0, 1].plot(plot_data['spoc_times'],
                            lc_model,
                            '-k',
                            linewidth=0.7,
                            zorder=45,
                            label='model')[0]
        )
        axes[1, 1].plot(plot_data['spoc_times'],
                        lc_model,
                        '-k',
                        linewidth=0.7,
                        zorder=45)


    print('Figure labels: ' + repr(pyplot.get_figlabels()))
    pyplot.figlegend(handles=legend_artists,
                     ncol=4,
                     loc='upper center',
                     borderaxespad=0)

    pyplot.show()
#    pyplot.savefig(
#        path.join(
#            plot_dir,
#            'tic%016d_s%06d_qlp_spoc_comparison.pdf' % (tic, sector)
#        )
#    )


if __name__ == '__main__':
#    main(443768508, 32)
    main(33419790, 6)
