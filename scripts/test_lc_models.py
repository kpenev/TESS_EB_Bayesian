#!/usr/bin/env python3

"""Compare BATMAN and PHOEBE transit models."""

from os import path
from multiprocessing import Process, Queue
from hashlib import sha1 as make_hasher
import logging

from matplotlib import pyplot
import numpy
from scipy.linalg import lstsq
from astropy import units
import phoebe
from configargparse import ArgumentParser, DefaultsFormatter

from general_purpose_python_modules.cmd_utils import CMDInterpolator

from phoebe_to_batman import get_batman_lc

_logger = logging.getLogger(__name__)


def parse_command_line():
    """Return command line configuration."""

    parser = ArgumentParser(
        description=('Compare LC models built with PHOEBE to models combining '
                     'BATMAN with few parameter phase curve models'),
        default_config_files=['test_lc_models.cfg'],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--stellar-masses',
        nargs='+',
        type=float,
        default=numpy.array([2.0, 1.5, 1.0, 0.5]),
        help='The stellar masses to use in building binaries (all possible '
        'combinations are evaluated).'
    )
    parser.add_argument(
        '--orbital-periods',
        nargs='+',
        type=float,
        default=2.0**numpy.arange(6, dtype=float),
        help='The orbital periods to compare models at combined in all possible'
        ' ways with stellar masses.'
    )
    parser.add_argument(
        '--eccentricities',
        nargs='+',
        type=float,
        default=numpy.linspace(0.0, 0.8, 9),
        help='The eccentricities to compare models at combined in all possible'
        ' ways with stellar masses and orbital periods.'
    )
    parser.add_argument(
        '--periapses',
        nargs='+',
        type=float,
        default=numpy.arange(0.0, 360.0, 30.0),
        help='The periapses angles in degrees to compare models at combined in '
        'all possible ways with stellar masses, orbital periods, and '
        'eccentricities.'
    )
    parser.add_argument(
        '--inclinations',
        nargs='+',
        type=float,
        default=(90.0 - numpy.linspace(0.0, 15.0, 4)),
        help='The inclination angles to compare models at combined in all '
        'possible ways with stellar masses, orbital periods, eccentricities, '
        'and periapsis angles.'
    )
    parser.add_argument(
        '--cmd-isochrone',
        default=path.join(path.dirname(path.dirname(path.abspath(__file__))),
                          'data',
                          'cmd_isochrone_1Gyr.dat'),
        help='The filename of an isochrone downloaded from the CMD website to '
        'use to find stellar parameters given mass.'
    )
    parser.add_argument(
        '--progress-fname',
        default='test_lc_models_progress.npy',
        help='Filename to save computed differences between the two LC models. '
        'Uses numpy.save multiple times.'
    )
    parser.add_argument(
        '--report-progress-only',
        action='store_true',
        help='If passed only reports the number of scenarios for which '
        'residuals have been computed and stored in the progress file.'
    )
    parser.add_argument(
        '--num-parallel-processes',
        type=int,
        default=16,
        help='How many multiprocessing processes to use.'
    )
    parser.add_argument(
        '--plot-vs',
        nargs='+',
        default=['porb', 'e'],
        choices=['m1', 'm2', 'msum', 'mratio', 'porb', 'e', 'i', 'periapsis'],
        help='What to plot the LC differences vs.'
    )
    parser.add_argument(
        '--plot-residuals',
        choices=['both', 'beer', 'orbit'],
        default='orbit',
        help='Select the phase curve model(s) for which to plot the LC '
        'residuals.'
    )
    parser.add_argument(
        '--plot-fname-format',
        default='lc_diff_vs_{vs:s}_m1{m1:s}_m2{m2:s}_i{i:s}.pdf',
        help='A format string specifying filenames to save requested plots as. '
        'Should include a `{vs}` substitution that will be replaced with '
        'the quantity shown on the x axis may also include substitutions for '
        'any of the quantities residuals can be plotted vs to split plots by '
        'some sub-set of of the available quantities to avoid too many lines '
        'in one plot.'
    )
    return parser.parse_args()


def fit_phase_curve(lc_residuals, rvs, depth, distance):
    """
    Fit for the free parameters of a BEER-like model.

    The assumptions are that the phase curve contains effects linearly
    proportional to:

      * the radial velocity of the primary: Doppler boosting and beaming

      * sin or cos of twice the angle between the line of sight and the
        coordinate of the primary relative to the center of mass divided by the
        cube of the distance between the objects: ellipsoidal variations

      * sin or cos of the angle between the line of sight and the coordinate of
        the primary relative to the center of mass divided by the square of the
        distance between the objects: reflection effect

    Args:
        lc_residuals(array):    The differences between the PHOEBE and BATMAN
            lightcurves at the evaluation times

        rvs(array):    Radial velocity of the primary relative to the center of
            mass of the system at the evaluation times.

        depth(array):    The coordinate of the primary in the direction
            perpendicular to the sky relative to the center of mass of the
            system at the evaluation times.

        distance(array):    The distance between the primary and the center of
            mass at the evaluation times.

    Returns:
        array:
            Best fit model of the lightcurve residuals.
    """

    lhs = numpy.empty((lc_residuals.size, 6))
    los_angle = numpy.arccos(depth / distance)
    lhs[:, 0] = rvs
    lhs[:, 1] = numpy.cos(2.0 * los_angle) / distance**3
    lhs[:, 2] = numpy.sin(2.0 * los_angle) / distance**3

    lhs[:, 3] = numpy.cos(los_angle) / distance**2
    lhs[:, 4] = numpy.sin(los_angle) / distance**2

    lhs[:, 5] = 1.0
    coef = lstsq(lhs, lc_residuals)[0]
    return lhs.dot(coef)


def fit_beer(lc_phases, lc_residuals):
    """Find best fit 4-component BEER model to LC residuals."""

    lhs = numpy.empty((lc_phases.size, 4))
    lhs[:, 0] = numpy.cos(4.0 * numpy.pi * lc_phases)
    lhs[:, 1] = numpy.cos(2.0 * numpy.pi * lc_phases)
    lhs[:, 2] = numpy.sin(4.0 * numpy.pi * lc_phases)
    lhs[:, 3] = numpy.sin(2.0 * numpy.pi * lc_phases)
    lhs[:, 3] = 1.0
    coef = lstsq(lhs, lc_residuals)[0]
    return lhs.dot(coef)


def plot_lc_model_comparison(plot_times,
                             phoebe_lc,
                             batman_lc,
                             beer_lc,
                             phase_lc):
    """Generate plots comparing PHOBE vs BATMAN + two phase curve models."""

    pyplot.subplot(311)
    pyplot.plot(plot_times, phoebe_lc, 'og', label='PHOEBE')
    pyplot.plot(plot_times, batman_lc, 'or', label='BATMAN')
    pyplot.axhline(y=1)
    pyplot.legend()

    pyplot.subplot(312)
    pyplot.plot(plot_times, phoebe_lc - batman_lc, 'or', label='PHOEBE-BATMAN')
    pyplot.plot(plot_times, beer_lc, 'og', label='BEER')
    pyplot.plot(plot_times, phase_lc, 'ob', label='PHASE')

    pyplot.legend()

    pyplot.subplot(313)
#    pyplot.plot(plot_times,
#                phoebe_lc - (batman_lc + beer_lc),
#                'og',
#                label='BEER residuals')

    pyplot.plot(plot_times,
                phoebe_lc - (batman_lc + phase_lc),
                'or',
                label='phase residuals')
    pyplot.legend()

    pyplot.show()


def calc_max_lc_difference(phoebe_binary, plot=False):
    """Return maximum difference between PHOEBE and BATMAN + phase curve LCs."""

    phoebe_binary.add_dataset('rv', dataset='rv01')
    phoebe_binary.add_dataset('orb') #LTT?

    eval_times = phoebe.linspace(
        0,
        phoebe_binary['orbit@period'].get_value(units.day),
        101
    )
    phoebe_binary.set_value_all('times', eval_times)
    phoebe_binary.set_value_all('compute_times', eval_times)

    try:
        phoebe_binary.run_compute()
    except ValueError:
        return numpy.nan, numpy.nan

    phoebe_lc = (
        phoebe_binary.get('fluxes@lc01@phoebe01@latest@lc@model').get_value()
    )
    batman_lc = get_batman_lc(phoebe_binary)

    batman_residuals = phoebe_lc - batman_lc
    beer_lc = fit_beer(
        eval_times / phoebe_binary['orbit@component@period'].get_value(),
        batman_residuals
    )

    primary_pos = {
        coord: phoebe_binary.get('orb@primary@' + coord).get_value('R_sun')
        for coord in ['us', 'vs', 'ws']
    }
    phase_lc = fit_phase_curve(
        batman_residuals,
        phoebe_binary.get(
            'rvs@primary@rv01@phoebe01@latest@rv@model'
        ).get_value('km/s'),
        primary_pos['ws'],
        (
            primary_pos['us']**2
            +
            primary_pos['vs']**2
            +
            primary_pos['ws']**2
        )**0.5
    )

    if plot:
        plot_lc_model_comparison(eval_times,
                                 phoebe_lc,
                                 batman_lc,
                                 beer_lc,
                                 phase_lc)

    return (
        (phoebe_lc - (batman_lc + beer_lc)).max(),
        numpy.abs(phoebe_lc - (batman_lc + phase_lc)).max() / batman_lc.max()
    )


def calc_approximation_diff(param_queue, result_queue):
    """Calc. approximation errors for both phase curve models given params"""

    for (
        param_hash,
        primary_params,
        secondary_params,
        orbit
    ) in iter(param_queue.get, 'STOP'):
        phoebe_binary = phoebe.default_binary()

        phoebe_binary.flip_constraint('mass@primary', solve_for='sma')
        phoebe_binary.flip_constraint('mass@secondary', solve_for='q')
        phoebe_binary.flip_constraint('logg@primary', solve_for='requiv')
        phoebe_binary.flip_constraint('logg@secondary', solve_for='requiv')

        for component, params in [('primary', primary_params),
                                  ('secondary', secondary_params)]:
            star = phoebe_binary[component]
            star['mass'].set_value(params['mass'] * units.M_sun)
            star['teff'].set_value(params['teff'] * units.K)
            star['logg'].set_value(params['logg'])

        phoebe_orbit = phoebe_binary['orbit']
        phoebe_orbit['period'].set_value(orbit['period'] * units.day)
        phoebe_orbit['ecc'].set_value(orbit['eccentricity'])
        phoebe_orbit['incl'].set_value(orbit['inclination'] * units.deg)
        phoebe_orbit['per0'].set_value(orbit['periapsis'] * units.deg)

        phoebe_binary.add_dataset('lc', dataset='lc01')
        for component in ['primary', 'secondary']:
            phoebe_binary['ld_mode@lc01@' + component].set_value('lookup')
            phoebe_binary['ld_func@lc01@' + component].set_value('quadratic')

        result_queue.put(
            (
                param_hash,
                primary_params,
                secondary_params,
                orbit,
                calc_max_lc_difference(phoebe_binary)
            )
        )


#Simple enough
#pylint: disable=too-many-locals
#pylint: disable=too-many-branches
def explore_batman_approximation(param_grids,
                                 num_parallel_processes,
                                 cmd_isochrone,
                                 progress_fname,
                                 progress_only):
    """Calculate deviations from PHOEBE LC on a grid of system parameters."""

    def get_orbits():
        """Create an array with the orbital elements to explore."""

        num_grid_points = 1
        for grid in param_grids.values():
            num_grid_points *= len(grid)
        result = numpy.empty(
            num_grid_points,
            dtype=[('period', float),
                   ('eccentricity', float),
                   ('inclination', float),
                   ('periapsis', float)]
        )
        orbit_i = 0
        for porb in param_grids['orbital_periods']:
            for ecc in param_grids['eccentricities']:
                for incl in param_grids['inclinations']:
                    for per in param_grids['periapses']:
                        result[orbit_i] = porb, ecc, incl, per
                        orbit_i += 1
        return result

    def get_stellar_params():
        """Create an array with the stellar parameters to explore."""

        interpolator = CMDInterpolator(cmd_isochrone)
        result = numpy.empty(
            len(param_grids['stellar_masses']),
            dtype=[('mass', float), ('logg', float), ('teff', float)]
        )
        result['mass'] = param_grids['stellar_masses']
        result['logg'] = interpolator.get_interpolated('logg',
                                                       result['mass'],
                                                       0.0)
        result['teff'] = 10.0**interpolator.get_interpolated('logTe',
                                                             result['mass'],
                                                             0.0)
        return result


    orbit_params = get_orbits()
    stellar_params = get_stellar_params()

    progress = set()
    if path.exists(progress_fname):
        with open(progress_fname, 'rb') as progress_f:
            while True:
                try:
                    progress.add(bytes(numpy.load(progress_f)))
                except ValueError:
                    break
                for _ in range(4):
                    numpy.load(progress_f)
    _logger.info('Found %d/%d evolutions in progres file.',
                 len(progress),
                 stellar_params.size**2 * orbit_params.size)
    if progress_only:
        return

    num_jobs = 0
    param_queue = Queue()
    result_queue = Queue()
    for primary_i in range(stellar_params.size):
        for secondary_i in range(primary_i, stellar_params.size):
            for orbit in orbit_params:
                hasher = make_hasher()
                hasher.update(stellar_params[primary_i].data.tobytes())
                hasher.update(stellar_params[secondary_i].data.tobytes())
                hasher.update(orbit.data.tobytes())
                config_hash = hasher.digest()
                if config_hash not in progress:
                    param_queue.put(
                        (
                            config_hash,
                            stellar_params[primary_i],
                            stellar_params[secondary_i],
                            orbit
                        )
                    )
                    num_jobs += 1

    for _ in range(num_parallel_processes):
        param_queue.put('STOP')

    workers = [
        Process(
            target=calc_approximation_diff,
            args=(param_queue, result_queue)
        )
        for _ in range(num_parallel_processes)
    ]
    for process in workers:
        process.start()
    with open(progress_fname, 'ab') as progress_f:
        for _ in range(num_jobs):
            result = result_queue.get()
            for entry in result:
                numpy.save(progress_f, entry)
#pylint: enable=too-many-locals
#pylint: enable=too-many-branches


def read_plot_data(progress_fname,
                   plot_vs,
                   plot_fname_format,
                   param_grids):
    """Read and organize the pre-computed data for plotting."""

    config_order = ['m1', 'm2', 'porb', 'e', 'i', 'periapsis']
    param_grid_order = ['stellar_masses',
                        'stellar_masses',
                        'orbital_periods',
                        'eccentricities',
                        'inclinations',
                        'periapsis']

    def get_config_indices(configuration, skip_quantity):
        """Return tuple of indices within param grids given configuration."""

        result = [
            numpy.argwhere(param_grids[param_q] == configuration[config_q])
            for param_q, config_q in zip(param_grid_order,
                                         config_order)
        ]
        del result[config_order.index(skip_quantity)]
        return result


    plot_data = dict()
    with open(progress_fname, 'rb') as progress_f:
        while True:
            try:
                numpy.load(progress_f)
            except ValueError:
                break
            data = {par_name: numpy.load(progress_f)
                    for par_name in ['primary', 'secondary', 'orbit']}
            data['residuals'] = numpy.load(progress_f)
            configuration = dict(
                m1=float(data['primary']['mass']),
                m2=float(data['secondary']['mass']),
                porb=float(data['orbit']['period']),
                e=float(data['orbit']['eccentricity']),
                i=float(data['orbit']['inclination']),
                periapsis=float(data['orbit']['periapsis'])
            )
            configuration['msum'] = (configuration['m1']
                                     +
                                     configuration['m2'])
            configuration['mratio'] = (configuration['m2']
                                       /
                                       configuration['m1'])
            for x_quantity in plot_vs:
                plot_fname = plot_fname_format.format(**configuration,
                                                      vs=x_quantity)
                if plot_fname not in plot_data:
                    plot_data[plot_fname] = dict()
                config_indices = get_config_indices(configuration, x_quantity)
                if config_indices not in plot_data:
                    plot_data[plot_fname][config_indices] = dict(x=[],
                                                                 beer_diff=[],
                                                                 orbit_diff=[])
                target = plot_data[plot_fname][config_indices]
                target['x'].append(configuration[x_quantity])
                target['beer_diff'].append(data['residuals'][0])
                target['orbit_diff'].append(data['residuals'][1])
    return plot_data


def create_plots(progress_fname, plot_vs, plot_fname_format, param_grids):
    """
    Plot the dependenc of LC differences on various quantities.

    Args:
        See command line documentation for description of arguments.

    Returns:
        None
    """

    plot_data = read_plot_data(progress_fname,
                               plot_vs,
                               plot_fname_format,
                               param_grids)


def main(config):
    """Avoid polluting global namespace."""

    phoebe.progressbars_off()
    param_grids = dict(
        stellar_masses=config.stellar_masses,
        orbital_periods=config.orbital_periods,
        eccentricities=config.eccentricities,
        periapses=config.periapses,
        inclinations=config.inclinations,
    )

    explore_batman_approximation(
        param_grids,
        num_parallel_processes=config.num_parallel_processes,
        cmd_isochrone=config.cmd_isochrone,
        progress_fname=config.progress_fname,
        progress_only=config.report_progress_only
    )


if __name__ == '__main__':
    main(parse_command_line())
