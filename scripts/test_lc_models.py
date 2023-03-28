#!/usr/bin/env python3

"""Compare BATMAN and PHOEBE transit models."""

from os import path

from matplotlib import pyplot
import numpy
from scipy.linalg import lstsq
from astropy import units
import phoebe

from general_purpose_python_modules.cmd_utils import CMDInterpolator

from phoebe_to_batman import get_batman_lc


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

    phoebe_binary.run_compute()
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


def calc_approximation_diff(primary_params, secondary_params, orbit):
    """Calc. approximation errors for both phase curve models given params"""

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

    print('Max diff: ' + repr(calc_max_lc_difference(phoebe_binary, plot=True)))


def explore_batman_approximation(
    *,
    stellar_masses,
    orbital_periods,
    eccentricities,
    inclinations,
    periapses,
    cmd_isochrone=path.join(path.dirname(path.dirname(path.abspath(__file__))),
                            'data',
                            'cmd_isochrone_1Gyr.dat')
):
    """Calculate deviations from PHOEBE LC on a grid of system parameters."""

    orbits = numpy.empty(
        (
            len(orbital_periods)
            *
            len(eccentricities)
            *
            len(inclinations)
            *
            len(periapses)
        ),
        dtype=[('period', float),
               ('eccentricity', float),
               ('inclination', float),
               ('periapsis', float)]
    )
    orbit_i = 0
    for porb in orbital_periods:
        for ecc in eccentricities:
            for incl in inclinations:
                for per in periapses:
                    orbits[orbit_i] = porb, ecc, incl, per
                    orbit_i += 1

    interpolator = CMDInterpolator(cmd_isochrone)
    stellar_params = numpy.empty(
        len(stellar_masses),
        dtype=[('mass', float), ('logg', float), ('teff', float)]
    )
    stellar_params['mass'] = stellar_masses
    stellar_params['logg'] = interpolator.get_interpolated(
        'logg',
        stellar_masses,
        0.0
    )
    stellar_params['teff'] = 10.0**interpolator.get_interpolated(
        'logTe',
        stellar_masses,
        0.0
    )
    calc_approximation_diff(stellar_params[0],
                            stellar_params[1],
                            orbits[0])


def main():
    """Avoid polluting global namespace."""

    explore_batman_approximation(
        stellar_masses=numpy.array([2.0, 1.5, 1.0, 0.5]),
        orbital_periods=[3.0],
        eccentricities=[0.4],
        periapses=[90.0],
        inclinations=[85.0]
    )


if __name__ == '__main__':
    main()
