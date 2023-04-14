#!/usr/bin/env python3

"""Create a plot comparing model to actual data for a single LC."""

from os import path
from multiprocessing import Pool

from matplotlib import pyplot
import numpy
from scipy.optimize import brentq, minimize
from configargparse import ArgumentParser, DefaultsFormatter
from astropy.io import fits
from astropy import units
import phoebe

from general_purpose_python_modules.cmd_utils import CMDInterpolator

from download_lcs import get_astroquery as get_lc, get_eb_params
from data_fnames import data_fnames
from phoebe_to_batman import get_batman_lc
from compare_lc_models import fit_phase_curve

def parse_command_line():
    """Return command line configuration."""

    parser = ArgumentParser(
        description='Fit BATMAN + phase curve model to a TESS LC.',
        default_config_files=['fit_lc.cfg'],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--tic',
        type=int,
        default=33419790,
        help='The TIC identifier of the LC to fit.'
    )
    parser.add_argument(
        '--lc-provenance',
        default='SPOC',
        choices=['SPOC', 'QLP'],
    )
    parser.add_argument(
        '--phoebe-model-fname',
        default='phoebe_models.npy'
    )
    return parser.parse_args()


def get_phobe_binary(params, cmd_interpolator):
    """Configure a PHOEBE binary per the given J&A (2021) parameters."""

    def mass_errfunc(mass, component):
        return (
            cmd_interpolator.get_interpolated('logTe', mass, 0.0)
            -
            numpy.log10(
                float(
                    params['Teff1' if component == 'primary' else 'Teff2']
                )
            )
        )

    binary = phoebe.default_binary()

    binary['primary']['teff'].set_value(params['Teff1'] * units.K)
    binary['secondary']['teff'].set_value(params['Teff2'] * units.K)

    orbit = binary['orbit']
    orbit['period'].set_value(params['Per'] * units.day)

    orbit['incl'].set_value(params['inc'] * units.deg)

    orbit['ecc'].set_value(
        (params['ecosw']**2 + params['esinw']**2)**0.5
    )
    orbit['per0'].set_value(numpy.arctan2(params['esinw'], params['ecosw']))

    binary.flip_constraint('mass@primary', solve_for='sma')
    binary.flip_constraint('mass@secondary', solve_for='q')
    for component in ['primary', 'secondary']:
        mass = brentq(mass_errfunc, 0.1, 1.8538578749, args=(component,))
        binary[component]['mass'].set_value(mass)

    binary['primary']['requiv'].set_value(orbit['sma'].get_value()
                                          /
                                          params['a_R1'])
    binary['secondary']['requiv'].set_value(
        binary['primary']['requiv'].get_value() * params['rp']
    )

    orbit['t0_supconj'].set_value(params['t1'] * units.day)
    return binary


    #('TIC', '>i4'),
    #('Per', '>f8'),
    #('t1', '>f8'),
    #('rp', '>f8'),
    #('a_R1', '>f8'),
    #('ecosw', '>f8'),
    #('esinw', '>f8'),
    #('inc', '>f8'),
    #('fp', '>f8'),
    #('Teff1', '>i4'),
    #('Teff2', '>i4'),
    #('f_Teff', 'u1')


def get_phoebe_model(tic, phoebe_binary, model_fname):
    """Return PHOEBE orbit + LC model pre-comuted if available or run rew."""

    if model_fname is not None and path.exists(model_fname):
        with open(model_fname, 'rb') as model_f:
            while True:
                try:
                    stored_tic = numpy.load(model_f)
                except ValueError:
                    break
                if stored_tic == tic:
                    return dict(
                        times=numpy.load(model_f),
                        rvs=numpy.load(model_f),
                        depths=numpy.load(model_f),
                        distances=numpy.load(model_f),
                        fluxes=numpy.load(model_f)
                    )
                for _ in range(5):
                    numpy.load(model_f)

    try:
        phoebe_binary.run_compute()
    except ValueError:
        print(repr(phoebe_binary.run_failed_constraints()))
        raise

    primary_pos = {
        coord: phoebe_binary.get('orb@primary@' + coord).get_value('R_sun')
        for coord in ['us', 'vs', 'ws']
    }
    result = dict(
        times=phoebe_binary.get(
            'times@lc01@phoebe01@latest@lc@model'
        ).get_value(units.day),
        rvs=phoebe_binary.get(
            'rvs@primary@rv01@phoebe01@latest@rv@model'
        ).get_value('km/s'),
        depths=primary_pos['ws'],
        distances=numpy.sqrt(primary_pos['us']**2
                             +
                             primary_pos['vs']**2
                             +
                             primary_pos['ws']**2),
        fluxes=phoebe_binary.get(
            'fluxes@lc01@phoebe01@latest@lc@model'
        ).get_value()
    )

    if model_fname is not None:
        with open(model_fname, 'ab') as model_f:
            numpy.save(model_f, tic)
            for param in ['times', 'rvs', 'depths', 'distances', 'fluxes']:
                numpy.save(model_f, result[param])

    return result


def interpolate_phoebe_model(model, porb, times):
    """Interpolate the phoebe model quantities to the given times."""

    interp_times = times % porb
    return {
        q: (model['times'] if q == 'times'
            else numpy.interp(interp_times,
                              model['times'],
                              model[q],
                              period=porb))
        for q in model.keys()
    }


def run_phoebe(params,
               interp_times,
               cmd_interpolator,
               eval_resolution=1e-2):
    """Run a PHOEBE calculation and return the interpolated model."""

    phoebe_binary = get_phobe_binary(params, cmd_interpolator)
    phoebe_binary.add_dataset('rv', dataset='rv01')
    phoebe_binary.add_dataset('orb') #LTT?
    phoebe_binary.add_dataset('lc', dataset='lc01')
    for component in ['primary', 'secondary']:
        phoebe_binary['ld_mode@lc01@' + component].set_value('lookup')
        phoebe_binary['ld_func@lc01@' + component].set_value('quadratic')
    eval_times = numpy.arange(0.0, params['Per'], eval_resolution)
    phoebe_binary.set_value_all('times', eval_times)
    phoebe_binary.set_value_all('compute_times', eval_times)
    phoebe_model = interpolate_phoebe_model(
        get_phoebe_model(None, phoebe_binary, None),
        params['Per'],
        interp_times
    )
    return phoebe_model


def get_lc_model(params, lightcurve, phoebe_model, cmd_interpolator):
    """
    Evaluate the LC model at the observation times in the given LC.

    If the params argument specifies ``'fit'`` as the value of ``'fp'`` (flux
    ratio) a combined fit for that parameter along with the phase curve is
    performed using linear least squares. The best fit flux ratio is returned as
    a second return value.
    """

    phoebe_binary = get_phobe_binary(params, cmd_interpolator)
    phoebe_binary.add_dataset('lc', dataset='lc01')
    for component in ['primary', 'secondary']:
        phoebe_binary['ld_mode@lc01@' + component].set_value('lookup')
        phoebe_binary['ld_func@lc01@' + component].set_value('quadratic')

    print('\t\tGenerating BATMAN LC')
    with Pool(processes=1) as pool:
        batman_lc = pool.starmap(
            get_batman_lc,
            [
                (
                    phoebe_binary,
                    lightcurve['TIME'],
                    'lc01',
                    ('split' if params['fp'] == 'fit' else params['fp'])
                )
            ]
        )[0]

    if params['fp'] == 'fit':
        extra_lc_components = numpy.vstack(batman_lc).T
    else:
        out_of_eclipse = (batman_lc == numpy.max(batman_lc))
        batman_lc /= numpy.median(batman_lc[out_of_eclipse])
        extra_lc_components = None

        lightcurve['PDCSAP_FLUX'] /= numpy.median(
            lightcurve['PDCSAP_FLUX'][out_of_eclipse]
        )

    print('\t\tFitting phase curve')
    phase_lc = fit_phase_curve(
        (
            lightcurve['PDCSAP_FLUX'] if params['fp'] == 'fit'
            else lightcurve['PDCSAP_FLUX'] - batman_lc
        ),
        phoebe_model['rvs'],
        phoebe_model['depths'],
        phoebe_model['distances'],
        extra_lc_components
    )
    if params['fp'] == 'fit':
        return phase_lc[0], float(phase_lc[1][1]) / float(phase_lc[1][0])
    model_lc = batman_lc + phase_lc
    return model_lc / numpy.median(model_lc[out_of_eclipse])


def fit_lc_model(initial_params,
                 lightcurve,
                 cmd_interpolator,
                 hold_fixed=('Per', 't1', 'Teff1', 'Teff2'),
                 **minimize_config):
    """Find best fit parameters and return model LC."""

    def to_minimize_inner(x, fixed, phoebe_model):
        """Function to minimize while holding (esinw, ecosw, inc) fixed."""

        x_iter = iter(x)
        params = dict(fixed)
        for quantity in ['Per', 't1', 'rp', 'a_R1', 'Teff1', 'Teff2']:
            if quantity not in params:
                params[quantity] = next(x_iter)

        if 'fp' not in params:
            params['fp'] = 'fit'

        print('\tInner fixed params: ' + repr(params))

        eccentricity = numpy.sqrt(params['esinw']**2 + params['ecosw']**2)
        if (
            params['rp'] < 0.7
            or
            params['rp'] > 1.5
            or
            (
                params['a_R1'] * (1.0 - eccentricity) / (1.0 + params['rp'])
                <
                1.0
            )
        ):
            print('\tInvalid parameters: ' + repr(params))
            return numpy.inf

        print('\tFitting LC model')
        model = get_lc_model(params, lightcurve, phoebe_model, cmd_interpolator)
        if params['fp'] == 'fit':
            model, params['fp'] = model

        residual = numpy.sqrt((lightcurve['PDCSAP_FLUX'] - model)**2).sum()
        print(
            (
                '\tFor a_R1 = {a_R1:20.16f}, rp = {rp:20.16f}, fp = {fp:20.16f}'
                ', inner residual: {residual!r}'
            ).format(
                **params,
                residual=residual
            )
        )
        if residual < to_minimize_inner.best_residual:
            print('\tNew best inner parameters found: ' + repr(params))
            to_minimize_inner.best_model = model
            to_minimize_inner.best_param = dict(params)
            to_minimize_inner.best_residual = residual
        return residual


    def to_minimize_outer(x):
        """Function to minimize to find (esinw, ecosw, inc)."""

        fixed_param = dict(esinw=x[0], ecosw=x[1], inc=x[2])
        phoebe_params = dict(initial_params)
        phoebe_params.update(fixed_param)

        eccentricity = numpy.sqrt(x[0]**2 + x[1]**2)
        if (
            eccentricity > 1
            or
            (
                initial_params['a_R1'] * (1.0 - eccentricity)
                <
                (1.0 + initial_params['rp'])
            )
        ):
            return numpy.inf

        inner_fit_params = ['Per', 't1', 'rp', 'a_R1', 'Teff1', 'Teff2']
        for quantity in hold_fixed:
            fixed_param[quantity] = initial_params[quantity]
            inner_fit_params.remove(quantity)

        phoebe_model = run_phoebe(phoebe_params,
                                  lightcurve['TIME'],
                                  cmd_interpolator)
        print('Performing inner optimization for: ' + repr(inner_fit_params))
        to_minimize_inner.best_residual = numpy.inf
        to_minimize_inner.best_model = None
        to_minimize_inner.best_param = None
        result = minimize(to_minimize_inner,
                          [initial_params[p] for p in inner_fit_params],
                          args=(fixed_param, phoebe_model),
                          **minimize_config)
        assert result.fun == to_minimize_inner.best_residual

        if to_minimize_inner.best_residual < to_minimize_outer.best_residual:
            print('New best parameters found: '
                  +
                  repr(to_minimize_inner.best_param))
            to_minimize_outer.best_model = to_minimize_inner.best_model
            to_minimize_outer.best_param = dict(to_minimize_inner.best_param)
            to_minimize_outer.best_residual = to_minimize_inner.best_residual
        return to_minimize_inner.best_residual

    to_minimize_outer.best_residual = numpy.inf
    to_minimize_outer.best_model = None
    to_minimize_outer.best_param = None
    result = minimize(to_minimize_outer,
                      [initial_params[p] for p in ['esinw', 'ecosw', 'inc']],
                      **minimize_config)
    print('Minimize result: ' + repr(result))
    return to_minimize_outer.best_model


def main(config):
    """Avoid polluting global namespace."""

    params = get_eb_params(config.tic)
    params = {k: params[k] if k=='TIC' else float(params[k])
              for k in params.dtype.names}
    lightcurve = get_lc(config.tic, 6, config.lc_provenance)
    lightcurve = lightcurve[
        numpy.logical_and(numpy.isfinite(lightcurve['TIME']),
                          numpy.isfinite(lightcurve['PDCSAP_FLUX']))
    ]
    lightcurve = lightcurve[lightcurve['TIME'] > 1480]
    cmd_interpolator = CMDInterpolator(data_fnames['cmd_isochrone'])

#    params = {'TIC': params['TIC'],
#              'Per': 2.016328992,
#              't1': 1468.611614,
#              'rp': 1.3931383117071463,
#              'a_R1': 6.3700741680447095,
#              'ecosw': -7.638199993427343e-08,
#              'esinw': 1.3068106610420261e-06,
#              'inc': 80.94762074,
#              'fp': 1.9199389147783785,
#              'Teff1': 6733.0,
#              'Teff2': 6635.0,
#              'f_Teff': 1.0}
#
#    params = {'TIC': params['TIC'],
#              'Per': 2.016328992,
#              't1': 1468.611614,
#              'rp': 1.3903087861320609,
#              'a_R1': 6.3652435372435185,
#              'ecosw': -0.0001730562592986993,
#              'esinw': 8.330374125507833e-07,
#              'inc': 80.94762074,
#              'fp': 1.9141808664800775,
#              'Teff1': 6733.0,
#              'Teff2': 6635.0,
#              'f_Teff': 1.0}

    model = fit_lc_model(params,
                         lightcurve,
                         cmd_interpolator,
                         method='Nelder-Mead')

    pyplot.subplot(211)
    pyplot.plot(lightcurve['TIME'], lightcurve['PDCSAP_FLUX'])
    pyplot.plot(lightcurve['TIME'], model)
    pyplot.subplot(212)
    pyplot.plot(lightcurve['TIME'], lightcurve['PDCSAP_FLUX'] - model)

    pyplot.show()


if __name__ == '__main__':
    main(parse_command_line())
