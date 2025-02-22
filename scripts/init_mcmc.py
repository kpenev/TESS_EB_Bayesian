"""Methods for finding initial walker positions for MCMC."""

from collections import namedtuple
from multiprocessing import Process, Queue
import logging

import numpy
from scipy.stats import norm, uniform
from scipy import optimize
from astropy import units, constants
from astroquery.mast import Catalogs

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)
from general_purpose_python_modules.emcee_util import (
    save_initial_position,
    load_initial_positions,
)

from sample_params import SampleParams

_logger = logging.getLogger(__name__)

FixedParamType = namedtuple(
    "FixedParamType", ["per", "eclipse_time", "mprimary"]
)


def _get_optimize_start(log_likelihood, overwrite):
    """Return an initial starting point for optimization."""

    initial_state = norm.rvs(size=len(SampleParams._fields))
    mratio = float(uniform.rvs(size=1))
    for param in [
        "mtotal",
        "mratio",
        "per",
        "eclipse_time",
        "primary_impact_param",
    ]:
        if param == "primary_impact_param":
            ind, value = log_likelihood.inverse_prior(param, 0.0)
        elif param == "mtotal":
            ind, value = log_likelihood.inverse_prior(
                param, overwrite.mprimary * (1.0 + mratio)
            )
        elif param == "mratio":
            ind, value = log_likelihood.inverse_prior(param, mratio)
        else:
            ind, value = getattr(overwrite, param)
        initial_state[ind] = value
    return initial_state


def _get_bounds(log_likelihood, fixed_params):
    """Return the bounds to use when optimizing."""

    bounds = [(None, None) for _ in SampleParams._fields]

    for param in ["per", "eclipse_time"]:
        value = getattr(fixed_params, param)
        #        if param == "eclipse_time":
        #            value -= eclipse_time_offset
        ind, low = log_likelihood.inverse_prior(param, value[0] - value[1])
        high = log_likelihood.inverse_prior(param, value[0] + value[1])[1]
        bounds[ind] = (low, high)
    return bounds


def _optimize_starting_positions(
    fixed_params_queue, result_queue, log_likelihood, config
):
    """Find a position that maximizes log(prob) starting with given Porb."""

    numpy.random.seed()
    setup_process_map(vars(config))

    for fixed_params in iter(fixed_params_queue.get, "STOP"):
        _logger.info(
            "Looking for starting position with: %s", repr(fixed_params)
        )

        eclipse_time_offset = fixed_params.per[0] * numpy.round(
            fixed_params.eclipse_time[0] / fixed_params.per[0]
        )
        overwrite = FixedParamType(
            per=log_likelihood.inverse_prior("per", fixed_params.per[0]),
            eclipse_time=(
                log_likelihood.inverse_prior(
                    "eclipse_time", fixed_params.eclipse_time[0]
                )
                # - eclipse_time_offset
            ),
            mprimary=fixed_params.mprimary[0],
        )

        while True:
            initial_state = _get_optimize_start(log_likelihood, overwrite)
            blob = log_likelihood(initial_state)
            _logger.debug(
                "Found log-likelihood(%s) = %s", repr(initial_state), repr(blob)
            )
            if numpy.isfinite(blob).all():
                break
        result = optimize.minimize(
            lambda x: -log_likelihood(x)[0],
            x0=initial_state,
            method="Nelder-Mead",
            bounds=_get_bounds(log_likelihood, fixed_params),
            options={"adapt": True, "fatol": 100.0},
        )
        if not result.success:
            _logger.warning("Optimization did not converge: %s", repr(result))
        _logger.info(
            "log-likelihood(%s) = %s", repr(result.x), repr(result.fun)
        )
        result_queue.put(result.x)
    _logger.info('Starting position optimizanio process finished.')


def _estimate_mass(logg, teff):
    """Return an estimate of the stellar mass assuming main sequence star."""

    logg = numpy.atleast_1d(logg)
    teff = numpy.atleast_1d(teff)
    assert logg.shape == teff.shape
    light_to_mass = (
        4.0
        * numpy.pi
        * constants.sigma_sb
        * (teff * units.K) ** 4
        * constants.G
        / (10.0**logg * units.cm / units.s**2)
        * constants.M_sun
        / constants.L_sun
    ).to_value("")
    result = numpy.empty(logg.shape)
    in_range = light_to_mass < 0.075
    result[in_range] = 3.1 * light_to_mass[in_range] ** 0.77
    in_range = light_to_mass <= 8
    result[in_range] = light_to_mass[in_range] ** (1 / 3)
    in_range = light_to_mass > 8
    result[in_range] = 0.85 * light_to_mass[in_range] ** 0.4
    return result


def get_initial_mcmc_state(log_likelihood, config, samples_fname):
    """Get suitable initial state to start MCMC from."""

    starting_positions, positions_found = load_initial_positions(
        samples_fname, config.num_walkers, len(SampleParams._fields)
    )

    _logger.info(
        "Found %d starting positions, looking for %d additional suitable "
        "starting positions.",
        positions_found,
        config.num_walkers - positions_found,
    )
    fixed_param_queue = Queue()
    result_queue = Queue()

    # False positive
    # pylint: disable=no-member
    tic_entry = Catalogs.query_criteria(catalog="Tic", ID=config.tic_id)
    # pylint: enable=no-member

    mprimary = _estimate_mass(
        *numpy.meshgrid(
            float(tic_entry["logg"])
            + numpy.array(
                [float(-tic_entry["e_logg"]), 0.0, float(tic_entry["e_logg"])]
            ),
            float(tic_entry["Teff"])
            + numpy.array(
                [float(-tic_entry["e_Teff"]), 0.0, float(tic_entry["e_Teff"])]
            ),
        )
    )
    mprimary_uncertainty = (mprimary.max() - mprimary.min()) / 2
    mprimary = mprimary[1, 1]

    for walker_ind in range(positions_found, config.num_walkers):
        fixed_param_queue.put(
            FixedParamType(
                per=(
                    log_likelihood.best_fit_bls["period"]
                    * (1 + walker_ind % 2),
                    log_likelihood.best_fit_bls["period_uncertainty"]
                    * (1 + walker_ind % 2),
                ),
                eclipse_time=(
                    log_likelihood.best_fit_bls["transit_time"],
                    0.1
                    * log_likelihood.best_fit_bls["period"]
                    * (1 + walker_ind % 2),
                ),
                mprimary=(mprimary, mprimary_uncertainty),
            )
        )
    for _ in range(config.num_parallel):
        fixed_param_queue.put('STOP')

    workers = [
        Process(
            target=_optimize_starting_positions,
            args=(fixed_param_queue, result_queue, log_likelihood, config),
        )
        for _ in range(config.num_parallel)
    ]
    for process in workers:
        process.start()

    for position_ind in range(positions_found, config.num_walkers):
        starting_positions[position_ind] = result_queue.get()
        save_initial_position(
            starting_positions[position_ind],
            samples_fname,
            nwalkers=config.num_walkers,
        )

    return starting_positions
