"""Methods for finding initial walker positions for MCMC."""

from collections import namedtuple
from multiprocessing import Process, Queue
import logging
from traceback import format_exc

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

InitialParamType = namedtuple(
    "InitialParamType", ["per", "eclipse_time", "mprimary"]
)


# Meant to serve as callable.
# pylint: disable=too-few-public-methods
class OptimizeStartingPosition:
    """Callable to find initial position close local likelihood maxima."""

    def _get_optimize_start(self, overwrite):
        """Return an initial starting point for optimization."""

        initial_state = norm.rvs(size=self._num_optimize)
        mratio_range = self._log_likelihood.get_range("mratio")
        mratio = float(
            uniform.rvs(
                mratio_range[0], mratio_range[1] - mratio_range[0], size=1
            )
        )
        _logger.debug("Setting mratio to: %s", repr(mratio))
        for param in [
            "mtotal",
            "mratio",
            "per",
            "eclipse_time",
            "primary_impact_param",
        ]:
            if param == "primary_impact_param":
                ind, value = self._log_likelihood.inverse_prior(param, 0.0)
            elif param == "mtotal":
                ind, value = self._log_likelihood.inverse_prior(
                    param, overwrite.mprimary * (1.0 + mratio)
                )
            elif param == "mratio":
                ind, value = self._log_likelihood.inverse_prior(param, mratio)
            else:
                ind, value = getattr(overwrite, param)
            initial_state[self._mcmc_to_optimize[ind]] = value
        return initial_state

    def _get_bounds(self, initial_params):
        """Return the bounds to use when optimizing."""

        bounds = [(None, None) for _ in range(self._num_optimize)]

        for param in ["per", "eclipse_time"]:
            value = getattr(initial_params, param)
            ind, low = self._log_likelihood.inverse_prior(
                param, value[0] - value[1]
            )
            high = self._log_likelihood.inverse_prior(
                param, value[0] + value[1]
            )[1]
            bounds[self._mcmc_to_optimize[ind]] = (low, high)

        return bounds

    def _get_mcmc_sample(self, x):
        """Return MCMC sample given optimize position."""

        mcmc_sample = numpy.empty(len(SampleParams._fields))
        for ind, value in self._do_not_optimize:
            mcmc_sample[ind] = value

        for i, value in enumerate(x):
            mcmc_sample[self._optimize_to_mcmc[i]] = value
        return mcmc_sample

    def _to_optimize(self, x):
        """Return the negative log-likelihood for given optimize position."""

        return -self._log_likelihood(self._get_mcmc_sample(x))[0]

    def __init__(self, log_likelihood):
        """Prepare the callable."""

        self._log_likelihood = log_likelihood
        self._do_not_optimize = sorted(
            [
                (SampleParams._fields.index(param), value)
                for param, value in [
                    ("primary_prot", 10.0),
                    ("secondary_prot", 10.0),
                    ("primary_reflection_coef", -10.0),
                    ("secondary_reflection_coef", -10.0),
                    ("primary_beaming_coef", -10.0),
                    ("secondary_beaming_coef", -10.0),
                    ("primary_limb_dark_2", -10.0),
                    ("secondary_limb_dark_2", -10.0),
                ]
            ]
        )
        self._exclude_priors = numpy.full(len(SampleParams._fields), False)
        self._exclude_priors[[i for i, _ in self._do_not_optimize]] = True
        self._num_optimize = len(SampleParams._fields) - len(
            self._do_not_optimize
        )
        self._optimize_to_mcmc = {}
        skip = 0
        for i in range(self._num_optimize):
            while (
                skip < len(self._do_not_optimize)
                and i + skip == self._do_not_optimize[skip][0]
            ):
                skip += 1
            self._optimize_to_mcmc[i] = i + skip
        self._mcmc_to_optimize = dict(
            (v, k) for k, v in self._optimize_to_mcmc.items()
        )

    def __call__(self, initial_params):
        """Find a local maximum in log-likelihood for given parameters."""

        _logger.info(
            "Looking for starting position with: %s", repr(initial_params)
        )

        overwrite = InitialParamType(
            per=self._log_likelihood.inverse_prior(
                "per", initial_params.per[0]
            ),
            eclipse_time=(
                self._log_likelihood.inverse_prior(
                    "eclipse_time", initial_params.eclipse_time[0]
                )
            ),
            mprimary=initial_params.mprimary[0],
        )

        log_likelihood = numpy.nan
        while not numpy.isfinite(log_likelihood):
            initial_state = self._get_optimize_start(overwrite)
            log_likelihood = -self._to_optimize(initial_state)
            _logger.debug(
                "Found log-likelihood(%s) = %s",
                repr(initial_state),
                repr(log_likelihood),
            )
        result = optimize.minimize(
            self._to_optimize,
            x0=initial_state,
            method="Nelder-Mead",
            bounds=self._get_bounds(initial_params),
            options={"adaptive": True, "fatol": 100.0},
        )
        if not result.success:
            _logger.warning("Optimization did not converge: %s", repr(result))
        _logger.info(
            "log-likelihood(%s) = %s", repr(result.x), repr(result.fun)
        )
        return self._get_mcmc_sample(result.x)


# pylint: enable=too-few-public-methods


def _optimize_starting_positions(
    initial_params_queue, result_queue, log_likelihood, config
):
    """Find a position that maximizes log(prob) starting with given Porb."""

    try:
        numpy.random.seed()
        setup_process_map(vars(config))
        get_starting_position = OptimizeStartingPosition(log_likelihood)

        for initial_params in iter(initial_params_queue.get, "STOP"):
            result_queue.put(get_starting_position(initial_params))
        _logger.info("Starting position optimizanio process finished.")
    finally:
        _logger.critical(
            "Optimizing initial positions failed:\n%s", format_exc()
        )
        result_queue.put(None)


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
        samples_fname,
        num_walkers=config.num_walkers,
        num_params=len(SampleParams._fields),
    )

    _logger.info(
        "Found %d starting positions, looking for %d additional suitable "
        "starting positions.",
        positions_found,
        config.num_walkers - positions_found,
    )
    initial_param_queue = Queue()
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
        initial_param_queue.put(
            InitialParamType(
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
        initial_param_queue.put("STOP")

    workers = [
        Process(
            target=_optimize_starting_positions,
            args=(initial_param_queue, result_queue, log_likelihood, config),
        )
        for _ in range(config.num_parallel)
    ]
    for process in workers:
        process.start()

    for position_ind in range(positions_found, config.num_walkers):
        position = result_queue.get()
        _logger.debug(
            "Saving initial position %d: %s",
            position_ind,
            repr(position),
        )
        if position is None:
            # pylint: disable=invalid-name
            for w in workers:
                w.terminate()
            # pylint: enable=invalid-name
            raise RuntimeError("Failed to find initial walker positions.")
        starting_positions[position_ind] = position
        save_initial_position(
            position,
            samples_fname,
            nwalkers=config.num_walkers,
        )

    return starting_positions
