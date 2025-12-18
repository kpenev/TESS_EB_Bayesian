"""Methods for finding initial walker positions for MCMC."""

from collections import namedtuple
from multiprocessing import Process, Queue
import logging
from traceback import format_exc

import numpy
from scipy.stats import norm, uniform
from scipy import optimize
from astropy import units, constants

from general_purpose_python_modules.multiprocessing_util import setup_process
from general_purpose_python_modules.emcee_util import (
    save_initial_position,
    load_initial_positions,
)

from paths import jktebob as jktebob_paths
from sample_params import SampleParams

from log_likelihood import LogLikelihood
from binary import Binary
from binary_parameters import calc_eclipse_phase_diff

_logger = logging.getLogger(__name__)

InitialParamType = namedtuple(
    "InitialParamType", ["per", "eclipse_time", "mprimary"]
)


class GoodEnough(Exception):
    """Raised when a solver or optimized encounters a good enough value."""


class FindStartingPositions:
    """Callable to find initial position close local likelihood maxima."""

    def _classify_bls(self):
        """Detect deeper & shallower BLS eclipses & is single eclipse viable."""

        bls_info = self._log_likelihood.best_fit_bls
        if self._log_likelihood.masked_is_significant():
            if bls_info["depth"][0] > bls_info["masked_depth"][0]:
                return "both", "masked", False
            return "masked", "both", False
        single = abs(
            bls_info["depth_even"][0] - bls_info["depth_odd"][0]
        ) < 5.0 * numpy.sqrt(
            bls_info["depth_even"][1] ** 2 + bls_info["depth_odd"][1] ** 2
        )

        if bls_info["depth_even"][0] > bls_info["depth_odd"][0]:
            return "even", "odd", single
        return "odd", "even", single

    def _get_logage_range(self, params):
        """Return the valid range for log(age) for the given parameters."""

        mprimary = params.mtotal / (1.0 + params.mratio)
        primary_log_age_range = Binary.get_star_log_age_range(
            mprimary, params.meh
        )
        secondary_log_age_range = Binary.get_star_log_age_range(
            mprimary * params.mratio, params.meh
        )
        likelihood_log_age_range = self._log_likelihood.get_range("age_gyr")
        return (
            max(
                likelihood_log_age_range[0],
                primary_log_age_range[0],
                secondary_log_age_range[0],
            ),
            min(
                likelihood_log_age_range[1],
                primary_log_age_range[1],
                secondary_log_age_range[1],
            ),
        )

    def _set_age(self, params, logage_fraction):
        """Return params with age set according to ``logage_fraction``."""

        assert (
            0.0 <= logage_fraction <= 1.0
        ), f"Log(age) fraction {logage_fraction} is not in [0, 1] range!"
        min_log_age, max_log_age = self._get_logage_range(params)
        _logger.debug(
            "Setting age for params %s, log(age) fraction = %s based on range "
            "(%s, %s)",
            params,
            logage_fraction,
            repr(min_log_age),
            repr(max_log_age),
        )
        return params._replace(
            age_gyr=10.0
            ** (min_log_age + logage_fraction * (max_log_age - min_log_age))
        )

    def _get_age_fraction(self, params):
        """Return what fraction of the log(age) interval is the current age."""

        min_log_age, max_log_age = self._get_logage_range(params)
        return (numpy.log10(params.age_gyr) - min_log_age) / (
            max_log_age - min_log_age
        )

    def _get_mcmc_sample(self, params):
        """Return the MCMC sample corresponding to the given parameters."""

        return numpy.array(
            [
                self._log_likelihood.inverse_prior(param, value)[1]
                for param, value in zip(params._fields, params)
            ]
        )

    def _match_eclipse_times(self, params, randomize_e):
        """Set the eccentricity to match the eclipse phases."""

        if (
            self._bls_eclipses["shallower"] != "masked"
            and self._bls_eclipses["deeper"] != "masked"
            and not randomize_e
        ):
            if params.w % 360 > 270:
                params = params._replace(w=params.w - 360.0)
            return params._replace(ecc=0.0)

        secondary_eclipse_phase = self._secondary_eclipse_phase
        if randomize_e:
            secondary_eclipse_phase += uniform.rvs(
                loc=-0.05
                * (
                    self._log_likelihood.best_fit_bls["duration"]
                    + self._log_likelihood.best_fit_bls["period"][1]
                ),
                scale=0.1
                * (
                    self._log_likelihood.best_fit_bls["duration"]
                    + self._log_likelihood.best_fit_bls["period"][1]
                ),
            )

        _logger.debug(
            "Matching eclipse times: Secondary eclipse phase = %s",
            repr(secondary_eclipse_phase),
        )

        final_w = params.w % 360
        if (secondary_eclipse_phase < 0.5 and not 90 < final_w < 270) or (
            secondary_eclipse_phase > 0.5 and (90 < final_w < 270)
        ):
            final_w += 180.0
        final_w %= 360
        if final_w > 270:
            final_w -= 360
        params = params._replace(w=final_w)

        _logger.debug("Matching eclipse times: final w = %s", repr(final_w))

        _logger.debug(
            "Matching eclipse times: e=0 phase diff = %s",
            calc_eclipse_phase_diff(0, params.w),
        )
        _logger.debug(
            "Matching eclipse times: e=%s phase diff = %s",
            repr(LogLikelihood.max_ecc),
            calc_eclipse_phase_diff(LogLikelihood.max_ecc, params.w),
        )

        def to_solve(ecc):
            return (
                calc_eclipse_phase_diff(ecc, params.w) - secondary_eclipse_phase
            )

        result = optimize.root_scalar(
            to_solve, bracket=(0.0, LogLikelihood.max_ecc)
        )
        assert (
            result.converged
        ), f"Optimization of eclipse times failed!: {result!r}"

        return params._replace(ecc=result.root)

    def _match_deeper_eclipse_phase(self, params):
        """Set binary LC model deeper eclipse to match the BLS deeper one."""

        _logger.debug("Matching deeper eclipse params: %s", repr(params))
        binary = Binary(from_mcmc=params._replace(primary_impact_param=0.0))
        bls_info = self._log_likelihood.best_fit_bls
        if self._bls_eclipses["shallower"] == "maked":
            duration = max(bls_info["duration"], bls_info["masked_duration"])
        if self._bls_eclipses["deeper"] == "maked":
            duration = max(bls_info["masked_duration"], bls_info["duration"])
        else:
            duration = bls_info["duration"]
        faintest = None
        for inverted in [False, True]:
            time = numpy.linspace(
                binary.t0 - 2 * duration,
                binary.t0 + 2 * duration,
                201,
            )
            if inverted:
                if (
                    binary.get_lightcurve(time).min() < faintest
                    and self._bls_eclipses["deeper"] != "odd"
                ) or (
                    binary.get_lightcurve(time).min() > faintest
                    and self._bls_eclipses["deeper"] == "odd"
                ):
                    return params._replace(
                        w=(-180.0 if params.w > 90.0 else 180.0) + params.w,
                        eclipse_time=params.eclipse_time
                        + binary.eclipse_time_difference,
                    )
                return params
            faintest = binary.get_lightcurve(time).min()
            binary.swap_components()
        assert False, "Neither even nor odd eclipse flagged as deeper!"

    def _match_deeper_eclipse_depth(self, params):
        """Tune the primary impact parameter to best fit deeper eclipses."""

        def to_minimize(impact, phase_matched_params):
            _logger.debug(
                "Impact: Trying b=%s to match %s eclipses for %s",
                impact,
                self._bls_eclipses["deeper"],
                phase_matched_params,
            )
            mod_params = phase_matched_params._replace(
                primary_impact_param=impact
            )
            try:
                binary = Binary(from_mcmc=mod_params)
            except ValueError:
                _logger.warning("Failed to create binary from %s!", params)
                return numpy.inf
            return -self._log_likelihood.calc_lc_log_likelihood(
                binary, 0.0, self._bls_eclipses["deeper"]
            )

        params = self._match_deeper_eclipse_phase(params)
        temp_binary = Binary(from_mcmc=params)
        max_impact = min(temp_binary.a, 1.0 + temp_binary.rp)
        _logger.debug(
            "Optimizing impact parameter in range (%s, %s) for params:\n%s\n"
            "binary:\n%s",
            0.0,
            max_impact,
            params,
            temp_binary,
        )
        result = optimize.minimize_scalar(
            to_minimize,
            bounds=(
                0.0,
                min(
                    self._log_likelihood.get_range("primary_impact_param")[1],
                    max_impact,
                ),
            ),
            args=(params,),
            options={"disp": 3, "xatol": 1e-3},
        )
        assert (
            result.success
        ), f"Failed to optimize impact parameter: {result!r}!"
        return params._replace(primary_impact_param=min(result.x, max_impact))

    def _match_both_depths(self, params, logage_fraction):
        """Tune primary impact and mratio to best fit both eclipses."""

        def to_minimize(mratio):
            try:
                binary = Binary(
                    from_mcmc=self._match_deeper_eclipse_depth(
                        self._set_age(
                            params._replace(mratio=mratio), logage_fraction
                        )
                    )
                )
            except ValueError:
                return numpy.inf
            result = -self._log_likelihood.calc_lc_log_likelihood(
                binary, 0.0, self._bls_eclipses["shallower"]
            )
            _logger.debug(
                "For m2/m1 = %s, shallower -LL: %s", repr(mratio), repr(result)
            )
            return result

        mass_range = self._log_likelihood.get_range("mtotal")
        mass_range = (mass_range[0] / 2, mass_range[1])
        _logger.debug("Mass range: %s", repr(mass_range))
        _logger.debug("Mtot: %s", repr(params.mtotal))
        # False positive
        # pylint: disable=unsubscriptable-object
        min_mratio = max(
            params.mtotal / mass_range[1] - 1,
            mass_range[0] / (params.mtotal - mass_range[0]),
        )
        _logger.debug(
            "Matching both depths: Mass ratio range = (%s, 1)", repr(min_mratio)
        )
        # pylint: enable=unsubscriptable-object
        min_mratio = max(
            self._log_likelihood.get_range("mratio")[0],
            min_mratio,
        )
        result = optimize.minimize_scalar(
            to_minimize, bounds=(min_mratio, 1.0), options={"xatol": 1e-3}
        )
        assert result.success, f"Failed to optimize mass ratio: {result!r}!"
        _logger.debug("Matching both depths solution: %s", repr(result))
        return self._match_deeper_eclipse_depth(
            self._set_age(
                params._replace(mratio=min(max(result.x, min_mratio), 1.0)),
                logage_fraction,
            )
        )

    def _match_eclipses_and_sed(self, params, logage_fraction):
        """Tune masses and impact parameter to best fit eclipses and SED."""

        _logger.debug(
            "Matching eclipses and SED: for [M/H]=%s, log(age) fraction=%s",
            repr(params.meh),
            repr(logage_fraction),
        )

        def to_minimize(mtotal, lc_tuned_params):
            try:
                binary = Binary(
                    from_mcmc=self._set_age(
                        lc_tuned_params._replace(mtotal=mtotal), logage_fraction
                    )
                )
            except ValueError:
                return numpy.inf
            return -self._log_likelihood.calc_sed_log_likelihood(binary, 0.0)

        def to_solve(mtotal, get_params=False):
            lc_tuned_params = self._match_both_depths(
                params._replace(mtotal=mtotal), logage_fraction
            )
            result = optimize.minimize_scalar(
                to_minimize,
                bounds=self._log_likelihood.get_range("mtotal"),
                args=(lc_tuned_params,),
                options={"xatol": 1e-3 * max(mtotal, 1)},
            )
            assert result.success, f"Failed to optimize total mass: {result!r}!"
            _logger.debug(
                "Starting from Mtotal = %s, found b = %s, m2/m1 = %s, "
                "Motal = %s",
                mtotal,
                lc_tuned_params.primary_impact_param,
                lc_tuned_params.mratio,
                result.x,
            )

            if get_params:
                return self._set_age(
                    lc_tuned_params._replace(mtotal=result.x), logage_fraction
                )
            if abs(result.x - mtotal) < 1e-3 * mtotal:
                raise GoodEnough("eclipses_and_sed", mtotal)
            return result.x - mtotal

        mtotal_range = self._log_likelihood.get_range("mtotal")
        try:
            result = optimize.root_scalar(
                to_solve,
                bracket=mtotal_range,
                rtol=1e-3,
            )
            assert result.converged, f"Failed to solve for mtotal: {result!r}!"
            result = result.root
        except GoodEnough as stopped:
            assert (
                stopped.args[0] == "eclipses_and_sed"
            ), f"Unexpected good enough caller: {stopped.args[0]!r}"
            result = stopped.args[1]
        result = min(max(result, mtotal_range[0]), mtotal_range[1])
        params = to_solve(result, True)
        _logger.info("Optimized parameters: %s", params)
        return params

    def _fit_age(self, params):
        """Fit for the age of the system to maximize LC log-likelihood."""

        def to_minimize(logage_fraction):
            """Return -log-likelihood for given log(age) fraction."""

            try:
                binary = Binary(
                    from_mcmc=self._match_both_depths(params, logage_fraction)
                )
            except ValueError:
                return numpy.inf
            result = -self._log_likelihood.calc_lc_log_likelihood(
                binary,
                0.0,
                (self._bls_eclipses["deeper"], self._bls_eclipses["shallower"]),
            )
            _logger.debug(
                "For log(age) fraction = %s\nbinary=%s\n LC -LL = %s",
                repr(logage_fraction),
                binary,
                repr(result),
            )
            return result

        result = optimize.minimize_scalar(
            to_minimize, bounds=(0.0, 1.0), options={"xatol": 1e-3}
        )
        assert (
            result.success
        ), f"Failed to optimize log(age) fraction: {result!r}!"
        _logger.debug("Age optimization result: %s", repr(result))
        return self._match_both_depths(params, result.x)

    def _fit_limbdark(self, params):
        """Fit for the limb darkening coefficients."""

        def get_params(x):
            """Return the parameters with LC coef set per optimization x."""

            return (
                params._replace(primary_limb_dark_1=x[0] * x[1])
                ._replace(primary_limb_dark_2=x[0] * (1.0 - x[1]))
                ._replace(secondary_limb_dark_1=x[2] * x[3])
                ._replace(secondary_limb_dark_2=x[2] * (1.0 - x[3]))
            )

        def to_minimize(x):
            """Set the limb darkening coefficients and return -LL."""

            binary = Binary(from_mcmc=get_params(x))
            return -self._log_likelihood.calc_lc_log_likelihood(
                binary, 0.0, "both"
            )

        result = optimize.minimize(
            to_minimize,
            [0.0, 0.0, 0.0, 0.0],
            bounds=optimize.Bounds(
                lb=[0.0, 0.0, 0.0, 0.0],
                ub=[1.0, 1.0, 1.0, 1.0],
                keep_feasible=True,
            ),
        )
        if not result.success:
            _logger.warning(
                "Failed to optimize limb darkening coefficients: %s",
                repr(result),
            )
            return params
        return get_params(result.x)

    def _get_init_scenarios(self, config):
        """Return log10(age), [M/H] and w values to base initial positions on"""

        def wmax_eq(wmax):
            """The equation defining maximum w given eclipse phases."""

            return (
                calc_eclipse_phase_diff(LogLikelihood.max_ecc, wmax)
                - self._secondary_eclipse_phase
            )

        logage_fractions = numpy.linspace(
            config.initial_logage_smear / 2,
            1.0 - config.initial_logage_smear / 2,
            config.initial_num_ages,
        )
        meh_values = numpy.linspace(
            -1.0 + config.initial_meh_smear / 2,
            0.5 - config.initial_meh_smear / 2,
            config.initial_num_mehs,
        )

        if self._secondary_eclipse_phase == 0.5:
            wmin, wmax = -180.0, 135.0
        else:
            wlimit = optimize.root_scalar(
                wmax_eq,
                bracket=(
                    (0.0, 90.0)
                    if self._secondary_eclipse_phase > 0.5
                    else (90.0, 180.0)
                ),
            ).root
            if self._secondary_eclipse_phase > 0.5:
                wmin, wmax = -wlimit, wlimit
            else:
                wmin, wmax = wlimit, 360 - wlimit

        w_values = numpy.linspace(
            wmin + config.initial_w_smear / 2,
            wmax - config.initial_w_smear / 2,
            config.initial_num_ws,
        )

        _logger.debug(
            "Initial grid from:\nlog(t) index=%s\n[M/H]=%s\nw=%s",
            repr(logage_fractions),
            repr(meh_values),
            repr(w_values),
        )
        grid = [
            arr.flatten()
            for arr in numpy.meshgrid(logage_fractions, meh_values, w_values)
        ]

        result = numpy.empty(
            shape=config.initial_num_ages
            * config.initial_num_mehs
            * config.initial_num_ws,
            dtype=[("logage_fraction", float), ("meh", float), ("w", float)],
        )
        result["logage_fraction"] = grid[0] + uniform.rvs(
            loc=-config.initial_logage_smear / 2,
            scale=config.initial_logage_smear,
            size=grid[0].size,
        )

        result["meh"] = grid[1] + uniform.rvs(
            loc=-config.initial_meh_smear / 2,
            scale=config.initial_meh_smear,
            size=grid[1].size,
        )
        result["w"] = grid[2] + uniform.rvs(
            loc=-config.initial_w_smear / 2,
            scale=config.initial_w_smear,
            size=grid[2].size,
        )
        assert not numpy.logical_or(
            result["w"] < wmin, result["w"] > wmax
        ).any(), (
            f"Found proposed arguments of periapsis outside {wmin!r} < w "
            f"< {wmax!r}: {result['w']!r}"
        )

        _logger.debug(
            "Initial scenarios:\n\t%s", "\n\t".join([str(e) for e in result])
        )
        return result

    def _params_to_sample(self, params):
        """Initialize non-optimized parameters and return MCMC sample."""

        mcmc_sample = self._get_mcmc_sample(params)
        non_finite = numpy.logical_not(numpy.isfinite(mcmc_sample))
        tiny = numpy.logical_and(non_finite, mcmc_sample < 0)
        tiny[SampleParams._fields.index("lc_sys")] = True
        tiny[SampleParams._fields.index("sed_sys")] = True

        huge = numpy.logical_and(non_finite, mcmc_sample > 0)

        mcmc_sample[tiny] = norm.ppf(uniform.rvs(size=tiny.sum(), scale=0.2))
        mcmc_sample[huge] = norm.ppf(
            uniform.rvs(size=huge.sum(), loc=0.8, scale=0.2)
        )
        if (
            non_finite[SampleParams._fields.index("mtotal")]
            or non_finite[SampleParams._fields.index("mratio")]
        ):
            tweaked_params = self._log_likelihood.get_sample_params(mcmc_sample)
            tweaked_params = self._set_age(
                tweaked_params, self._get_age_fraction(params)
            )
            mcmc_sample = self._get_mcmc_sample(tweaked_params)
            assert numpy.isfinite(mcmc_sample).all(), (
                "Even after fixing params, non-finite sample entries found: "
                f"{mcmc_sample!r}"
            )

        _logger.info(
            "Generated optimized sample:\n%s",
            mcmc_sample,
        )
        params = self._log_likelihood.get_sample_params(mcmc_sample)
        _logger.info("Above sample corresponds to parameters:\n%s", params)
        _logger.info(
            "Above corresponds to binary:\n%s", Binary(from_mcmc=params)
        )
        assert numpy.isfinite(
            mcmc_sample
        ).all(), f"Non-finite sample entries found: {mcmc_sample!r}"

        return mcmc_sample

    def _find_initial_samples(self, scenario_queue, optimized_queue, config):
        """Executed in worker threads to find optimal initial positions."""

        _logger.info("Starting position optimization process.")
        try:
            numpy.random.seed()
            setup_process(task="find_starting_positions", **vars(config))
            for scenario_ind, scenario in iter(scenario_queue.get, "STOP"):
                _logger.debug("Looking for position %d", scenario_ind)
                try:
                    params = self.optimize(
                        scenario["logage_fraction"],
                        scenario["meh"],
                        scenario["w"],
                        scenario_ind > 0
                        and self._bls_eclipses["shallower"] != "masked"
                        and self._bls_eclipses["deeper"] != "masked",
                    )
                except ValueError:
                    _logger.warning(
                        "Proposed initial position scenario (%s) appears "
                        "unphysical, using random position:\n%s",
                        repr(scenario),
                        format_exc(),
                    )
                    optimized_queue.put(
                        (scenario_ind, norm.rvs(size=len(SampleParams._fields)))
                    )
                    continue
                lstsq_sample = self._params_to_sample(params)
                max_loggprob = self._log_likelihood(lstsq_sample)[0]
                _logger.info(
                    "Starting least squares optimization from probability: %s",
                    repr(max_loggprob),
                )
                try:
                    lstsq_sample = fit_least_squares(
                        self._log_likelihood, lstsq_sample
                    )
                    _logger.info(
                        "Least squares optimization result:\n%s",
                        lstsq_sample,
                    )
                    lstsq_sample = lstsq_sample.x
                    lstsq_log_likelihood = self._log_likelihood(lstsq_sample)[0]
                    _logger.info(
                        "Least squares optimized log-likelihood: %s",
                        repr(lstsq_log_likelihood),
                    )
                    if lstsq_log_likelihood <= max_loggprob:
                        _logger.warning(
                            "Least squares optimization did not improve "
                            "log-likelihood, using pre-optimization sample!"
                        )
                    else:
                        params = self._log_likelihood.get_sample_params(
                            lstsq_sample
                        )
                except AssertionError:
                    _logger.warning(
                        "Least squares optimization failed, using "
                        "pre-optimization sample."
                    )
                _logger.info("Least squares optimized parameters:\n%s", params)

                period_tweak = min(
                    self._log_likelihood.best_fit_bls["period"][1] / 4,
                    0.2
                    * self._log_likelihood.best_fit_bls["duration"]
                    * self._log_likelihood.best_fit_bls["period"][0]
                    / (
                        self._log_likelihood.time_span[1]
                        - self._log_likelihood.time_span[0]
                    ),
                )
                period_tweak = uniform.rvs(
                    loc=-period_tweak,
                    scale=2 * period_tweak,
                )
                timing_tweak = uniform.rvs(
                    loc=-0.1 * self._log_likelihood.best_fit_bls["duration"],
                    scale=0.2 * self._log_likelihood.best_fit_bls["duration"],
                ) - period_tweak * (
                    (
                        self._log_likelihood.time_span[1]
                        - self._log_likelihood.time_span[0]
                    )
                    / params.per
                    / 2
                )

                params = params._replace(
                    eclipse_time=params.eclipse_time + timing_tweak,
                    per=params.per + period_tweak,
                )

                optimized_queue.put(
                    (scenario_ind, self._params_to_sample(params))
                )
            _logger.info("Position optimization process finished.")
        # pylint: disable=bare-except
        except:
            _logger.critical(
                "Initial position worker failed:\n%s", format_exc()
            )
            optimized_queue.put(None)
        # pylint: enable=bare-except

    @property
    def secondary_eclipse_phase(self):
        """The allowed range for the argument of periapsis per eclipse times."""

        return self._secondary_eclipse_phase

    def __init__(self, log_likelihood):
        """Prepare the callable."""

        self._log_likelihood = log_likelihood
        self._bls_eclipses = dict(
            zip(("deeper", "shallower", "allow_single"), self._classify_bls())
        )
        bls_info = self._log_likelihood.best_fit_bls
        if self._bls_eclipses["shallower"] == "masked":
            self._secondary_eclipse_phase = (
                (bls_info["masked_transit_time"] - bls_info["transit_time"])
                % bls_info["period"][0]
            ) / bls_info["period"][0]
        elif self._bls_eclipses["deeper"] == "masked":
            self._secondary_eclipse_phase = (
                (bls_info["transit_time"] - bls_info["masked_transit_time"])
                % bls_info["period"][0]
            ) / bls_info["period"][0]

        else:
            self._secondary_eclipse_phase = 0.5

        _logger.info("From BLS: %s", repr(self._bls_eclipses))

    def optimize(self, logage_or_mass_fraction, meh, w, randomize_e):
        """Find a local maximum in log-likelihood for given parameters."""

        params = self._set_age(
            SampleParams(
                mtotal=2.0,
                mratio=1.0,
                age_gyr=0.0,
                meh=meh,
                per=(
                    self._log_likelihood.best_fit_bls["period"][0]
                    * (
                        1
                        if self._bls_eclipses["shallower"] == "masked"
                        or self._bls_eclipses["deeper"] == "masked"
                        else 2
                    )
                ),
                ecc=0.0,
                w=w,
                primary_impact_param=0.0,
                eclipse_time=(
                    self._log_likelihood.best_fit_bls["masked_transit_time"]
                    if self._bls_eclipses["deeper"] == "masked"
                    else self._log_likelihood.best_fit_bls["transit_time"]
                ),
            ),
            logage_or_mass_fraction,
        )

        _logger.debug("Starting params: %s", params)
        params = self._match_eclipse_times(params, randomize_e)
        _logger.debug("Eclipse timing matched params: %s", params)
        if numpy.isfinite(self._log_likelihood.sed[0]).any():
            params = self._match_eclipses_and_sed(
                params, logage_or_mass_fraction
            )
            _logger.debug("Eclipse and SED matched params: %s", params)
        else:
            mtotal_range = self._log_likelihood.get_range("mtotal")
            _logger.debug(
                "Fitting age for mass fraction %s of range %s",
                repr(logage_or_mass_fraction),
                repr(mtotal_range),
            )
            params = self._fit_age(
                params._replace(
                    mtotal=mtotal_range[0]
                    + logage_or_mass_fraction
                    * (mtotal_range[1] - mtotal_range[0])
                )
            )
            _logger.debug("Age fit params: %s", params)
        params = self._fit_limbdark(params)
        _logger.debug("Suggested starting params: %s", params)

        return params

    # Trying to address makes function less readable
    # pylint: disable=too-many-locals
    def __call__(self, config):
        """Generate the specified scenario per command line."""

        assert config.tic_id == self._log_likelihood.tic_id, (
            f"Log-likelihood TIC ID ({self._log_likelihood.tic_id}) does not "
            f"match config TIC ID ({config.tic_id})!"
        )
        initial_scenarios = self._get_init_scenarios(config)
        num_params = len(SampleParams._fields)
        num_walkers = initial_scenarios.size + config.num_random_walkers
        samples_fname = config.samples_fname_pattern.format(
            tic_id=config.tic_id
        )
        starting_positions, positions_found = load_initial_positions(
            samples_fname,
            num_walkers=num_walkers,
            num_params=num_params,
        )

        positions_needed = numpy.flatnonzero(numpy.logical_not(positions_found))
        initial_scenarios = initial_scenarios[positions_needed]

        _logger.info(
            "Need to find %d additional starting positions: %s",
            positions_needed.size,
            positions_needed,
        )
        _logger.info("Optimizing scenarios: %s", repr(initial_scenarios))

        scenario_queue = Queue()
        for task in zip(positions_needed, initial_scenarios):
            scenario_queue.put(task)

        for _ in range(config.num_parallel):
            scenario_queue.put("STOP")

        optimized_queue = Queue()

        workers = [
            Process(
                target=self._find_initial_samples,
                args=(scenario_queue, optimized_queue, config),
            )
            for _ in range(config.num_parallel)
        ]
        for process in workers:
            process.start()
        for _ in initial_scenarios:
            position = optimized_queue.get()
            if position is None:
                # pylint: disable=invalid-name
                for w in workers:
                    w.terminate()
                # pylint: enable=invalid-name
                raise RuntimeError("Failed to find initial walker positions.")
            _logger.debug("Saving initial position %d: %s", *position)
            save_initial_position(
                position[1],
                samples_fname,
                nwalkers=num_walkers,
                index=position[0],
            )
            starting_positions[position[0]] = position[1]
        if config.num_random_walkers > 0:
            starting_positions[-config.num_random_walkers :, :] = norm.rvs(
                size=config.num_random_walkers * num_params
            ).reshape(config.num_random_walkers, num_params)
        return starting_positions

    # pylint: enable=too-many-locals


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


def create_jktebob_inputs(log_likelihood):
    """Create input files for running jktebob to optimize given lightcurve."""

    fname_substitutions = {"mode": "simplefit", "tic_id": log_likelihood.tic_id}

    bls = log_likelihood.best_fit_bls
    values = {"esinw": 0.0, "rratio": 1.0, "porb": bls["period"][0]}
    for fname_type, fname_template in jktebob_paths.items():
        values[fname_type] = fname_template.format_map(fname_substitutions)
    log_likelihood.save_jktebob_lc(values["inlcfname"])

    with open(
        values.pop("template"),
        "r",
        encoding="ascii",
    ) as template, open(
        values.pop("inputfname"),
        "w",
        encoding="ascii",
    ) as outf:
        if log_likelihood.masked_is_significant():
            timing_anomaly = (
                (bls["masked_transit_time"] - bls["transit_time"])
                % bls["period"]
            ) / bls["period"]
            timing_anomaly = min(timing_anomaly, 1.0 - timing_anomaly)
            values["ecosw"] = numpy.cos(numpy.pi * timing_anomaly)
            values["iratio"] = bls["depth"][0] / bls["masked_depth"][0]
        else:
            values["porb"] *= 2
            values["ecosw"] = 0.0
            values["iratio"] = bls["depth_odd"][0] / bls["depth_even"][0]

        values["rsum"] = numpy.sin(
            numpy.pi
            * bls["duration"]
            / (
                values["porb"]
                * (1.0 - values["esinw"] ** 2 - values["ecosw"] ** 2)
            )
        )
        outf.write(template.read().format_map(values))


def fit_least_squares(log_likelihood, initial_mcmc_sample):
    """
    Find least squares MCMC sample starting from given position.

    Args:
        initial_mcmc_sample(array):    Initial guess for the MCMC sample
            values. Omit the last two entiers (corresponding to `lc_sys` and
            `sed_sys`) to keep those fixed at zero during the fit.

    Returns:
        OptimizeResult:
            The result of the optimization containing the best fit
            parameters in the `x` attribute. See
            `scipy.optimize.least_squares`.
    """

    def residuals(x, num_residuals):
        """Return array of residuals (LC and SED) for given MCMC sample."""

        residuals.num_eval += 1
        print(f"Function evaluation {residuals.num_eval}")
        x = numpy.concatenate((x, [0.0, 0.0]))
        assert x.size == len(SampleParams._fields)
        sample_params = log_likelihood.get_sample_params(x)
        try:
            binary = Binary(from_mcmc=sample_params)
        except ValueError:
            return numpy.full(num_residuals, numpy.inf)
        lc_residuals = log_likelihood.calc_lc_log_likelihood(
            binary, sample_params.lc_sys, return_residuals=True
        )
        if not numpy.isfinite(lc_residuals).all():
            return numpy.full(num_residuals, numpy.inf)
        sed_residuals = log_likelihood.calc_sed_log_likelihood(
            binary, sample_params.sed_sys, return_residuals=True
        )
        print(
            f"Contatenating {lc_residuals.size} LC and {sed_residuals.size} "
            "SED residuals"
        )
        return numpy.concatenate((lc_residuals, sed_residuals, x))

    if initial_mcmc_sample.size == len(SampleParams._fields):
        # Fix lc_sys and sed_sys to zero during least squares fit
        initial_mcmc_sample = initial_mcmc_sample[:-2]

    residuals.num_eval = 0
    initial_resdiuals = residuals(initial_mcmc_sample, 0)
    print(
        "Initial residuals evaluated at MCMC sample\n"
        f"{initial_mcmc_sample!r}:\n{initial_resdiuals!r}"
    )
    assert (
        initial_resdiuals.size > 0 and numpy.isfinite(initial_resdiuals).all()
    ), (
        "Likelihood must be defined at initial residuals for least squares "
        "fit."
    )
    return optimize.least_squares(
        residuals,
        initial_mcmc_sample,
        args=(initial_resdiuals.size,),
        method="lm",
        xtol=1e-6,
        max_nfev=300 * initial_mcmc_sample.size,
    )
