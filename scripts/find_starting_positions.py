"""Methods for finding initial walker positions for MCMC."""

from collections import namedtuple
from multiprocessing import Process, Queue
import logging
from traceback import format_exc
import sys

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
from utils import (
    set_logage_fraction,
    params_to_sample,
    lmfit_and_tweak,
    tweak_sample,
    InitialSample,
    get_logage_range,
)

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
        max_e_phase_diff = calc_eclipse_phase_diff(
            LogLikelihood.max_ecc, params.w
        )
        if abs(max_e_phase_diff - secondary_eclipse_phase) < 1e-8:
            _logger.debug(
                "Eclipse times require max e: e=%s for phase diff = %s vs %s "
                "(diff=%s)",
                repr(LogLikelihood.max_ecc),
                repr(max_e_phase_diff),
                repr(secondary_eclipse_phase),
                repr(max_e_phase_diff - secondary_eclipse_phase),
            )
            return params._replace(ecc=LogLikelihood.max_ecc)

        _logger.debug(
            "Matching eclipse times: e=%s phase diff = %s",
            repr(LogLikelihood.max_ecc),
            max_e_phase_diff,
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
                        set_logage_fraction(
                            params._replace(mratio=mratio),
                            logage_fraction,
                            self._log_likelihood,
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

        mass_range = self._mtotal_range
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
            set_logage_fraction(
                params._replace(mratio=min(max(result.x, min_mratio), 1.0)),
                logage_fraction,
                self._log_likelihood,
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
                    from_mcmc=set_logage_fraction(
                        lc_tuned_params._replace(mtotal=mtotal),
                        logage_fraction,
                        self._log_likelihood,
                    )
                )
            except ValueError:
                _logger.warning(
                    "Mtotal to_minimize failed for mtotal=%s, params=%s:\n%s",
                    repr(mtotal),
                    repr(lc_tuned_params),
                    format_exc(),
                )
                return numpy.inf
            return -self._log_likelihood.calc_sed_log_likelihood(binary, 0.0)

        def to_solve(mtotal, get_params=False):
            try:
                lc_tuned_params = self._match_both_depths(
                    params._replace(mtotal=mtotal), logage_fraction
                )
                result = optimize.minimize_scalar(
                    to_minimize,
                    bounds=self._mtotal_range,
                    args=(lc_tuned_params,),
                    options={"xatol": 1e-3 * max(mtotal, 1)},
                )
            except ValueError:
                _logger.warning(
                    "Mtotal equation failed for mtotal=%s, params=%s:\n%s",
                    repr(mtotal),
                    repr(params),
                    format_exc(),
                )
                if get_params:
                    raise
                return self._mtotal_range[0] - mtotal
            assert result.success, f"Failed to optimize total mass: {result!r}!"
            _logger.debug(
                "Starting from Mtotal = %s, found b = %s, m2/m1 = %s, "
                "Motal = %s. Minimization result: %s",
                mtotal,
                lc_tuned_params.primary_impact_param,
                lc_tuned_params.mratio,
                result.x,
                repr(result),
            )

            if get_params:
                return set_logage_fraction(
                    lc_tuned_params._replace(mtotal=result.x),
                    logage_fraction,
                    self._log_likelihood,
                )
            if abs(result.x - mtotal) < 1e-3 * mtotal:
                raise GoodEnough("eclipses_and_sed", mtotal)
            return result.x - mtotal

        try:
            result = optimize.root_scalar(
                to_solve,
                bracket=self._mtotal_range,
                rtol=1e-3,
            )
            _logger.debug("Mtotal root finding result: %s", repr(result))
            assert result.converged, f"Failed to solve for mtotal: {result!r}!"
            result = result.root
        except ValueError:
            _logger.warning(
                "Root finding failed for mass range %s, trying endpoints.",
                repr(self._mtotal_range),
            )
            residuals = to_solve(self._mtotal_range[0]), to_solve(
                self._mtotal_range[1]
            )
            if residuals[0] * residuals[1] < 0:
                _logger.warning(
                    "Unexpected failure for mass range %s for params:\n%s:\n%s",
                    repr(self._mtotal_range),
                    repr(params),
                    format_exc(),
                )
                raise
            _logger.debug(
                "Endpoint residuals: %s -> %s, %s -> %s",
                repr(self._mtotal_range[0]),
                repr(residuals[0]),
                repr(self._mtotal_range[1]),
                repr(residuals[1]),
            )
            result = (
                self._mtotal_range[0]
                if abs(residuals[0]) < abs(residuals[1])
                else self._mtotal_range[1]
            )
        except GoodEnough as stopped:
            assert (
                stopped.args[0] == "eclipses_and_sed"
            ), f"Unexpected good enough caller: {stopped.args[0]!r}"
            result = stopped.args[1]
        result = min(max(result, self._mtotal_range[0]), self._mtotal_range[1])
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
            max(0.01, config.initial_logage_smear / 2),
            min(0.99, 1.0 - config.initial_logage_smear / 2),
            config.initial_num_ages,
        )
        meh_values = numpy.linspace(
            max(-0.95, -1.0 + config.initial_meh_smear / 2),
            min(0.45, 0.5 - config.initial_meh_smear / 2),
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
            wmin + max(5.0, config.initial_w_smear / 2),
            wmax - max(5.0, config.initial_w_smear / 2),
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
                        "unphysical, using random position near top sample:\n"
                        "%s",
                        repr(scenario),
                        format_exc(),
                    )
                    optimized_queue.put((scenario_ind, None))
                    continue

                result = lmfit_and_tweak(
                    params_to_sample(params, self._log_likelihood, True),
                    self._log_likelihood,
                    0.01,
                )
                _logger.info(
                    "Generated optimized sample (ll=%s):\n%s\nTweaked to "
                    "(ll=%s):\n%s",
                    repr(result.lstsq_log_likelihood),
                    repr(result.lstsq_sample),
                    repr(result.tweaked_log_likelihood),
                    repr(result.tweaked_sample),
                )
                params = self._log_likelihood.get_sample_params(result[1])
                _logger.info(
                    "Above tweaked sample corresponds to parameters:\n%s",
                    params,
                )
                _logger.info(
                    "Above corresponds to binary:\n%s", Binary(from_mcmc=params)
                )

                optimized_queue.put((scenario_ind, result))
            _logger.info("Position optimization process finished.")
        # pylint: disable=bare-except
        except:
            _logger.critical(
                "Initial position worker failed:\n%s", format_exc()
            )
            optimized_queue.put(None)
        # pylint: enable=bare-except

    def _load_initial_samples(self, samples_fname, num_walkers):
        """Load saved initial positions from a previous run."""

        initial_position_data = load_initial_positions(
            samples_fname,
            num_walkers=num_walkers,
            num_params=len(SampleParams._fields),
            blobs_dtype=[("log_likelihood", float)]
            + [
                (f"s{i:02d}", float) for i, _ in enumerate(SampleParams._fields)
            ],
        )
        if len(initial_position_data) == 2:
            assert not initial_position_data[1].any()
            _logger.info("No previously saved starting positions found.")
            return initial_position_data + (None,)
        (
            starting_positions,
            starting_log_likelihood,
            lstsq_data,
            positions_found,
        ) = initial_position_data
        top_position = None
        for tweaked_sample, tweaked_log_likelihood, lstsq_result in zip(
            starting_positions, starting_log_likelihood, lstsq_data
        ):
            if (
                top_position is None
                or lstsq_result[0] > top_position.lstsq_log_likelihood
            ):
                _logger.debug(
                    "Processing loaded LSTSQ result:\n%s", repr(lstsq_result)
                )
                top_position = InitialSample(
                    numpy.array(list(lstsq_result)[1:]),
                    tweaked_sample,
                    lstsq_result["log_likelihood"],
                    tweaked_log_likelihood,
                )
        return starting_positions, positions_found, top_position

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

        self._mtotal_range = log_likelihood.get_range("mtotal")

        _logger.info("From BLS: %s", repr(self._bls_eclipses))

    def optimize(self, logage_or_mass_fraction, meh, w, randomize_e):
        """Find a local maximum in log-likelihood for given parameters."""

        params = set_logage_fraction(
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
            self._log_likelihood,
        )

        _logger.debug("Starting params: %s", params)
        params = self._match_eclipse_times(params, randomize_e)
        _logger.debug("Eclipse timing matched params: %s", params)
        while self._mtotal_range[1] - self._mtotal_range[0] > 1e-3:
            try:
                get_logage_range(
                    params._replace(mtotal=self._mtotal_range[1], mratio=1.0),
                    self._log_likelihood,
                )
                break
            except ValueError as err:
                _logger.warning(
                    "Upper mass range %s invalid for params %s: %s",
                    repr(self._mtotal_range[1]),
                    repr(
                        params._replace(
                            mtotal=self._mtotal_range[1], mratio=1.0
                        )
                    ),
                    str(err),
                )
                self._mtotal_range = (
                    self._mtotal_range[0],
                    self._mtotal_range[1]
                    - 0.1 * (self._mtotal_range[1] - self._mtotal_range[0]),
                )
        if self._mtotal_range[1] - self._mtotal_range[0] <= 1e-3:
            raise ValueError(f"No valid mass range found for params {params!r}")

        if numpy.isfinite(self._log_likelihood.sed[0]).any():
            params = self._match_eclipses_and_sed(
                params, logage_or_mass_fraction
            )
            _logger.debug("Eclipse and SED matched params: %s", params)
        else:
            _logger.debug(
                "Fitting age for mass fraction %s of range %s",
                repr(logage_or_mass_fraction),
                repr(self._mtotal_range),
            )
            params = self._fit_age(
                params._replace(
                    mtotal=self._mtotal_range[0]
                    + logage_or_mass_fraction
                    * (self._mtotal_range[1] - self._mtotal_range[0])
                )
            )
            _logger.debug("Age fit params: %s", params)
        params = self._fit_limbdark(params)
        _logger.info("Suggested starting params: %s", params)

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

        starting_positions, positions_found, top_position = (
            self._load_initial_samples(samples_fname, num_walkers)
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
        unphysical = []
        for _ in initial_scenarios:
            position = optimized_queue.get()
            if position is None:
                for w in workers:
                    w.kill()
                _logger.critical("Failed to find initial walker positions.")
                sys.exit(1)
                # pylint: enable=invalid-name
            else:
                scenario_ind, position = position
            if position is None:
                unphysical.append(scenario_ind)
            else:
                if top_position is None or (
                    position.lstsq_log_likelihood
                    > top_position.lstsq_log_likelihood
                ):
                    top_position = position
                    _logger.debug(
                        "Top position updated to\n%s",
                        repr(top_position.lstsq_sample),
                    )
                _logger.info("Saving initial position %d: %s", *position)
                save_initial_position(
                    position.tweaked_sample,
                    samples_fname,
                    nwalkers=num_walkers,
                    index=scenario_ind,
                    log_prob_result=(
                        position.tweaked_log_likelihood,
                        position.lstsq_log_likelihood,
                    )
                    + tuple(position.lstsq_sample),
                )
                starting_positions[scenario_ind] = position.tweaked_sample
        for scenario_ind in unphysical + list(
            range(-config.num_random_walkers, 0)
        ):
            starting_positions[scenario_ind] = tweak_sample(
                top_position.lstsq_sample, self._log_likelihood
            )
            if starting_positions[scenario_ind] is None:
                starting_positions[scenario_ind] = norm.rvs(size=num_params)
            _logger.info(
                "Adding random sample %d (ll=%s):\n%s\nCorresponding to "
                "params:\n%s",
                scenario_ind,
                self._log_likelihood(starting_positions[scenario_ind]),
                starting_positions[scenario_ind],
                self._log_likelihood.get_sample_params(
                    starting_positions[scenario_ind]
                ),
            )

            save_initial_position(
                starting_positions[scenario_ind],
                samples_fname,
                nwalkers=num_walkers,
                index=scenario_ind,
            )

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
