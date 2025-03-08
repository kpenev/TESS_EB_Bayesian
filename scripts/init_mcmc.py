"""Methods for finding initial walker positions for MCMC."""

from collections import namedtuple
from multiprocessing import Process, Queue
import logging
from traceback import format_exc

#from matplotlib.backends.backend_pdf import PdfPages
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
        if masked_is_significant(bls_info):
            assert bls_info["depth"][0] > bls_info["masked_depth"][0]
            return "both", "masked", False
        single = abs(
            bls_info["depth_even"][0] - bls_info["depth_odd"][0]
        ) < 5.0 * numpy.sqrt(
            bls_info["depth_even"][1] ** 2 + bls_info["depth_odd"][1] ** 2
        )

        if bls_info["depth_even"][0] > bls_info["depth_odd"][0]:
            return "even", "odd", single
        return "odd", "even", single

    def _match_eclipse_times(self, params):
        """Set the eccentricity to match the eclipse phases."""

        if self._bls_eclipses["shallower"] != "masked":
            return params._replace(ecc=0.0)

        if (
            self._secondary_eclipse_phase < 0.5
            and not 90 < params.w % 360 < 270
        ) or (
            self._secondary_eclipse_phase > 0.5 and (90 < params.w % 360 < 270)
        ):
            params = params._replace(w=(180.0 + params.w) % 360)

        def to_solve(ecc):
            return (
                calc_eclipse_phase_diff(ecc, params.w)
                - self._secondary_eclipse_phase
            )

        result = optimize.root_scalar(to_solve, bracket=(0.0, 1.0))
        assert result.converged
        return params._replace(ecc=result.root)

    def _match_deeper_eclipse_phase(self, params):
        """Set binary LC model deeper eclipse to match the BLS deeper one."""

        binary = Binary(from_mcmc=params._replace(primary_impact_param=0.0))
        bls_info = self._log_likelihood.best_fit_bls
        if self._bls_eclipses["shallower"] == "maked":
            duration = max(bls_info["duration"], bls_info["masked_duration"])
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
                        w=180.0 + params.w,
                        eclipse_time=params.eclipse_time
                        + binary.eclipse_time_difference,
                    )
                return params
            faintest = binary.get_lightcurve(time).min()
            binary.swap_components()
        assert False

    def _match_deeper_eclipse_depth(self, params):
        """Tune the primary impact parameter to best fit deeper eclipses."""

        def to_minimize(impact, phase_matched_params):
            mod_params = phase_matched_params._replace(
                primary_impact_param=impact
            )
            try:
                binary = Binary(from_mcmc=mod_params)
            except ValueError:
                return numpy.inf
            return -self._log_likelihood.calc_lc_log_likelihood(
                binary, 0.0, self._bls_eclipses["deeper"]
            )

        params = self._match_deeper_eclipse_phase(params)
        temp_binary = Binary(from_mcmc=params)
        _logger.debug(
            "Optimizing impact parameter for params:\n%s\nbinary:\n%s",
            params,
            temp_binary,
        )
        result = optimize.minimize_scalar(
            to_minimize,
            bounds=(
                0.0,
                min(
                    self._log_likelihood.get_range("primary_impact_param")[1],
                    temp_binary.a,
                ),
            ),
            args=(params,),
            options={"disp": 3, "xatol": 1e-3},
        )
        assert result.success
        return params._replace(primary_impact_param=result.x)

    def _match_both_depths(self, params):
        """Tune primary impact and mratio to best fit both eclipses."""

        def to_minimize(mratio):
            try:
                binary = Binary(
                    from_mcmc=self._match_deeper_eclipse_depth(
                        params._replace(mratio=mratio)
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

        min_mass_to_mtot = Binary.mini_range[0] / params.mtotal
        min_mratio = max(
            self._log_likelihood.get_range("mratio")[0],
            min_mass_to_mtot / (1.0 - min_mass_to_mtot),
        )
        result = optimize.minimize_scalar(
            to_minimize, bounds=(min_mratio, 1.0), options={"xatol": 1e-3}
        )
        assert result.success
        return self._match_deeper_eclipse_depth(
            params._replace(mratio=result.x)
        )

    def _match_eclipses_and_sed(self, params):
        """Tune masses and impact parameter to best fit eclipses and SED."""

        def to_minimize(mtotal, lc_tuned_params):
            try:
                binary = Binary(
                    from_mcmc=lc_tuned_params._replace(mtotal=mtotal)
                )
            except ValueError:
                return numpy.inf
            return -self._log_likelihood.calc_sed_log_likelihood(binary, 0.0)

        def to_solve(mtotal, get_params=False):
            lc_tuned_params = self._match_both_depths(
                params._replace(mtotal=mtotal)
            )
            result = optimize.minimize_scalar(
                to_minimize,
                bounds=self._log_likelihood.get_range("mtotal"),
                args=(lc_tuned_params,),
                options={"xatol": 1e-3 * max(mtotal, 1)},
            )
            assert result.success
            _logger.debug(
                "Starting from Mtotal = %s, found b = %s, m2/m1 = %s, "
                "Motal = %s",
                mtotal,
                lc_tuned_params.primary_impact_param,
                lc_tuned_params.mratio,
                result.x,
            )

            if get_params:
                return lc_tuned_params._replace(mtotal=result.x)
            if abs(result.x - mtotal) < 1e-3 * mtotal:
                raise GoodEnough("eclipses_and_sed", mtotal)
            return result.x - mtotal

        try:
            result = optimize.root_scalar(
                to_solve,
                bracket=self._log_likelihood.get_range("mtotal"),
                rtol=1e-3,
            )
            assert result.converged
            result = result.root
        except GoodEnough as stopped:
            assert stopped.args[0] == "eclipses_and_sed"
            result = stopped.args[1]
        params = to_solve(result, True)
        _logger.info("Optimized parameters: %s", params)
        return params

    def _get_init_scenarios(self, config):
        """Return log10(age), [M/H] and w values to base initial positions on"""

        def wmax_eq(wmax):
            """The equation defining maximum w given eclipse phases."""

            return (
                calc_eclipse_phase_diff(LogLikelihood.max_ecc, wmax)
                - self._secondary_eclipse_phase
            )

        log_age_values = numpy.linspace(-2.5, 1, config.initial_num_ages)
        meh_values = numpy.linspace(-1.0, 0.5, config.initial_num_mehs)

        if self._secondary_eclipse_phase == 0.5:
            w_values = numpy.linspace(-180.0, 135.0, config.initial_num_ws)
        else:
            wlimit = optimize.root_scalar(
                wmax_eq,
                bracket=(
                    (0.0, 90.0)
                    if self._secondary_eclipse_phase > 0.5
                    else (90.0, 270.0)
                ),
            ).root
            if self._secondary_eclipse_phase > 0.5:
                w_values = numpy.linspace(
                    -wlimit, wlimit, config.initial_num_ws
                )
            else:
                w_values = numpy.linspace(
                    wlimit, 360 - wlimit, config.initial_num_ws
                )
        grid = [
            arr.flatten()
            for arr in numpy.meshgrid(log_age_values, meh_values, w_values)
        ]

        result = numpy.empty(
            shape=config.initial_num_ages
            * config.initial_num_mehs
            * config.initial_num_ws,
            dtype=[("age_gyr", float), ("meh", float), ("w", float)],
        )
        result["age_gyr"] = 10.0 ** (
            grid[0]
            + uniform(
                loc=-config.initial_logage_smear / 2,
                scale=config.initial_logage_smear,
                size=config.initial_num_ages,
            )
        )
        result["meh"] = grid[1] + uniform(
            loc=-config.initial_meh_smear / 2,
            scale=config.initial_meh_smear,
            size=config.initial_num_mehs,
        )
        result["w"] = (
            grid[2]
            + uniform(
                loc=-config.initial_w_smear / 2,
                scale=config.initial_w_smear,
                size=config.initial_num_ws,
            )
        ) % 360
        return result

    def _find_initial_samples(self, scenario_queue, optimized_queue, config):
        """Executed in worker threads to find optimal initial positions."""

        try:
            numpy.random.seed()
            setup_process(task="find_starting_positions", **vars(config))
            for scenario_ind, scenario in iter(scenario_queue.get, "STOP"):
                try:
                    params = self.optimize(
                        scenario["age_gyr"], scenario["meh"], scenario["w"]
                    )
                except ValueError:
                    _logger.warning(
                        "Propesd initial position scenario (%s) appears "
                        "unphysical, using random position:\n%s",
                        repr(scenario),
                        format_exc(),
                    )
                    optimized_queue.put(
                        scenario_ind, norm.rvs(size=len(SampleParams._fields))
                    )
                    continue
                optimized_queue.put(
                    (
                        scenario_ind,
                        numpy.array(
                            [
                                self._log_likelihood.inverse_prior(
                                    param, value
                                )[0]
                                for param, value in zip(params._fields, params)
                            ]
                        ),
                    )
                )
            _logger.info("Starting position optimizanio process finished.")
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
        else:
            self._secondary_eclipse_phase = 0.5

        _logger.info("From BLS: %s", repr(self._bls_eclipses))

    def optimize(self, age_gyr, meh, w):
        """Find a local maximum in log-likelihood for given parameters."""

        params = SampleParams(
            mtotal=2.0,
            mratio=1.0,
            age_gyr=age_gyr,
            meh=meh,
            per=(
                self._log_likelihood.best_fit_bls["period"][0]
                * (1 if self._bls_eclipses["shallower"] == "masked" else 2)
            ),
            ecc=0.0,
            w=w,
            primary_impact_param=0.0,
            eclipse_time=self._log_likelihood.best_fit_bls["transit_time"],
        )

        _logger.debug("Starting params: %s", params)
        params = self._match_eclipse_times(params)
        _logger.debug("Eclipse timing matched params: %s", params)
        params = self._match_eclipses_and_sed(params)
        _logger.debug("Suggested starting params: %s", params)

        return params

    #Trying to address makes function less readable
    #pylint: disable=too-many-locals
    def __call__(self, config):
        """Generate the specified scenario per command line."""

        assert config.tic_id == self._log_likelihood.tic_id
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

        positions_needed = numpy.flatnonzero(numpy.logical_and(positions_found))
        initial_scenarios = initial_scenarios[positions_needed]

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
        starting_positions[-config.num_random_walkers :, :] = norm.rvs(
            size=config.num_random_walkers * num_params
        ).reshape(config.num_random_walkers, num_params)
        return starting_positions
    #pylint: enable=too-many-locals

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


def masked_is_significant(bls):
    """
    Return True iff the masked BLS fit appears to fit real eclipses.

    To be marked as significant all of the following must be satisfied:

      * maked period should be close to unmasked period (within 5 unmasked
        uncertanties)

      * depth should exceed its uncertanity by at least a factor of 5

      * masked_harmonic_delta_log_likelihood < -5
    """

    return (
        abs(bls["masked_period"][0] - bls["period"][0]) < 5.0 * bls["period"][1]
        and bls["masked_depth"][0] > 5.0 * bls["masked_depth"][1]
        and bls["masked_harmonic_delta_log_likelihood"] < -5.0
    )


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
        if masked_is_significant(bls):
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


def test():
    """Place to implement various manual tests."""

    test_tic = 16805617  # 189639080 # 4629065  #

    log_likelihood = LogLikelihood(test_tic)
    FindStartingPositions(log_likelihood)
    #params = SampleParams(
    #    mtotal=3.1180989692599335,
    #    mratio=0.9797246475335591,
    #    age_gyr=1.0,
    #    meh=0.0,
    #    per=11.793843600300352,
    #    ecc=0.0,
    #    w=180.0,
    #    primary_impact_param=0.7279367331829606,
    #    eclipse_time=1420.3748436711664,
    #    primary_limb_dark_1=0.0,
    #    primary_limb_dark_2=0.0,
    #    secondary_limb_dark_1=0.0,
    #    secondary_limb_dark_2=0.0,
    #    primary_prot=100.0,
    #    secondary_prot=100.0,
    #    primary_reflection_coef=0.01,
    #    secondary_reflection_coef=0.01,
    #    primary_beaming_coef=0.01,
    #    secondary_beaming_coef=0.01,
    #    lc_sys=1e-10,
    #    sed_sys=1e-10,
    #)
    #with PdfPages(f"tess{test_tic}_deeper_depth_match.pdf") as pdf:
    #    for impact in numpy.linspace(0.5, 1.0, 10):
    #        binary = Binary(
    #            from_mcmc=params._replace(primary_impact_param=impact)
    #        )
    #        binary.lc_sys = 0.0
    #        log_likelihood.plot_lc_model_comparison(
    #            binary,
    #            pdf,
    #            f"b: {impact}, LL: "
    #            + repr(
    #                log_likelihood.calc_lc_log_likelihood(binary, 0.0, "odd")
    #            ),
    #        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test()
