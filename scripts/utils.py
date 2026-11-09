"""Collection of utility functions useful for MCMC init and sampling."""

from multiprocessing import Pool
import logging
from collections import namedtuple
from traceback import format_exc
import time

import numpy
from scipy import optimize
from scipy.stats import uniform, norm

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)

from sample_params import SampleParams
from binary import Binary

_logger = logging.getLogger(__name__)


class _StopFit(Exception):
    """Raised from inside the residual function to end a fit early."""


def get_lstsq_residuals(log_likelihood, sample, grid_excess_scale=1e-6):
    """
    Return the least squares residuals for a latent sample.

    The residuals are those of the light curve and the SED, the latent sample
    itself (its prior is unit normal), the residual making the age prior
    uniform in age (`LogLikelihood.get_age_prior_residual()`), and for each
    star how far outside the
    isochrones its mass is [Msun] divided by ``grid_excess_scale``. The last
    two are zero inside the isochrones, where the sum of squares is -2
    log-probability up to a constant. Outside them the masses are clipped to
    the isochrones (see `BinaryParams`).

    Returns:
        array or None:
            The residuals, or None if the parameters are unphysical or the
            light curve residuals are not finite.
    """

    sample_params = log_likelihood.get_sample_params(sample)
    try:
        binary = Binary(from_mcmc=sample_params, clip_to_grid=True)
    except ValueError:
        _logger.warning(
            "Least squares encountered unphysical parameters:\n%s\n%s",
            repr(sample_params),
            format_exc(),
        )
        return None
    lc_residuals = log_likelihood.calc_lc_log_likelihood(
        binary, sample_params.lc_sys, return_residuals=True
    )
    if not numpy.isfinite(lc_residuals).all():
        return None
    sed_residuals = log_likelihood.calc_sed_log_likelihood(
        binary, sample_params.sed_sys, return_residuals=True
    )
    _logger.debug(
        "Concatenating %d LC, %d SED residuals, and %d prior residuals:\n"
        "%s\n%s\n%s",
        lc_residuals.size,
        sed_residuals.size,
        sample.size,
        repr(lc_residuals),
        repr(sed_residuals),
        repr(sample),
    )
    grid_excess = numpy.array(list(binary.grid_excess.values()))
    return numpy.concatenate(
        (
            lc_residuals,
            sed_residuals,
            sample,
            [log_likelihood.get_age_prior_residual(sample)],
            grid_excess / grid_excess_scale,
        )
    )


def default_give_up(
    state, *, time_limit=None, stall_evaluations=None, stall_tolerance=1e-6
):
    """
    Return why a least squares fit should stop early, or None to continue.

    Args:
        state:    The state of the fit (the residual function of
            `fit_least_squares()`), with attributes ``num_eval`` (residual
            evaluations so far, including those for the numerical Jacobian),
            ``start`` (`time.monotonic()` when the fit started), and
            ``best_cost``, ``best_x`` and ``best_fun`` (the lowest cost
            evaluated inside the isochrones, its sample and its residuals).

        time_limit(float or None):    Stop after this many seconds.

        stall_evaluations(int or None):    Stop if the lowest cost has not
            dropped by a fraction of at least ``stall_tolerance`` within this
            many residual evaluations.

        stall_tolerance(float):    See ``stall_evaluations``.
    """

    if time_limit is not None and time.monotonic() - state.start > time_limit:
        return f"Time limit of {time_limit} s reached."
    if stall_evaluations is None:
        return None
    # The lowest cost and the evaluation number at the last drop by more than
    # the tolerance, kept with the state of this fit.
    if not hasattr(state, "stall_reference") or state.best_cost < (
        state.stall_reference[0] * (1.0 - stall_tolerance)
    ):
        state.stall_reference = (state.best_cost, state.num_eval)
        return None
    if state.num_eval - state.stall_reference[1] > stall_evaluations:
        return (
            f"Cost did not improve by a fraction of {stall_tolerance} in "
            f"{stall_evaluations} residual evaluations."
        )
    return None


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def fit_least_squares(
    log_likelihood,
    initial_mcmc_sample,
    fit_sys_err=True,
    fixed=None,
    progress_interval=300.0,
    *,
    grid_excess_scale=1e-6,
    give_up=(default_give_up,),
    **fit_kwargs,
):
    """
    Find least squares MCMC sample starting from given position.

    Outside the isochrones the model is evaluated with the masses clipped to
    them (see `BinaryParams`), and one residual per star, how far outside its
    mass is [Msun] divided by ``grid_excess_scale``, pushes the fit back in. This
    keeps the cost finite and continuous there, instead of infinite, which
    would stop the fit at the edge in all parameters.

    Args:
        initial_mcmc_sample(array):    Initial guess for the MCMC sample
            values.

        fit_sys_err(bool):    If False, the last two entries (corresponding to
            `lc_sys` and `sed_sys`) are fixed at zero during the fit.

        fixed(array of bool or None):    Entries of the sample to hold at
            their initial values during the fit.

        progress_interval(float):    Log the number of residual evaluations
            (including those for the numerical Jacobian) and the lowest cost
            so far at INFO level at most this often [s].

        grid_excess_scale(float):    The distance of the masses outside the
            isochrones [Msun] that adds 1/2 to the cost.

        give_up(sequence of callables):    Checks of whether to stop the fit
            early, each called after every residual evaluation with the state
            of the fit (see `default_give_up()`); each returns None to
            continue or the reason to stop. The default never stops; use e.g.
            ``functools.partial(default_give_up, time_limit=3600)`` for limits.

    Returns:
        OptimizeResult:
            The result of the optimization; see
            `scipy.optimize.least_squares`. Its `x` attribute is always the
            full sample, including the fixed entries, while `jac` covers only
            the fitted entries. If the fit is stopped by ``give_up``, or ends
            outside the isochrones, `x`, `cost`
            and `fun` are those of the lowest cost evaluated inside them,
            `jac` is None and `message` gives the reason; `status` is -10 for
            the first two.
    """

    num_params = len(SampleParams._fields)
    assert initial_mcmc_sample.size == num_params
    template = numpy.array(initial_mcmc_sample, dtype=float)
    free = (
        numpy.ones(num_params, dtype=bool)
        if fixed is None
        else numpy.logical_not(fixed)
    )
    if not fit_sys_err:
        # Fix lc_sys and sed_sys to zero during least squares fit if instructed
        template[-2:] = 0.0
        free[-2:] = False

    def residuals(free_values, num_residuals):
        """Return array of residuals (LC and SED) for given MCMC sample."""

        residuals.num_eval += 1
        _logger.debug("LSTSQ Function evaluation %d", residuals.num_eval)
        x = template.copy()
        x[free] = free_values
        result = get_lstsq_residuals(log_likelihood, x, grid_excess_scale)
        if result is None:
            return numpy.full(num_residuals, numpy.inf)
        cost = 0.5 * numpy.dot(result, result)
        # Outside the grid the model is not valid, only a way back in.
        in_grid = not result[-2:].any()
        if in_grid and cost < residuals.best_cost:
            residuals.best_cost = cost
            residuals.best_x = x
            residuals.best_fun = result
        now = time.monotonic()
        if now - residuals.last_report >= progress_interval:
            _logger.info(
                "Least squares progress: %d residual evaluations, lowest cost"
                " %s.",
                residuals.num_eval,
                repr(residuals.best_cost),
            )
            residuals.last_report = now
        for check in give_up:
            reason = check(residuals)
            if reason is not None:
                raise _StopFit(reason)
        return result

    residuals.num_eval = 0
    residuals.best_cost = numpy.inf
    residuals.best_x = template.copy()
    residuals.best_fun = None
    residuals.start = residuals.last_report = time.monotonic()
    initial_resdiuals = residuals(template[free], 0)
    _logger.debug(
        "Initial residuals evaluated at MCMC sample\n%s:\n%s\nsize: %d, "
        "num non finite: %d (ind: %s)",
        repr(template),
        repr(initial_resdiuals),
        initial_resdiuals.size,
        numpy.logical_not(numpy.isfinite(initial_resdiuals)).sum(),
        numpy.nonzero(numpy.logical_not(numpy.isfinite(initial_resdiuals))),
    )
    assert (
        initial_resdiuals.size > 0 and numpy.isfinite(initial_resdiuals).all()
    ), (
        "Likelihood must be defined at initial residuals for least squares "
        "fit."
    )
    # Even xtol=1e-8 stopped fits while the gradient was still large, so the
    # step-size test is effectively disabled ("lm" requires xtol >= machine
    # epsilon; 10 eps keeps it just above the precision of the parameters).
    # Fits end on ftol, gtol, max_nfev or the time and stall limits. Jacobian
    # scaling is the scipy >= 1.16 default for "lm" but not before, so it is
    # set explicitly for consistent behavior across installations.
    for arg, default in [
        ("method", "lm"),
        ("xtol", 10.0 * numpy.finfo(float).eps),
        ("x_scale", "jac"),
        ("max_nfev", 100 * free.sum()),
    ]:
        if arg not in fit_kwargs:
            fit_kwargs[arg] = default
    try:
        result = optimize.least_squares(
            residuals,
            template[free],
            args=(initial_resdiuals.size,),
            **fit_kwargs,
        )
    except _StopFit as stop:
        _logger.info(
            "Least squares stopped early after %d residual evaluations: %s",
            residuals.num_eval,
            stop,
        )
        return optimize.OptimizeResult(
            x=residuals.best_x,
            cost=residuals.best_cost,
            fun=residuals.best_fun,
            jac=None,
            optimality=numpy.nan,
            nfev=residuals.num_eval,
            status=-10,
            success=False,
            message=str(stop),
        )
    full_x = template.copy()
    full_x[free] = result.x
    result.x = full_x
    if result.fun[-2:].any():
        # Ended just outside the isochrones (pulled out by the rest of the
        # cost against the finite grid excess residuals).
        _logger.info(
            "Least squares ended %s Msun outside the isochrones; returning "
            "the lowest cost evaluated inside.",
            repr(result.fun[-2:] * grid_excess_scale),
        )
        result.x = residuals.best_x
        result.cost = residuals.best_cost
        result.fun = residuals.best_fun
        result.jac = None
        result.message += " Ended outside the isochrones."
    return result


# pylint: enable=too-many-arguments
# pylint: enable=too-many-locals


def tweak_params(
    params,
    log_likelihood,
    abs_tweak_scale=SampleParams(
        mtotal=3e-4,
        mratio=1e-6,
        age_gyr=1e-5,
        meh=0.01,
        per=0.0,
        ecc=3e-5,
        w=0.1,
        primary_impact_param=1e-3,
        eclipse_time=0.0,
        primary_limb_dark_1=1e-6,
        primary_limb_dark_2=1e-6,
        secondary_limb_dark_1=1e-6,
        secondary_limb_dark_2=1e-6,
        primary_prot=0.1,
        secondary_prot=0.1,
        primary_reflection_coef=1e-8,
        secondary_reflection_coef=1e-8,
        primary_beaming_coef=1e-8,
        secondary_beaming_coef=1e-8,
        lc_sys=1e-6,
        sed_sys=1e-6,
    ),
    rel_tweak_scale=SampleParams(
        mtotal=0.0,
        mratio=0.01,
        age_gyr=0.05,
        meh=0.0,
        per=0.0,
        ecc=0.0,
        w=0.0,
        primary_impact_param=0.0,
        eclipse_time=0.0,
        primary_limb_dark_1=0.01,
        primary_limb_dark_2=0.01,
        secondary_limb_dark_1=0.01,
        secondary_limb_dark_2=0.01,
        primary_prot=0.0,
        secondary_prot=0.0,
        primary_reflection_coef=0.01,
        secondary_reflection_coef=0.01,
        primary_beaming_coef=0.01,
        secondary_beaming_coef=0.01,
        lc_sys=0.1,
        sed_sys=0.1,
    ),
):
    """Slightly tweak the given parameters to allow MCMC sampling near them."""

    period_tweak = min(
        log_likelihood.best_fit_bls["period"][1] / 4,
        0.2
        * log_likelihood.best_fit_bls["duration"]
        * log_likelihood.best_fit_bls["period"][0]
        / (log_likelihood.time_span[1] - log_likelihood.time_span[0]),
    )
    tweak_scale = SampleParams(
        *(
            abs_tweak + rel_tweak * abs(orig)
            for abs_tweak, rel_tweak, orig in zip(
                abs_tweak_scale, rel_tweak_scale, params
            )
        )
    )
    tweak_scale = tweak_scale._replace(
        per=period_tweak,
        eclipse_time=0.1 * log_likelihood.best_fit_bls["duration"],
    )
    _logger.info(
        "Using tweak scale:\n%s\naround%s", repr(tweak_scale), repr(params)
    )
    result = SampleParams(
        *(
            orig + uniform.rvs(loc=-scale, scale=2 * scale)
            for orig, scale in zip(params, tweak_scale)
        )
    )
    # pylint: disable=unsubscriptable-object
    if result.meh < Binary.meh_range[0]:
        result = result._replace(
            meh=Binary.meh_range[0] + uniform.rvs(tweak_scale.meh / 2)
        )
    elif result.meh > Binary.meh_range[1]:
        result = result._replace(
            meh=Binary.meh_range[1] - uniform.rvs(tweak_scale.meh / 2)
        )
    # pylint: enable=unsubscriptable-object

    logage_range = get_logage_range(result, log_likelihood)
    if result.age_gyr < 10.0 ** logage_range[0]:
        result = result._replace(
            age_gyr=10.0 ** logage_range[0] + uniform.rvs(tweak_scale.age_gyr)
        )
    elif result.age_gyr > 10.0 ** logage_range[1]:
        result = result._replace(
            age_gyr=10.0 ** logage_range[1] - uniform.rvs(tweak_scale.age_gyr)
        )

    result = result._replace(
        eclipse_time=(
            result.eclipse_time
            - (result.per - params.per)
            * (
                (log_likelihood.time_span[1] - log_likelihood.time_span[0])
                / params.per
                / 2
            )
        )
    )

    return result


def find_log_age_bound(params, bad_bound, bound_limit):
    """
    Find the smallest(largest) log(age) that makes a binary with a > 1+r2/r1.

    Args:
        params(SampleParams):   The binary parameters for everything other than
            age.

        bad_bound(float):   The bad bound value for log(age).

        bound_limit(float):    The limit for the bound search (i.e., the other
            end of the log(age) range).
    """

    def to_solve(log_age):
        """Return a - (1 + r2/r1) for the given log(age)."""

        binary = Binary(from_mcmc=params._replace(age_gyr=10.0**log_age))
        return binary.a - (1 + binary.rp)

    for try_log_age in numpy.linspace(bad_bound, bound_limit, 32)[1:-1]:
        try:
            binary = Binary(
                from_mcmc=params._replace(age_gyr=10.0**try_log_age)
            )
            if binary.a >= 1 + binary.rp:
                root = optimize.brentq(
                    to_solve,
                    bad_bound,
                    try_log_age,
                    xtol=1e-5,
                )
                while to_solve(root) < 0:
                    root += (1 if bad_bound < root else -1) * 1e-5
                return root
        except ValueError:
            _logger.warning(
                "Attempt to find log(age) bound for params %s with log(age) = "
                "%s failed:\n%s",
                params,
                try_log_age,
                format_exc(),
            )
            pass
        bad_bound = try_log_age
    raise ValueError(
        "Could not find valid log(age) bound for params: "
        f"{params}, bad_bound: {bad_bound}, bound_limit: {bound_limit}"
    )


def get_logage_range(params, log_likelihood):
    """Return the valid range for log(age) for the given parameters."""

    mprimary = params.mtotal / (1.0 + params.mratio)
    primary_log_age_range = Binary.get_star_log_age_range(mprimary, params.meh)
    secondary_log_age_range = Binary.get_star_log_age_range(
        mprimary * params.mratio, params.meh
    )
    likelihood_log_age_range = log_likelihood.get_range("age_gyr")
    result = (
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
    try:
        binary = Binary(from_mcmc=params._replace(age_gyr=10.0 ** result[0]))
        bad_bound = binary.a < 1 + binary.rp
    except ValueError:
        bad_bound = True
    if bad_bound:
        result = find_log_age_bound(params, *result), result[1]
    try:
        binary = Binary(from_mcmc=params._replace(age_gyr=10.0 ** result[1]))
        bad_bound = binary.a < 1 + binary.rp
    except ValueError:
        bad_bound = True
    if bad_bound:
        result = result[0], find_log_age_bound(params, result[1], result[0])

    return result


def get_age_fraction(params, log_likelihood):
    """Return what fraction of the log(age) interval is the current age."""

    min_log_age, max_log_age = get_logage_range(params, log_likelihood)
    return (numpy.log10(params.age_gyr) - min_log_age) / (
        max_log_age - min_log_age
    )


def set_logage_fraction(params, logage_fraction, log_likelihood):
    """Return params with age set according to ``logage_fraction``."""

    assert (
        0.0 <= logage_fraction <= 1.0
    ), f"Log(age) fraction {logage_fraction} is not in [0, 1] range!"
    min_log_age, max_log_age = get_logage_range(params, log_likelihood)
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


def params_to_sample(params, log_likelihood, reset_sys_err=False):
    """Initialize non-optimized parameters and return MCMC sample."""

    def get_mcmc_sample(params):
        """Return the MCMC sample corresponding to the given parameters."""

        params = log_likelihood.orbit_to_latent(
            log_likelihood.limb_darkening_to_q(params)
        )
        return numpy.array(
            [
                log_likelihood.inverse_prior(param, value)[1]
                for param, value in zip(params._fields, params)
            ]
        )

    _logger.debug("Converting params to MCMC sample: %s", repr(params))
    if params.ecc >= log_likelihood.max_ecc:
        # Infinite latents, so placed in the top 20% of the range, as the
        # other parameters are below.
        params = params._replace(
            ecc=log_likelihood.max_ecc * uniform.rvs(loc=0.8, scale=0.2)
        )
    mcmc_sample = get_mcmc_sample(params)
    non_finite = numpy.logical_not(numpy.isfinite(mcmc_sample))
    tiny = numpy.logical_and(non_finite, mcmc_sample < 0)
    if reset_sys_err:
        tiny[SampleParams._fields.index("lc_sys")] = True
        tiny[SampleParams._fields.index("sed_sys")] = True

    huge = numpy.logical_and(non_finite, mcmc_sample > 0)
    _logger.debug("Replacing tiny: %s\nhuge: %s", tiny, huge)

    mcmc_sample[tiny] = norm.ppf(uniform.rvs(size=tiny.sum(), scale=0.05))
    mcmc_sample[huge] = norm.ppf(
        uniform.rvs(size=huge.sum(), loc=0.8, scale=0.2)
    )
    _logger.debug("Replaced MCMC sample: %s", repr(mcmc_sample))
    if (
        non_finite[SampleParams._fields.index("mtotal")]
        or non_finite[SampleParams._fields.index("mratio")]
    ):
        repaired_params = log_likelihood.get_sample_params(mcmc_sample)
        repaired_params = set_logage_fraction(
            repaired_params,
            get_age_fraction(params, log_likelihood),
            log_likelihood,
        )
        mcmc_sample = get_mcmc_sample(repaired_params)
        assert numpy.isfinite(mcmc_sample).all(), (
            "Even after fixing params, non-finite sample entries found: "
            f"{mcmc_sample!r}"
        )

    assert numpy.isfinite(
        mcmc_sample
    ).all(), f"Non-finite sample entries found: {mcmc_sample!r}"

    return mcmc_sample


def tweak_sample(
    sample, log_likelihood, tweak_scale=None, max_tweak_attempts=1000
):
    """Slightly tweak the given MCMC sample to allow MCMC sampling near it."""

    params = log_likelihood.get_sample_params(sample)
    for _ in range(max_tweak_attempts):
        if tweak_scale is None:
            params = tweak_params(params, log_likelihood)
        else:
            params = tweak_params(params, log_likelihood, tweak_scale)

        tweaked = params_to_sample(params, log_likelihood)
        if numpy.isfinite(log_likelihood(tweaked)[0]):
            return tweaked
    return None


def line_tweak_sample(tweak_from, tweak_toward, max_fraction=0.1):
    """Slightly tweak parameters from ``tweak_from`` toward ``tweak_toward``."""

    tweak_frac = uniform.rvs(loc=0.0, scale=max_fraction)

    return tweak_from + (tweak_toward - tweak_from) * tweak_frac


def process_sample(sample, log_likelihood, callback=None):
    """Get Least squares optimized sample and resulting log-likelihood."""

    lstsq_result = fit_least_squares(log_likelihood, sample)
    result = (
        lstsq_result.x,
        log_likelihood(lstsq_result.x)[0],
    )
    if callback is not None:
        callback(result)
    return result


def lstsq_optimize_samples(samples, log_likelihood, config, callback):
    """
    Optimize a list of MCMC samples using least squares fitting.

    Args:
        samples(2-D array):   MCMC samples to optimize.

        log_likelihood(LogLikelihood): LogLikelihood object used to compute
            residuals.

        config: The command line configuration for the MCMC (mostly used for I/O
            redirect).

    Returns:
        tuple of two numpy arrays::
            The first array contains the optimized samples, and the second
            contains the corresponding log-likelihoods.
    """

    with Pool(
        config.num_parallel,
        initializer=setup_process_map,
        initargs=[vars(config)],
        maxtasksperchild=1,
    ) as pool:
        result = pool.starmap(
            process_sample,
            [(s, log_likelihood, callback) for s in samples],
            chunksize=1,
        )
    return tuple(numpy.array(list(e[i] for e in result)) for i in range(2))


InitialSample = namedtuple(
    "InitialSample",
    [
        "lstsq_sample",
        "tweaked_sample",
        "lstsq_log_likelihood",
        "tweaked_log_likelihood",
    ],
)


def lmfit_and_tweak(
    input_sample, log_likelihood, max_tweak_fraction=0.1, max_tweak_attemps=100
):
    """Perform least squares fit and then tweak the result."""

    lstsq_result = fit_least_squares(log_likelihood, input_sample)
    _logger.debug("Least squares result: %s", repr(lstsq_result))
    lstsq_sample = lstsq_result.x
    lstsq_log_likelihood = log_likelihood(lstsq_sample)[0]
    tweaked_log_likelihood = -numpy.inf
    for _ in range(max_tweak_attemps):
        _logger.debug("Re-tweaking sample.")
        tweaked_sample = line_tweak_sample(
            lstsq_sample, input_sample, max_tweak_fraction
        )
        tweaked_log_likelihood = log_likelihood(tweaked_sample)[0]
        if numpy.isfinite(tweaked_log_likelihood):
            return InitialSample(
                lstsq_sample,
                tweaked_sample,
                lstsq_log_likelihood,
                tweaked_log_likelihood,
            )
    _logger.warning(
        "Tweaking LSTSQ sample failed, returning un-tweaked sample."
    )
    return InitialSample(
        lstsq_sample,
        lstsq_sample,
        lstsq_log_likelihood,
        lstsq_log_likelihood,
    )
