"""
Bridge sampling of the mass of each confirmed region (step 4 of TEC-142).

So far: means over a region's samples with their errors by walker batch
means (`get_batch_mean()`), and the reference density of a region, a
Gaussian mixture fitted to samples of the region, with the number of
components chosen by how well the mixture matches the posterior on samples
held out from the fit, and the bridge estimate of a region's log evidence
from its samples and draws from that density (`estimate_log_evidence()`).
The cross-fitting (which samples are fitted and which held out) and the
region rule for the draws are to come.
"""

from collections import namedtuple
import logging

import numpy
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.mixture import GaussianMixture

_logger = logging.getLogger(__name__)


class ReferenceDensity:
    """
    A Gaussian mixture as a normalized density in the chain's coordinates.

    The mixture is fitted in coordinates standardized by the fitted samples'
    mean and spread in each parameter (so the parameters' very different
    scales do not matter, e.g. for the covariance regularization), and its
    log-density is converted back with the standardization's Jacobian.
    """

    def __init__(self, mixture, mean, scale):
        """
        Wrap a mixture fitted to standardized samples.

        Args:
            mixture(GaussianMixture):    The mixture, fitted to samples
                standardized as ``(samples - mean) / scale``.

            mean(array):    The mean of each parameter used to standardize.

            scale(array):    The spread of each parameter used to
                standardize.
        """

        self.mixture = mixture
        self._mean = mean
        self._scale = scale
        self._log_jacobian = numpy.log(scale).sum()

    @property
    def num_components(self):
        """The number of mixture components."""

        return self.mixture.n_components

    def log_density(self, points):
        """Return the log-density at points (one per row)."""

        return (
            self.mixture.score_samples((points - self._mean) / self._scale)
            - self._log_jacobian
        )

    def draw(self, num_draws, rng):
        """
        Return independent draws from the density.

        Not ``self.mixture.sample()``: it reseeds from the mixture's integer
        ``random_state`` on every call (so it would repeat the same draws)
        and returns draws grouped by component. Cholesky factors suffice
        (not SVD as in numpy's ``multivariate_normal``): sklearn regularizes
        every covariance to be positive definite, and if one were not, the
        error is raised instead of drawing from a degenerate Gaussian.

        Args:
            num_draws(int):    How many draws.

            rng(numpy.random.Generator):    Chooses the components and the
                draws from them.

        Returns:
            array:    The draws, one per row, in random order.
        """

        components = rng.choice(
            self.mixture.n_components, size=num_draws, p=self.mixture.weights_
        )
        cholesky = numpy.linalg.cholesky(self.mixture.covariances_)
        standardized = self.mixture.means_[components] + numpy.einsum(
            "nij,nj->ni",
            cholesky[components],
            rng.standard_normal((num_draws, self._mean.size)),
        )
        return self._mean + self._scale * standardized


def fit_mixture(points, num_components, seed, config):
    """
    Return a Gaussian mixture fitted to samples, as a `ReferenceDensity`.

    Args:
        points(array):    The samples, one per row.

        num_components(int):    The number of mixture components.

        seed(int):    Seeds the initializations.

        config:    The command line configuration
            (``bridge_mixture_inits`` and ``bridge_mixture_max_iterations``).

    Returns:
        ReferenceDensity:    The fitted density (the best of the
            initializations).
    """

    mean = points.mean(axis=0)
    scale = points.std(axis=0)
    if (scale == 0).any():
        raise RuntimeError(
            f"Parameters {numpy.flatnonzero(scale == 0).tolist()} do not vary "
            "among the samples: no density can be fitted to them."
        )
    mixture = GaussianMixture(
        n_components=num_components,
        covariance_type="full",
        n_init=config.bridge_mixture_inits,
        max_iter=config.bridge_mixture_max_iterations,
        random_state=seed,
    ).fit((points - mean) / scale)
    if not mixture.converged_:
        raise RuntimeError(
            f"The {num_components}-component mixture did not converge in "
            f"{config.bridge_mixture_max_iterations} iterations."
        )
    return ReferenceDensity(mixture, mean, scale)


BatchMean = namedtuple(
    "BatchMean", ["mean", "variance", "effective_size", "effective_walkers"]
)


def get_batch_mean(values, walkers):
    """
    Return the mean of values over samples and its variance, by walker.

    Batch means with each walker as one batch: a walker's sum of the values
    is one measurement, and walkers are treated as independent (their
    correlation is weak, since at every step each walker is paired with a
    different, random one). Residuals S_w - n_w * mean allow walkers with
    different numbers of samples, e.g. walkers that leave and re-enter a
    region. Gaps, several stretches of one walker and the autocorrelation
    within a walker all end up inside its sum, so no autocorrelation time
    is estimated.

    Args:
        values(array):    The value at each sample along the last axis;
            leading axes are separate functions, each estimated on its own.

        walkers(array):    The walker of each sample, in any order.

    Returns:
        BatchMean:
            ``mean``, its ``variance`` and the ``effective_size`` (the
            variance of the values over the variance of their mean, at most
            the number of samples, which is also used where the values do
            not vary), each with the leading shape of ``values``; and the
            ``effective_walkers``, N^2 / sum of n_w^2 for N samples of which
            n_w are walker w's (a diagnostic of how many walkers the
            estimate rests on).
    """

    walker_index = numpy.unique(walkers, return_inverse=True)[1]
    num_walkers = walker_index.max() + 1
    if num_walkers < 2:
        raise RuntimeError(
            "Samples from a single walker: the variance of their mean cannot "
            "be estimated."
        )
    num_samples = walker_index.size
    counts = numpy.bincount(walker_index, minlength=num_walkers)
    rows = numpy.reshape(values, (-1, num_samples))
    walker_sums = numpy.stack(
        [
            numpy.bincount(walker_index, weights=row, minlength=num_walkers)
            for row in rows
        ]
    )
    mean = rows.mean(axis=1)
    residuals = walker_sums - counts * mean[:, None]
    variance = (
        num_walkers
        / (num_walkers - 1)
        * numpy.sum(residuals**2, axis=1)
        / num_samples**2
    )
    pooled = rows.var(axis=1, ddof=1)
    effective_size = numpy.full(mean.shape, float(num_samples))
    varies = pooled > 0
    effective_size[varies] = numpy.minimum(
        pooled[varies] / variance[varies], num_samples
    )
    shape = numpy.shape(values)[:-1]
    return BatchMean(
        mean.reshape(shape),
        variance.reshape(shape),
        effective_size.reshape(shape),
        num_samples**2 / numpy.sum(counts**2),
    )


def get_score_differences(log_ratios, best, walkers):
    """
    Return each candidate's held-out score minus the best one's, with errors.

    The score of a candidate density g is the variance of log q - log g over
    the held-out samples. Differences are paired (the same samples for every
    candidate), and their standard errors come from `get_batch_mean()`
    (walkers as independent units: each walker's samples are
    autocorrelated, different walkers nearly independent).

    Args:
        log_ratios(array):    log q - log g of each candidate (rows) at each
            held-out sample (columns).

        best(int):    The candidate with the lowest score.

        walkers(array):    The walker of each held-out sample. The samples
            can be in any order, and walkers can contribute different
            numbers of them because a walker can (rarely) leave/enter a region.

    Returns:
        (array, array):
            Each candidate's score minus the best one's, and its standard
            error (both 0 for the best).
    """

    contributions = (log_ratios - log_ratios.mean(axis=1, keepdims=True)) ** 2
    differences = get_batch_mean(contributions - contributions[best], walkers)
    return differences.mean, numpy.sqrt(differences.variance)


def get_max_components(num_parameters, effective_size, config):
    """
    Return the most mixture components the samples can determine.

    Args:
        num_parameters(int):    The dimension of the samples.

        effective_size(float):    Their effective sample size.

        config:    The command line configuration
            (``bridge_max_components`` and ``bridge_samples_per_parameter``).

    Returns:
        int:    ``config.bridge_max_components``, but at most
            ``effective_size`` / (``config.bridge_samples_per_parameter`` x
            the parameters of one component), and at least 1 (with a
            warning if even one Gaussian is not determined).
    """

    needed = config.bridge_samples_per_parameter * (
        num_parameters + num_parameters * (num_parameters + 1) // 2
    )
    if effective_size < needed:
        _logger.warning(
            "An effective sample size of %.0f is below the %.0f needed for "
            "one Gaussian (%g per parameter): fitting one anyway.",
            effective_size,
            needed,
            config.bridge_samples_per_parameter,
        )
        return 1
    return min(config.bridge_max_components, int(effective_size // needed))


def fit_reference_density(fit_points, held_out, effective_size, rng, config):
    """
    Return a region's reference density, its components chosen on held-out
    samples.

    Mixtures of 1 to K_max components (full covariances, the best of
    ``config.bridge_mixture_inits`` initializations each) are fitted to
    ``fit_points``. Each is scored on the held-out samples by the variance of
    log q - log g, with q the posterior the chain stored (unnormalized):
    zero exactly when g is proportional to the posterior, it sets how much
    the bridge's weights spread, and it needs no new likelihood
    evaluations. The smallest number of components whose score is within
    ``config.bridge_noise_factor`` standard errors of the best one's is used
    (see `get_score_differences()`). K_max is
    ``config.bridge_max_components``, but at most ``effective_size`` /
    (``config.bridge_samples_per_parameter`` x the parameters of one
    component), so every component's mean and covariance are determined.

    Args:
        fit_points(array):    The samples to fit, one per row.

        held_out(dict):    The held-out samples: their positions
            ("positions", one per row), the log-probabilities the chain
            stored there ("log_prob") and the walker of each ("walkers").

        effective_size(float):    The effective sample size of
            ``fit_points``.

        rng(numpy.random.Generator):    Seeds the fits.

        config:    The command line configuration (the ``bridge_*``
            options).

    Returns:
        (ReferenceDensity, [dict]):
            The chosen density, and for each candidate: its number of
            components, score, score minus the best score and that
            difference's standard error.
    """

    candidates = [
        fit_mixture(
            fit_points, num_components, int(rng.integers(2**32)), config
        )
        for num_components in range(
            1,
            get_max_components(fit_points.shape[1], effective_size, config) + 1,
        )
    ]
    log_ratios = numpy.stack(
        [
            held_out["log_prob"] - candidate.log_density(held_out["positions"])
            for candidate in candidates
        ]
    )
    scores = log_ratios.var(axis=1)
    best = int(numpy.argmin(scores))
    differences, std_errors = get_score_differences(
        log_ratios, best, held_out["walkers"]
    )
    chosen = next(
        index
        for index in range(len(candidates))
        if differences[index] <= config.bridge_noise_factor * std_errors[index]
    )
    records = [
        {
            "components": candidate.num_components,
            "score": score,
            "difference": difference,
            "std_error": std_error,
        }
        for candidate, score, difference, std_error in zip(
            candidates, scores, differences, std_errors
        )
    ]
    for record in records:
        _logger.info(
            "%d components: held-out score %.4g (best + %.3g +- %.2g)%s.",
            record["components"],
            record["score"],
            record["difference"],
            record["std_error"],
            ": chosen" if record["components"] == chosen + 1 else "",
        )
    return candidates[chosen], records


BridgeEstimate = namedtuple(
    "BridgeEstimate",
    [
        "log_evidence",
        "relative_error",
        "sample_effective_size",
        "effective_walkers",
    ],
)


def solve_bridge_equation(sample_log_ratios, draw_log_ratios, sample_weight):
    """
    Solve the optimal bridge's estimating equation for fixed weights.

    With r = q / g, s1 = ``sample_weight``, s2 = 1 - s1 and u = s2 Z, the
    optimal bridge 1 / (s1 q + s2 Z g) gives
    (s1 / s2) mean_i t(r_i) + mean_j t(r_j) = 1 with t(r) = u / (s1 r + u)
    = expit(log u - log s1 - log r), over samples i and draws j. Every t
    increases strictly with u, from 0 to 1 (always 1 for draws outside the
    region, r = 0), so the root is unique; it is found by Brent's method.

    Args:
        sample_log_ratios(array):    log q - log g at the samples (finite).

        draw_log_ratios(array):    log q - log g at the draws, -inf for draws
            outside the region; at least one finite.

        sample_weight(float):    s1, in (0, 1).

    Returns:
        (float, array, array):
            log u, t at each sample (proportional to the samples' bridge
            terms f2 = 1 / (s1 rho + s2), rho = r / Z), and 1 - t at each
            draw (proportional to the draws' terms f1 = rho / (s1 rho + s2)).
    """

    log_sample_weight = numpy.log(sample_weight)
    weight_ratio = sample_weight / (1.0 - sample_weight)

    def excess(log_u):
        """Return the estimating equation's left side minus its right."""

        return (
            weight_ratio
            * expit(log_u - log_sample_weight - sample_log_ratios).mean()
            + expit(log_u - log_sample_weight - draw_log_ratios).mean()
            - 1.0
        )

    finite = numpy.concatenate(
        [sample_log_ratios, draw_log_ratios[numpy.isfinite(draw_log_ratios)]]
    )
    log_u = brentq(
        excess,
        log_sample_weight + finite.min() - 50.0,
        log_sample_weight + finite.max() + 50.0,
        xtol=1e-12,
    )
    return (
        log_u,
        expit(log_u - log_sample_weight - sample_log_ratios),
        expit(log_sample_weight + draw_log_ratios - log_u),
    )


def estimate_log_evidence(
    sample_log_ratios,
    walkers,
    draw_log_ratios,
    tolerance=1e-3,
    max_iterations=50,
):
    """
    Return a region's log evidence log Z by optimal bridge sampling.

    The samples' effective size n1 sets the bridge's weights s1 = n1 / (n1 +
    n2), s2 = n2 / (n1 + n2) for n2 draws. It starts as the effective size
    of log q - log g at the samples (free of Z), and is replaced by the
    effective size of the samples' bridge terms f2 at the solution until it
    changes by less than ``tolerance`` (relative): that is the size the
    optimal bridge calls for with autocorrelated samples, and each
    iteration only re-solves the equation (`solve_bridge_equation()`).

    The relative error of Z (Fruhwirth-Schnatter 2004, with walker batch
    means in place of an autocorrelation time) is RE^2 = Var(f1) / (n2
    mean(f1)^2) + V(mean f2) / mean(f2)^2: the first term from the
    independent draws, the second from the samples (`get_batch_mean()`).
    The variance of log Z is RE^2.

    Args:
        sample_log_ratios(array):    log q - log g at the region's samples,
            with q the posterior the chain stored (unnormalized) and g the
            reference density.

        walkers(array):    The walker of each sample.

        draw_log_ratios(array):    log q - log g at independent draws from g,
            -inf for draws the region rule assigns to another region.

        tolerance(float):    The relative change of n1 at which its
            iterations stop.

        max_iterations(int):    Raise an error if n1 has not settled after
            this many iterations.

    Returns:
        BridgeEstimate:
            ``log_evidence``, its ``relative_error`` (RE), the samples'
            final effective size n1 (``sample_effective_size``), and the
            effective number of walkers they come from
            (``effective_walkers``, see `get_batch_mean()`).
    """

    if not numpy.isfinite(sample_log_ratios).all():
        raise RuntimeError(
            "log q - log g is not finite at every sample of the region."
        )
    if not numpy.isfinite(draw_log_ratios).any():
        raise RuntimeError(
            "No draw from the reference density lands in its region: the "
            "region's evidence cannot be estimated."
        )
    # Keeps log u near 0 (log q is ~2e5), so Brent's tolerance is absolute.
    shift = numpy.median(sample_log_ratios)
    samples = sample_log_ratios - shift
    draws = draw_log_ratios - shift
    sample_size = float(get_batch_mean(samples, walkers).effective_size)
    for _ in range(max_iterations):
        sample_weight = sample_size / (sample_size + draws.size)
        log_u, sample_terms, draw_terms = solve_bridge_equation(
            samples, draws, sample_weight
        )
        sample_mean = get_batch_mean(sample_terms, walkers)
        if (
            abs(sample_mean.effective_size - sample_size)
            <= tolerance * sample_size
        ):
            break
        sample_size = float(sample_mean.effective_size)
    else:
        raise RuntimeError(
            f"The samples' effective size did not settle in {max_iterations} "
            f"iterations (last {sample_size:.1f})."
        )
    return BridgeEstimate(
        log_u + shift - numpy.log(1.0 - sample_weight),
        numpy.sqrt(
            draw_terms.var(ddof=1) / (draws.size * draw_terms.mean() ** 2)
            + sample_mean.variance / sample_mean.mean**2
        ),
        sample_size,
        sample_mean.effective_walkers,
    )
