"""
Bridge sampling of the mass of each confirmed region (step 4 of TEC-142).

So far: the reference density of a region, a Gaussian mixture fitted to
samples of the region, with the number of components chosen by how well the
mixture matches the posterior on samples held out from the fit. The
cross-fitting (which samples are fitted and which held out) and the bridge
estimate itself are to come.
"""

import logging

import numpy
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


def get_score_differences(log_ratios, best, walkers):
    """
    Return each candidate's held-out score minus the best one's, with errors.

    The score of a candidate density g is the variance of log q - log g over
    the held-out samples. Differences are paired (the same samples for every
    candidate), and their standard errors treat walkers as independent
    units (each walker's samples are autocorrelated, different walkers
    nearly independent).

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
    differences = contributions - contributions[best]
    walker_index = numpy.unique(walkers, return_inverse=True)[1]
    num_walkers = walker_index.max() + 1
    if num_walkers < 2:
        raise RuntimeError(
            "Held-out samples from a single walker: the score's error cannot "
            "be estimated."
        )
    counts = numpy.bincount(walker_index, minlength=num_walkers)
    walker_sums = numpy.stack(
        [
            numpy.bincount(walker_index, weights=row, minlength=num_walkers)
            for row in differences
        ]
    )
    mean_differences = differences.mean(axis=1)
    residuals = walker_sums - counts * mean_differences[:, None]
    return mean_differences, numpy.sqrt(
        num_walkers / (num_walkers - 1) * numpy.sum(residuals**2, axis=1)
    ) / len(walkers)


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
