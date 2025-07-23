"""Define log-likeihoods that use only priors for testing MCMC sampling."""

import numpy
from scipy.stats import truncnorm

from log_likelihood import LogLikelihood
from sample_params import SampleParams

class LogLikelihoodPriorsOnly(LogLikelihood):
    """Allows MCMC sampling using just the priors for testing."""

    def __call__(self, mcmc_sample, exclude_priors=False):
        """Return the log-likelihood of the given MCMC sample."""

        return (
            self.calc_prior_loglikelihood(
                mcmc_sample[numpy.logical_not(exclude_priors)]
            ),
        ) + self.get_sample_params(mcmc_sample)


class LogLikelihoodUnitCubePriors(LogLikelihood):
    """Overwrite the prior transform to be from U(0,1) instead of normal."""

    def prior_transform(self, param, sample_entry):
        """
        Return the value of the given parameter given a sample entry.

        Apply a transformation to go from identical random variables with Normal
        priors to the paramaters needed to evaluate the likelihood.
        """

        if param == "meh":
            return truncnorm.ppf(sample_entry, *self._range.meh, scale=0.5)

        low, high = self.get_range(param)
        value = low + (high - low) * sample_entry
        if param in self._log_uniform:
            return 10.0**value
        return value

    def inverse_prior(self, param, value):
        """Return index and value within sample to set param to given value."""

        param_ind = SampleParams._fields.index(param)
        if param == "meh":
            return param_ind, truncnorm.cdf(value, *self._range.meh, scale=0.5)
        if param in self._log_uniform:
            value = numpy.log10(value)
        low, high = self.get_range(param)
        return param_ind, (value - low) / (high - low)

    def calc_prior_loglikelihood(self, mcmc_sample):
        """Return the sum of prior log-likelihoods."""

        return 0.0
