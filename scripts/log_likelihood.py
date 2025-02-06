"""Define the log-likelihood function to use for MCMC."""

from collections import namedtuple

import numpy
from scipy.stats import norm

from download_lcs import get_astroquery as download_lcs
from extinction_correction import Green19Correction
from binary import Binary

SampleParams = namedtuple(
    "SampleParams",
    [
        "mtotal",
        "mratio",
        "age_gyr",
        "meh",
        "per",
        "esinw",
        "ecosw",
        "primary_impact_param",
        "perpass_phase",
        "primary_limb_dark_1",
        "primary_limb_dark_2",
        "secondary_limb_dark_1",
        "secondary_limb_dark_2",
        "primary_prot",
        "secondary_prot",
        "primary_reflection_coef",
        "secondary_reflection_coef",
        "primary_beaming_coef",
        "secondary_beaming_coef",
        "lc_sys",
        "sed_sys",
    ],
)


class LogLikelihood:
    """Class for calculating the log-likelihood function for a given EB."""

    def _prior_transform(self, mcmc_sample):
        """
        Return the `InputParams`_ and systematic errors per given MCMC sample.

        Apply a transformation to go identical random variables with Normal
        priors to the paramaters needed to evaluate the likelihood.
        """

        sample_entry = iter(mcmc_sample)

        def uniform_prior(low, high):
            return low + (high - low) * norm.cdf(next(sample_entry))

        # TODO: pick good priors for eBEER coefficients
        return SampleParams(
            mtotal=uniform_prior(0.2, 4),
            mratio=uniform_prior(0.01, 2),
            age_gyr=10.0 ** uniform_prior(-3, 1.1),
            meh=0.5 * next(sample_entry),
            per=uniform_prior(0.5, 300),
            esinw=uniform_prior(-0.99, 0.99),
            ecosw=uniform_prior(-0.99, 0.99),
            primary_impact_param=uniform_prior(-10, 10),
            perpass_phase=uniform_prior(-1, 1),
            primary_limb_dark_1=uniform_prior(0, 1),
            primary_limb_dark_2=uniform_prior(0, 1),
            secondary_limb_dark_1=uniform_prior(0, 1),
            secondary_limb_dark_2=uniform_prior(0, 1),
            primary_prot=uniform_prior(1, 100),
            secondary_prot=uniform_prior(1, 100),
            primary_reflection_coef=uniform_prior(0, 2),
            secondary_reflection_coef=uniform_prior(0, 2),
            primary_beaming_coef=uniform_prior(0, 2),
            secondary_beaming_coef=uniform_prior(0, 2),
            lc_sys=10.0 ** uniform_prior(-10, 0),
            sed_sys=10.0 ** uniform_prior(-10, 0),
        )

    def __init__(self, tic_id):
        """Prepare to evaluate the log-likelihood for the given TIC ID."""

        self._lcs = {
            provenance: download_lcs(tic_id, "all", provenance=provenance)
            for provenance in ["SPOC", "QLP"]
        }
        for lc_collection in self._lcs.values():
            for header, _ in lc_collection.values():
                assert header["TIMEPIXR"] == 0.5
        self._sed = Green19Correction().get_absolute_magnitudes(tic_id)
        print(f"LCs: {self._lcs!r}")
        print(f"SED: {self._sed!r}")

    def __call__(self, mcmc_sample):
        """Return the log-likelihood of the given MCMC sample."""

        sample_params = self._prior_transform(mcmc_sample)
        print(f"Sample params: {sample_params}")
        binary = Binary(from_mcmc=sample_params)
        print(f"Binary: {binary}")
        result = 0.0
        for provenance, lc_collection in self._lcs.items():
            for sector, (header, observed_lc) in lc_collection.items():
                finite = numpy.logical_and(
                    numpy.isfinite(observed_lc["PDCSAP_FLUX"]),
                    numpy.isfinite(observed_lc["PDCSAP_FLUX_ERR"]),
                )
                print(
                    f"{provenance} LC for sector {sector} has {finite.sum()} "
                    "finite points"
                )
                observed_lc = observed_lc[finite]

                model_lc = binary.get_lightcurve(
                    observed_lc["TIME"],
                    supersample_factor=100,
                    exp_time=header["INT_TIME"] * header["NUM_FRM"],
                )
                print(f'Model LC: {model_lc}')
                lc_sq_errors = (
                    observed_lc["PDCSAP_FLUX_ERR"] ** 2
                    + sample_params.lc_sys**2
                )
                print(f'Square LC errors: {lc_sq_errors}')
                result -= (
                    (observed_lc["PDCSAP_FLUX"] - model_lc) ** 2 / lc_sq_errors
                    + numpy.log(lc_sq_errors)
                ).sum()
                print(f'Log likelihood now: {result}')
        result /= 2
        print(f"Log LogLikelihood: {result!r}")
        return result


if __name__ == "__main__":
    log_likelihood = LogLikelihood(18250189)
    log_likelihood(norm.rvs(size=21))
