"""Define the log-likelihood function to use for MCMC."""

from collections import namedtuple
import logging

from matplotlib import pyplot
import numpy
from scipy.stats import norm, truncnorm
from astropy.timeseries import BoxLeastSquares

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
        "ecc",
        "w",
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

    _logger = logging.getLogger(__name__)

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
        # TODO: limit [M/H] to CMD interpolation range
        return SampleParams(
            mtotal=uniform_prior(0.2, 4),
            mratio=uniform_prior(0.01, 2),
            age_gyr=10.0 ** uniform_prior(-3, 1.1),
            meh=truncnorm.ppf(
                norm.cdf(next(sample_entry)), *Binary.meh_range, scale=0.5
            ),
            per=uniform_prior(0.5, 300),
            ecc=uniform_prior(0, 1),
            w=uniform_prior(0, 360),
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

    @staticmethod
    def _get_best_fit_bls(observed_lc):
        """Return the best fit orbital period and time of primary transit."""

        print(
            "Plotting:\n\t"
            + "\n\t".join(
                [
                    f't:{observed_lc["TIME"]!r}',
                    f'y:{observed_lc["PDCSAP_FLUX"]!r}',
                    f'dy:{observed_lc["PDCSAP_FLUX_ERR"]!r}',
                ]
            )
        )
        model = BoxLeastSquares(
            observed_lc["TIME"],
            observed_lc["PDCSAP_FLUX"],
            dy=observed_lc["PDCSAP_FLUX_ERR"],
        )
        periodogram = model.autopower(numpy.linspace(0.02, 0.2, 100))
        best_index = numpy.argmax(periodogram.power)
        result = {
            param: getattr(periodogram, param)[best_index]
            for param in ["period", "duration", "transit_time"]
        }

        return result

        pyplot.plot(observed_lc["TIME"], observed_lc["PDCSAP_FLUX"], "-g")
        transit_time = result["transit_time"]
        while transit_time < observed_lc["TIME"][-1]:
            pyplot.axvline(x=transit_time - result["duration"] / 2, color="r")
            pyplot.axvline(x=transit_time + result["duration"] / 2, color="b")
            transit_time += result["period"]
        pyplot.show()

    @staticmethod
    def _average_best_fit_bls(best_fit_bls):
        """Average the best fit BLS results for each sector."""

        min_transit_time = numpy.inf
        total_points = 0
        averaged = {
            param: 0 for param in ["period", "duration", "transit_time"]
        }
        for bls_collection in best_fit_bls.values():
            for bls_results in bls_collection.values():
                min_transit_time = min(
                    min_transit_time, bls_results["transit_time"]
                )
                total_points += bls_results["num_points"]
                for param in ["period", "duration"]:
                    averaged[param] += (
                        bls_results[param] * bls_results["num_points"]
                    )
        for param in ["period", "duration"]:
            averaged[param] /= total_points

        for bls_collection in best_fit_bls.values():
            for bls_results in bls_collection.values():
                averaged["transit_time"] += (
                    bls_results["transit_time"]
                    - numpy.round(
                        (bls_results["transit_time"] - min_transit_time)
                        / averaged["period"]
                    )
                    * averaged["period"]
                ) * bls_results["num_points"]
        averaged["transit_time"] /= total_points
        return averaged

    def __init__(self, tic_id):
        """Prepare to evaluate the log-likelihood for the given TIC ID."""

        self._lcs = {
            provenance: download_lcs(tic_id, "all", provenance=provenance)
            for provenance in ["SPOC", "QLP"]
        }
        bad_spoc_mask = 0
        # See: https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0-DataProductOverview-Table:CadenceQualityFlags
        for bad_ind in [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]:
            bad_spoc_mask |= 1 << (bad_ind - 1)

        best_fit_bls = {}
        for provenance, lc_collection in self._lcs.items():
            # TODO: figure out QLP
            if provenance == "QLP":
                continue
            print(f"Provenance: {provenance}")
            best_fit_bls[provenance] = {}
            for sector, (header, observed_lc) in lc_collection.items():

                assert header["TIMEPIXR"] == 0.5
                usable = numpy.logical_and(
                    numpy.isfinite(observed_lc["PDCSAP_FLUX"]),
                    numpy.isfinite(observed_lc["PDCSAP_FLUX_ERR"]),
                )
                usable = numpy.logical_and(
                    usable,
                    numpy.logical_not(observed_lc["QUALITY"] & bad_spoc_mask),
                )
                observed_lc = observed_lc[usable]
                lc_collection[sector] = (header, observed_lc)
                bls_results = self._get_best_fit_bls(observed_lc)
                bls_results["num_points"] = usable.sum()
                best_fit_bls[provenance][sector] = bls_results
                self._logger.debug(
                    "%s LC for sector %d has %s usable points, BLS results: "
                    "period = %s, transit time = %s, duration = %s",
                    provenance,
                    sector,
                    repr(bls_results["num_points"]),
                    bls_results["period"],
                    bls_results["transit_time"],
                    bls_results["duration"],
                )
        self._best_fit_bls = self._average_best_fit_bls(best_fit_bls)
        self._logger.info(
            "Averaged best fit BLS: "
            "period = %s, transit time = %s, duration = %s",
            self._best_fit_bls["period"],
            self._best_fit_bls["transit_time"],
            self._best_fit_bls["duration"],
        )

        self._sed = Green19Correction().get_absolute_magnitudes(tic_id)[0]
        self._logger.debug("LCs: %s", repr(self._lcs))
        self._logger.debug("SED: %s", repr(self._sed))

    def calc_lc_log_likelihood(self, binary, lc_sys_err):
        """Return log-likelihood of observing the TESS LCs for given binary."""

        result = 0.0
        for provenance, lc_collection in self._lcs.items():
            # TODO: figure out QLP
            if provenance == "QLP":
                continue
            for sector, (header, observed_lc) in lc_collection.items():
                lc_sq_errors = (
                    observed_lc["PDCSAP_FLUX_ERR"] ** 2 + lc_sys_err**2
                )
                model_lc = binary.get_lightcurve(
                    observed_lc["TIME"],
                    supersample_factor=100,
                    exp_time=header["INT_TIME"] * header["NUM_FRM"],
                )

                self._logger.debug("Model LC:\n%s", repr(model_lc))

                model_lc *= (
                    model_lc * observed_lc["PDCSAP_FLUX"] / lc_sq_errors
                ).sum() / (model_lc**2 / lc_sq_errors).sum()

                self._logger.debug("Square LC errors: %s", repr(lc_sq_errors))
                result -= (
                    (observed_lc["PDCSAP_FLUX"] - model_lc) ** 2 / lc_sq_errors
                    + numpy.log(lc_sq_errors)
                ).sum()
                self._logger.debug("Log likelihood now: %s", repr(result / 2))

        return result / 2

    def calc_sed_log_likelihood(self, binary, sed_sys_err):
        """Return log-likelihood of observed SED for given binary."""

        sed_sq_errors = self._sed[1] ** 2 + sed_sys_err**2
        self._logger.debug(
            "Adding SED log-likelihood. Observed: %s, model: %s, unc^2: %s",
            self._sed[0],
            binary.absmag,
            sed_sq_errors,
        )

        result = (
            -(
                (self._sed[0] - binary.absmag) ** 2 / sed_sq_errors
                + numpy.log(sed_sq_errors)
            ).sum()
            / 2
        )
        self._logger.debug("SED log-likelihood: %s", result)
        return result

    def __call__(self, mcmc_sample):
        """Return the log-likelihood of the given MCMC sample."""

        sample_params = self._prior_transform(mcmc_sample)
        self._logger.debug("Sample params: %s", sample_params)
        binary = Binary(from_mcmc=sample_params)
        self._logger.debug("Binary: %s", binary)

        return self.calc_lc_log_likelihood(
            binary, sample_params.lc_sys
        ) + self.calc_sed_log_likelihood(binary, sample_params.sed_sys)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_likelihood = LogLikelihood(11119600)
    log_likelihood.get_best_fit_bls()
    exit(0)
    for _ in range(20):
        log_likelihood._logger.info(
            "Final log likelihood: %s", repr(log_likelihood(norm.rvs(size=21)))
        )
