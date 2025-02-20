"""Define the log-likelihood function to use for MCMC."""

import logging

import pandas
from matplotlib import pyplot, use
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from scipy.stats import norm, truncnorm
from astropy.timeseries import BoxLeastSquares
from sqlalchemy import select, delete

from download_lcs import get_astroquery as download_lcs
from extinction_correction import Green19Correction
from binary import Binary
from paths import prsa_ebs
from cache_interface import CacheSession, CachedSED, CachedBLS
from sample_params import SampleParams


class LogLikelihood:
    """Class for calculating the log-likelihood function for a given EB."""

    _logger = logging.getLogger(__name__)
    _log_uniform = [
        "age_gyr",
        "primary_reflection_coef",
        "secondary_reflection_coef",
        "primary_beaming_coef",
        "secondary_beaming_coef",
        "lc_sys",
        "sed_sys",
    ]

    @staticmethod
    def _get_best_fit_bls(lightcurve):
        """Return the best fit orbital period and time of primary transit."""

        print(
            "Plotting:\n\t"
            + "\n\t".join(
                [
                    f't:{lightcurve["time"]!r}',
                    f'y:{lightcurve["flux"]!r}',
                    f'dy:{lightcurve["flux_err"]!r}',
                ]
            )
        )
        model = BoxLeastSquares(
            lightcurve["time"],
            lightcurve["flux"],
            dy=lightcurve["flux_err"],
        )
        periodogram = model.autopower(numpy.linspace(0.02, 0.2, 100))
        best_index = numpy.argmax(periodogram.power)
        if periodogram.duration[best_index] > 0.15:
            assert periodogram.period[best_index] > 2
            periodogram = model.autopower(numpy.linspace(0.1, 1.0, 100))
            best_index = numpy.argmax(periodogram.power)
        elif periodogram.duration[best_index] < 0.04:
            assert periodogram.period[best_index] < 0.5
            periodogram = model.autopower(
                numpy.linspace(0.01, 0.05, 100), maximum_period=0.5
            )
            best_index = numpy.argmax(periodogram.power)

        index_range = (
            numpy.where(
                periodogram.power[:best_index]
                < 0.3 * periodogram.power[best_index]
            )[0][-1],
            numpy.where(
                periodogram.power[best_index:]
                < 0.3 * periodogram.power[best_index]
            )[0][0]
            + best_index,
        )
        # pyplot.plot(periodogram.period, periodogram.power, "-k")
        # for color, i in zip(
        #    "rgb", [index_range[0], best_index, index_range[1]]
        # ):
        #    pyplot.axvline(x=periodogram.period[i], color=color)
        # pyplot.show()

        assert index_range[0] < best_index
        assert index_range[1] > best_index

        result = {
            param: getattr(periodogram, param)[best_index]
            for param in ["period", "duration", "transit_time"]
        }
        result["period_uncertainty"] = max(
            result["period"] - periodogram.period[index_range[0]],
            periodogram.period[index_range[1]] - result["period"],
        )
        pyplot.plot(periodogram.period, periodogram.power, "-k")
        pyplot.axvline(
            x=result["period"] - result["period_uncertainty"], color="r"
        )
        pyplot.axvline(x=result["period"], color="g")
        pyplot.axvline(
            x=result["period"] + result["period_uncertainty"], color="b"
        )
        pyplot.show()

        return result

    def plot_best_fit_bls(self, plot_fname):
        """Create plots showing the best fit BLS paramaters on top of LCs."""

        use("PDF")
        with PdfPages(plot_fname) as pdf:
            for header, lightcurve in self._lcs:
                for label, axis in pyplot.subplot_mosaic(
                    [["full", "full"], ["first", "last"]]
                )[1].items():
                    pyplot.sca(axis)
                    pyplot.plot(
                        lightcurve["time"],
                        lightcurve["flux"],
                        ".k",
                        markersize=1,
                    )
                    transit_time = self._best_fit_bls["transit_time"]
                    while transit_time < lightcurve["time"][-1]:
                        if transit_time > lightcurve["time"][0]:
                            pyplot.axvline(
                                x=transit_time
                                - self._best_fit_bls["duration"] / 2,
                                color="r",
                                linewidth=1,
                            )
                            pyplot.axvline(
                                x=transit_time, color="g", linewidth=1
                            )
                            pyplot.axvline(
                                x=transit_time
                                + self._best_fit_bls["duration"] / 2,
                                color="b",
                                linewidth=1,
                            )
                        transit_time += self._best_fit_bls["period"]
                        if label == "first":
                            pyplot.xlim(
                                lightcurve["time"][0],
                                lightcurve["time"][0]
                                + 3 * self._best_fit_bls["period"],
                            )
                        elif label == "last":
                            pyplot.xlim(
                                lightcurve["time"][-1]
                                - 3 * self._best_fit_bls["period"],
                                lightcurve["time"][-1],
                            )
                pyplot.suptitle(f"TIC {self.tic_id}, sector {header['sector']}")
                pdf.savefig()

    @staticmethod
    def _average_best_fit_bls(best_fit_bls):
        """Average the best fit BLS results for each sector."""

        min_transit_time = numpy.inf
        total_points = 0
        averaged = {
            param: 0
            for param in [
                "period",
                "duration",
                "transit_time",
                "period_uncertainty",
            ]
        }
        for bls_results in best_fit_bls:
            min_transit_time = min(
                min_transit_time, bls_results["transit_time"]
            )
            total_points += bls_results["num_points"]
            for param in ["period", "duration"]:
                averaged[param] += (
                    bls_results[param] * bls_results["num_points"]
                )
            averaged["period_uncertainty"] = max(
                averaged["period_uncertainty"],
                bls_results["period_uncertainty"],
            )
        for param in ["period", "duration"]:
            averaged[param] /= total_points

        for bls_results in best_fit_bls:
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

    def _get_cached(self, tic_id):
        """Set-up using cached information for given TIC ID if available."""

        sed = bls = None
        # False positive
        # pylint: disable=no-member
        with CacheSession.begin() as cache_session:
            # pylint: enable=no-member
            cached_sed = cache_session.execute(
                select(CachedSED).filter_by(tic_id=tic_id)
            ).scalar_one_or_none()
            cached_bls = cache_session.execute(
                select(CachedBLS).filter_by(tic_id=tic_id)
            ).scalar_one_or_none()

            if cached_sed:
                sed = (
                    numpy.array(
                        [getattr(cached_sed, f + "p1") for f in "grizy"]
                        + [getattr(cached_sed, f + "2m") for f in "jhk"]
                        + [cached_sed.w1, cached_sed.w2]
                    ),
                    numpy.array(
                        [getattr(cached_sed, f + "p1_err") for f in "grizy"]
                        + [getattr(cached_sed, f + "2m_err") for f in "jhk"]
                        + [cached_sed.w1_err, cached_sed.w2_err]
                    ),
                )

            if cached_bls:
                bls = {
                    param: getattr(cached_bls, param)
                    for param in [
                        "period",
                        "transit_time",
                        "duration",
                        "period_uncertainty",
                    ]
                }
        return sed, bls

    def _cache(self, tic_id):
        """Add the SED ind best fit BLS to cache (overwriting if necessary)."""

        # False positive
        # pylint: disable=no-member
        with CacheSession.begin() as cache_session:
            # pylint: enable=no-member
            cache_session.execute(delete(CachedSED).filter_by(tic_id=tic_id))
            cache_session.execute(delete(CachedBLS).filter_by(tic_id=tic_id))

            cache_session.add(
                CachedSED(
                    tic_id=tic_id,
                    **{f + "p1": mag for f, mag in zip("grizy", self._sed[0])},
                    **{
                        f + "2m": mag for f, mag in zip("jhk", self._sed[0][5:])
                    },
                    w1=self._sed[0][8],
                    w2=self._sed[0][9],
                    **{
                        f + "p1_err": err
                        for f, err in zip("grizy", self._sed[1])
                    },
                    **{
                        f + "2m_err": err
                        for f, err in zip("jhk", self._sed[1][5:])
                    },
                    w1_err=self._sed[1][8],
                    w2_err=self._sed[1][9],
                )
            )
            cache_session.add(CachedBLS(tic_id=tic_id, **self._best_fit_bls))

    @property
    def tic_id(self):
        """The TIC identifier of the EB being modeled."""

        return self._tic_id

    @property
    def best_fit_bls(self):
        """The best fit BLS parameters."""

        return self._best_fit_bls

    def __init__(self, tic_id, overwrite_cache=False):
        """Prepare to evaluate the log-likelihood for the given TIC ID."""

        lcs = {
            provenance: download_lcs(tic_id, "all", provenance=provenance)
            for provenance in ["SPOC", "QLP"]
        }
        self._tic_id = tic_id

        # https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0-DataProductOverview-Table:CadenceQualityFlags
        bad_spoc_mask = 0
        for bad_ind in [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]:
            bad_spoc_mask |= 1 << (bad_ind - 1)

        self._sed, self._best_fit_bls = self._get_cached(tic_id)

        best_fit_bls = []
        self._lcs = []
        for provenance, lc_collection in lcs.items():
            # TODO: figure out QLP
            if provenance == "QLP":
                continue
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
                formatted_lc = numpy.empty(
                    usable.sum(),
                    dtype=[
                        ("time", ">f8"),
                        ("flux", ">f4"),
                        ("flux_err", ">f4"),
                    ],
                )
                formatted_lc["time"] = observed_lc["TIME"]
                formatted_lc["flux"] = observed_lc["PDCSAP_FLUX"]
                formatted_lc["flux_err"] = observed_lc["PDCSAP_FLUX_ERR"]
                self._lcs.append(
                    (
                        {
                            "exptime": (
                                header["INT_TIME"] * header["NUM_FRM"] / 86400
                            ),
                            "sector": sector,
                            "provenance": provenance,
                        },
                        formatted_lc,
                    )
                )

                if self._best_fit_bls is None or overwrite_cache:
                    bls_results = self._get_best_fit_bls(formatted_lc)
                    bls_results["num_points"] = formatted_lc.size
                    best_fit_bls.append(bls_results)
                    self._logger.debug(
                        "%s LC for sector %d has %s usable points, BLS "
                        "results: period = %s, transit time = %s, "
                        "duration = %s",
                        provenance,
                        sector,
                        repr(formatted_lc.size),
                        bls_results["period"],
                        bls_results["transit_time"],
                        bls_results["duration"],
                    )
        if self._best_fit_bls is None or overwrite_cache:
            self._best_fit_bls = self._average_best_fit_bls(best_fit_bls)
            overwrite_cache = True

        self._range = SampleParams(
            mtotal=(0.2, 4),
            mratio=(0.01, 2),
            age_gyr=(-3, 1.1),
            meh=Binary.meh_range,
            per=(0.5, 300),
            ecc=(0, 0.96),
            w=(0, 360),
            primary_impact_param=(-10, 10),
            eclipse_time=(
                self._best_fit_bls["transit_time"]
                - 5.0 * self._best_fit_bls["period"],
                self._best_fit_bls["transit_time"]
                + 5.0 * self._best_fit_bls["period"],
            ),
            primary_limb_dark_1=(0, 1),
            primary_limb_dark_2=(0, 1),
            secondary_limb_dark_1=(0, 1),
            secondary_limb_dark_2=(0, 1),
            primary_prot=(1, 100),
            secondary_prot=(1, 100),
            primary_reflection_coef=(-2, 2),
            secondary_reflection_coef=(-2, 2),
            primary_beaming_coef=(-2, 2),
            secondary_beaming_coef=(-2, 2),
            lc_sys=(-10, 0),
            sed_sys=(-10, 0),
        )

        assert self._best_fit_bls["period"] > self._range.per[0]

        self._logger.info(
            "Averaged best fit BLS: "
            "period = %s, transit time = %s, duration = %s",
            self._best_fit_bls["period"],
            self._best_fit_bls["transit_time"],
            self._best_fit_bls["duration"],
        )

        if self._sed is None or overwrite_cache:
            self._sed = Green19Correction().get_absolute_magnitudes(tic_id)[0]
            overwrite_cache = True

        self._logger.debug("LCs: %s", repr(self._lcs))
        self._logger.debug("SED: %s", repr(self._sed))

        if overwrite_cache:
            self._cache(tic_id)

    def prior_transform(self, param, sample_entry):
        """
        Return the value of the given parameter given a sample entry.

        Apply a transformation to go from identical random variables with Normal
        priors to the paramaters needed to evaluate the likelihood.
        """

        if param == "meh":
            return truncnorm.ppf(
                norm.cdf(sample_entry), *self._range.meh, scale=0.5
            )
        low, high = getattr(self._range, param)
        value = low + (high - low) * norm.cdf(sample_entry)
        if param in self._log_uniform:
            return 10.0**value
        return value

    def inverse_prior(self, param, value):
        """Return value and index within sample to set param to given value."""

        param_ind = SampleParams._fields.index(param)
        if param == "meh":
            return param_ind, norm.ppf(
                truncnorm.cdf(value, *self._range.meh, scale=0.5)
            )
        if param in self._log_uniform:
            value = numpy.log10(value)
        low, high = getattr(self._range, param)
        return param_ind, norm.ppf((value - low) / (high - low))

    def get_sample_params(self, mcmc_sample):
        """Return prior-transformed parameters given sample."""

        return SampleParams(
            *[
                self.prior_transform(param, sample_entry)
                for param, sample_entry in zip(
                    SampleParams._fields, mcmc_sample
                )
            ]
        )

    def calc_lc_log_likelihood(self, binary, lc_sys_err):
        """Return log-likelihood of observing the TESS LCs for given binary."""

        result = 0.0
        for header, lightcurve in self._lcs:
            lc_sq_errors = lightcurve["flux_err"] ** 2 + lc_sys_err**2
            model_lc = binary.get_lightcurve(
                lightcurve["time"],
                supersample_factor=100,
                exp_time=header["exptime"],
            )

            self._logger.debug("Model LC:\n%s", repr(model_lc))

            model_lc *= (model_lc * lightcurve["flux"] / lc_sq_errors).sum() / (
                model_lc**2 / lc_sq_errors
            ).sum()

            if getattr(self, "enable_plots", False):
                pyplot.plot(lightcurve["time"], lightcurve["flux"], "-r")
                pyplot.plot(lightcurve["time"], model_lc, "-b")
                if isinstance(self.enable_plots, str):
                    pyplot.savefig(self.enable_plots.format_map(header))
                else:
                    pyplot.show()
                pyplot.cla()
                pyplot.clf()

            self._logger.debug("Square LC errors: %s", repr(lc_sq_errors))
            result -= (
                (lightcurve["flux"] - model_lc) ** 2 / lc_sq_errors
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

        sample_params = self.get_sample_params(mcmc_sample)
        self._logger.debug("Sample params: %s", sample_params)
        binary = Binary(from_mcmc=sample_params)
        if binary.out_of_range:
            self._logger.warning(
                "Out of range parameters:\n\t%s",
                "\n\t".join(binary.out_of_range),
            )
            return (-numpy.inf,) + sample_params
        self._logger.debug("Binary: %s", binary)

        result = (
            norm.logpdf(mcmc_sample).sum()
            + self.calc_lc_log_likelihood(binary, sample_params.lc_sys)
            + self.calc_sed_log_likelihood(binary, sample_params.sed_sys)
        )

        self._logger.debug(
            "Final log likelihood(%s): %s", repr(mcmc_sample), repr(result)
        )
        return (result,) + sample_params


class LogLikelihoodPriorsOnly(LogLikelihood):
    """Allows MCMC sampling using just the priors for testing."""

    def __call__(self, mcmc_sample):
        """Return the log-likelihood of the given MCMC sample."""

        return (norm.logpdf(mcmc_sample).sum(),) + self.get_sample_params(
            mcmc_sample
        )


if __name__ == "__main__":
    test_tic = 189639080
    eb_cat = pandas.read_csv(prsa_ebs, index_col="tess_id")
    print(f"Prsa EB params for TIC {test_tic}: {eb_cat.loc[test_tic]!r}")
    logging.basicConfig(level=logging.DEBUG)
    log_likelihood = LogLikelihood(test_tic)
    log_likelihood.enable_plots = True#"TESS189639080_s{sector}_best_fit_model3.pdf"
    log_likelihood(
        numpy.array(
            [
                -1.24408646e00,
                -3.71483063e00,
                1.41475127e-01,
                8.34203817e-01,
                -2.37418813e00,
                -2.24196430e00,
                -8.30168794e-02,
                8.71420582e-06,
                -1.08640447e-03,
                1.72306533e00,
                -7.87778788e-01,
                -2.47859827e00,
                3.18854784e00,
                9.90237029e-01,
                7.10868651e-01,
                -3.36746148e-01,
                5.12819850e-01,
                -2.99760512e00,
                2.40542923e00,
                -7.98815313e-01,
                -8.75360123e-01,
            ]
        )
    )
    log_likelihood.plot_best_fit_bls("best_fit_bls.pdf")
