"""Define the log-likelihood function to use for MCMC."""

import logging
from functools import partial

import numpy
from scipy.stats import norm, truncnorm
from sqlalchemy import select, delete

from tess_target import TESSTarget, get_bls_eclipse_mask
from detrending import masked_detrend, calc_moving_median
from extinction_correction import Green19Correction
from binary import Binary
from cache_interface import CacheSession, CachedSED, CachedBLS
from sample_params import SampleParams


class LogLikelihood(TESSTarget):
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

    max_ecc = 0.96
    max_outlier_iterations = 5
    outlier_thresh2 = 25.0
    outlier_bins = 30

    @classmethod
    def get_cached_sed_and_bls(cls, tic_id):
        """Set-up using cached information for given TIC ID if available."""

        def none_to_nan(value):
            return numpy.nan if value is None else value

        sed = None
        bls = None
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
                cls._logger.debug("Found cached SED: %s", repr(cached_sed))

                sed = (
                    numpy.array(
                        [
                            none_to_nan(getattr(cached_sed, f + "p1"))
                            for f in "grizy"
                        ]
                        + [
                            none_to_nan(getattr(cached_sed, f + "2m"))
                            for f in "jhk"
                        ]
                        + [
                            none_to_nan(cached_sed.w1),
                            none_to_nan(cached_sed.w2),
                        ]
                    ),
                    numpy.array(
                        [
                            none_to_nan(getattr(cached_sed, f + "p1_err"))
                            for f in "grizy"
                        ]
                        + [
                            none_to_nan(getattr(cached_sed, f + "2m_err"))
                            for f in "jhk"
                        ]
                        + [
                            none_to_nan(cached_sed.w1_err),
                            none_to_nan(cached_sed.w2_err),
                        ]
                    ),
                )

            if cached_bls:
                cls._logger.debug("Found cached BLS: %s", repr(cached_bls))
                bls_columns = [
                    column.key for column in CachedBLS.__table__.columns
                ]
                print(f"BLS columns: {bls_columns!r}")
                bls = {
                    column: (
                        (
                            getattr(cached_bls, column),
                            getattr(cached_bls, column + "_uncertainty"),
                        )
                        if (column + "_uncertainty" in bls_columns)
                        else getattr(cached_bls, column)
                    )
                    for column in bls_columns
                    if not column.endswith("_uncertainty")
                }
                cls._logger.debug("Loaded cached BLS: %s", repr(bls))
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
            cache_session.add(
                CachedBLS(
                    tic_id=tic_id,
                    **{
                        column: (
                            value[0] if isinstance(value, tuple) else value
                        )
                        for column, value in self._best_fit_bls.items()
                    },
                    **{
                        column + "_uncertainty": value[1]
                        for column, value in self._best_fit_bls.items()
                        if isinstance(value, tuple)
                    },
                )
            )

    @property
    def best_fit_bls(self):
        """The best fit BLS parameters."""

        return self._best_fit_bls

    @property
    def sed(self):
        """The observed spectral energy distribution in absolute magnitudes."""

        return self._sed

    @property
    def bls_porb(self):
        """The best fit orbital period of the target."""

        return (1 if self.masked_is_significant() else 2) * self._best_fit_bls[
            "period"
        ][0]

    def get_range(self, param):
        """The support of the prior for a given parameter."""

        return getattr(self._range, param)

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tic_id,
        *,
        overwrite_cache=False,
        ignore_extinction_flags=True,
        plot_bls=False,
        detrend=partial(masked_detrend, get_trend=calc_moving_median),
    ):
        """Prepare to evaluate the log-likelihood for the given TIC ID."""

        super().__init__(tic_id)

        # https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0-DataProductOverview-Table:CadenceQualityFlags

        self._sed, self._best_fit_bls = self.get_cached_sed_and_bls(tic_id)

        if self._sed is None or overwrite_cache:
            self._sed = Green19Correction(
                ignore_extinction_flags
            ).get_absolute_magnitudes(tic_id)[0]
            overwrite_cache = True

        if self._best_fit_bls is None or overwrite_cache:
            self._best_fit_bls = self.fit_bls(plot_bls)
            overwrite_cache = True

        if overwrite_cache:
            self._cache(tic_id)

        self._range = SampleParams(
            mtotal=(0.2, 4),
            mratio=(0.01, 2),
            age_gyr=(-3, 1.1),
            meh=Binary.meh_range,
            per=(0.1, 300),
            ecc=(0, self.max_ecc),
            w=(-360, 360),
            primary_impact_param=(-10, 10),
            eclipse_time=(
                self._best_fit_bls["transit_time"]
                - 5.0 * self._best_fit_bls["period"][0],
                self._best_fit_bls["transit_time"]
                + 5.0 * self._best_fit_bls["period"][0],
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
            lc_sys=(min(numpy.log10(self._find_syserr_and_outliers()), -1), 0),
            sed_sys=(-10, -1.5),
        )

        assert self._best_fit_bls["period"][0] > 2 * self._range.per[0]

        if detrend is not None:
            self._lcs = [
                (header, detrend(lightcurve, header["exptime"], self))
                for header, lightcurve in self._lcs
            ]

        self._logger.debug("LCs: %s", repr(self._lcs))
        self._logger.debug("SED: %s", repr(self._sed))

    # pylint: enable=too-many-arguments

    def masked_is_significant(self):
        """
        Return True iff the masked BLS fit appears to fit real eclipses.

        To be marked as significant all of the following must be satisfied:

          * maked period should be close to unmasked period (within 5 unmasked
            uncertanties)

          * depth should exceed its uncertanity by at least a factor of 5

          * masked_harmonic_delta_log_likelihood < -5
        """

        bls = self._best_fit_bls
        return (
            abs(bls["masked_period"][0] - bls["period"][0])
            < 5.0 * bls["period"][1]
            and bls["masked_depth"][0] > 5.0 * bls["masked_depth"][1]
            and bls["masked_harmonic_delta_log_likelihood"] < -5.0
        )

    @staticmethod
    def get_model(binary, header, lightcurve, lc_sys_err):
        """Fit the model scaling to match that of the lightcurve."""

        lc_sq_errors = lightcurve["flux_err"] ** 2 + lc_sys_err**2

        model_lc = binary.get_lightcurve(
            lightcurve["time"],
            supersample_factor=100,
            exp_time=header["exptime"],
        )

        model_lc *= (model_lc * lightcurve["flux"] / lc_sq_errors).sum() / (
            model_lc**2 / lc_sq_errors
        ).sum()
        return model_lc, lc_sq_errors

    def prior_transform(self, param, sample_entry):
        """
        Return the value of the given parameter given a sample entry.

        Apply a transformation to go from identical random variables with Normal
        priors to the paramaters needed to evaluate the likelihood.
        """

        if param == "meh":
            return truncnorm.ppf(
                norm.cdf(sample_entry),
                2 * self._range.meh[0],
                2 * self._range.meh[1],
                scale=0.5,
            )
        low, high = self.get_range(param)
        value = low + (high - low) * norm.cdf(sample_entry)
        if param in self._log_uniform:
            return 10.0**value
        return value

    def inverse_prior(self, param, value):
        """Return index and value within sample to set param to given value."""

        param_ind = SampleParams._fields.index(param)
        if param == "meh":
            return param_ind, norm.ppf(
                truncnorm.cdf(
                    value,
                    2 * self._range.meh[0],
                    2 * self._range.meh[1],
                    scale=0.5,
                )
            )
        if param in self._log_uniform:
            value = numpy.log10(value)
        low, high = self.get_range(param)
        self._logger.debug(
            "Inverse prior %s: value=%s, range=(%s, %s)",
            param,
            repr(value),
            repr(low),
            repr(high),
        )
        if value < low:
            return param_ind, -numpy.inf
        if value > high:
            return param_ind, numpy.inf
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

    def _bin_lightcurve(self, lightcurve, num_bins, phase):
        """Bin the given lightcurve in phase."""

        bin_destinations = (phase) // (1.0 / num_bins)
        binned_lc = {
            quantity: numpy.full(num_bins, numpy.nan)
            for quantity in ["time", "flux"]
        }
        for quantity, binned in binned_lc.items():
            for bin_ind in range(num_bins):
                in_bin = bin_destinations == bin_ind
                binned[bin_ind] = numpy.median(lightcurve[quantity][in_bin])
        return binned_lc

    def _find_syserr_and_outliers(self):
        """
        Estimate the systematic error and find outliers in the lightcurves.

        Bin the phase-folded eclipse masked LC in phase after masking eclipses
        and find the variance from the median in each bin after iteratively
        rejecting outliers.

        Modify `self.max_outlier_iterations`, `self.outlier_thresh2` and
        `self.outlier_bins` attributes to control the process.
        """

        def calc_bin_var(bin_flux):
            """Calculate stddev(flux) and flag >5 sigma outliers."""

            assert numpy.isfinite(bin_flux).all()
            outliers = None
            for _ in range(self.max_outlier_iterations):
                square_dev = (bin_flux - numpy.median(bin_flux)) ** 2
                var = numpy.mean(
                    square_dev
                    if outliers is None
                    else square_dev[numpy.logical_not(outliers)]
                )
                assert numpy.isfinite(var)
                new_outliers = square_dev > self.outlier_thresh2 * var
                if outliers is None:
                    outliers = new_outliers
                elif (outliers == new_outliers).all():
                    break
            return var, outliers

        combined_lc = self.get_combined_lc()
        period = self._best_fit_bls["period"][0]
        if self.masked_is_significant():
            ooe_mask = numpy.logical_not(
                numpy.logical_or(
                    get_bls_eclipse_mask(
                        self._best_fit_bls, combined_lc, "both"
                    ),
                    get_bls_eclipse_mask(
                        self._best_fit_bls, combined_lc, "masked"
                    ),
                )
            )
        else:
            period *= 2
            ooe_mask = numpy.logical_not(
                get_bls_eclipse_mask(self._best_fit_bls, combined_lc, "both")
            )

        phase = (combined_lc["time"] % period) / period
        phase_sorter = numpy.argsort(phase)
        ooe_mask = ooe_mask[phase_sorter]
        phase_sorter = phase_sorter[ooe_mask]

        bin_boundaries = numpy.searchsorted(
            phase[phase_sorter], numpy.linspace(0, 1, self.outlier_bins + 1)
        )
        sys_err = numpy.zeros(self.outlier_bins)
        outliers = numpy.zeros_like(combined_lc["flux"], dtype=bool)

        weights = numpy.empty(shape=self.outlier_bins, dtype=float)

        for i in range(self.outlier_bins):
            bin_indices = phase_sorter[
                bin_boundaries[i] : bin_boundaries[i + 1]
            ]
            if bin_indices.size < 10:
                weights[i] = 0.0
                continue
            sys_err[i], outliers[bin_indices] = calc_bin_var(
                combined_lc["flux"][bin_indices]
            )
            sys_err[i] -= (
                numpy.median(combined_lc["flux_err"][bin_indices]) ** 2
            )
            if sys_err[i] < 0:
                sys_err[i] = 0
            weights[i] = numpy.logical_not(outliers[bin_indices]).sum() - 1
            if weights[i] < 10:
                weights[i] = 0.0
                sys_err[i] = 0.0

        sys_err = numpy.average(sys_err, weights=weights) ** 0.5

        lc_start = 0
        for _, formatted_lc in self.lcs:
            formatted_lc["good"] = numpy.logical_not(
                outliers[lc_start : lc_start + formatted_lc.size]
            )
            lc_start += formatted_lc.size

        self._logger.info(
            "TIC %d: LC systematic error: %f, outliers %d",
            self.tic_id,
            sys_err,
            outliers.sum(),
        )
        return max(sys_err, 1e-10)

    def calc_lc_log_likelihood(self, binary, lc_sys_err, eclipse_only=False):
        """
        Return log-likelihood of observing the TESS LCs for given binary.

        Args:
            binary(Binary):    The binary for which to evaluate the model to
                compare to the lightcurves.

            lc_sys_err(float):    The systematic error to assume for
                lightcurves. (added in quadrature to formal errors).

            eclipse_only(str or False):    If not False, only lightcurve points
            near eclipse are considered:

                * even: Only the points in the vicinity of even eclipses (per
                  best fit BLS) are included.

                * odd: Only the points in the vicinity of even eclipses (per
                  best fit BLS) are included.

                * both: Points near both eclipses.

                * masked: Points near the eclipses reported in the masked BLS
                  are considered
        """

        if binary.a < 1 + binary.rp or binary.out_of_range:
            return -numpy.inf
        result = 0.0
        for header, lightcurve in self._lcs:
            if lightcurve["good"].sum() < 10:
                continue
            lightcurve = lightcurve[lightcurve["good"]]
            if eclipse_only:
                lightcurve = lightcurve[
                    get_bls_eclipse_mask(
                        self._best_fit_bls, lightcurve, eclipse_only
                    )
                ]
            model_lc, lc_sq_errors = self.get_model(
                binary, header, lightcurve, lc_sys_err
            )
            result -= (
                (lightcurve["flux"] - model_lc) ** 2 / lc_sq_errors
                + numpy.log(lc_sq_errors)
            ).sum()
            self._logger.debug("Log likelihood now: %s", repr(result / 2))

        if not numpy.isfinite(result):
            self._logger.error(
                "Non-finite log-likelihood for binary: %s", binary
            )
        return result / 2

    def calc_sed_log_likelihood(self, binary, sed_sys_err):
        """Return log-likelihood of observed SED for given binary."""

        finite = numpy.isfinite(self._sed[0])
        sed_sq_errors = self._sed[1][finite] ** 2 + sed_sys_err**2

        result = (
            -(
                (self._sed[0][finite] - binary.absmag[finite]) ** 2
                / sed_sq_errors
                + numpy.log(sed_sq_errors)
            ).sum()
            / 2
        )
        self._logger.debug("SED log-likelihood: %s", result)
        return result

    def calc_prior_loglikelihood(self, mcmc_sample):
        """Return the sum of prior log-likelihoods."""

        return norm.logpdf(mcmc_sample).sum()

    def save_jktebob_lc(self, filename, provenance="SPOC"):
        """Create a file with given name suitable to run through JKTEBOB."""

        with open(filename, "w", encoding="ascii") as outf:
            for header, lightcurve in self._lcs:
                if provenance == "all" or header["provenance"] != provenance:
                    continue
                normalized = lightcurve[:]
                normalized["flux"] /= numpy.nanmedian(lightcurve["flux"])
                normalized["time"] -= self._best_fit_bls["transit_time"]
                for row in normalized:
                    outf.write(
                        f"{row['time']:-25.16g} "
                        f"{-2.5*numpy.log10(row['flux']):-25.16g} "
                        "noerr\n"
                    )

    def __call__(self, mcmc_sample, exclude_priors=False):
        """Return the log-likelihood of the given MCMC sample."""

        sample_params = self.get_sample_params(mcmc_sample)
        self._logger.debug("Sample params: %s", sample_params)
        try:
            binary = Binary(from_mcmc=sample_params)
        except ValueError as error:
            self._logger.warning(
                "Attempted log-likelihood evaluation for out of range "
                "parameters (returning -inf):\n%s\n%s",
                sample_params,
                error.args[0],
            )
            return (-numpy.inf,) + sample_params
        if binary.out_of_range:
            self._logger.warning(
                "Out of range parameters:\n\t%s",
                "\n\t".join(binary.out_of_range),
            )
            return (-numpy.inf,) + sample_params
        self._logger.debug("Binary: %s", binary)

        result = (
            self.calc_prior_loglikelihood(
                mcmc_sample[numpy.logical_not(exclude_priors)]
            )
            + self.calc_lc_log_likelihood(binary, sample_params.lc_sys)
            + self.calc_sed_log_likelihood(binary, sample_params.sed_sys)
        )

        self._logger.debug(
            "Final log likelihood(%s): %s", repr(mcmc_sample), repr(result)
        )
        return (result,) + sample_params


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


def experiment():
    """Manually experiment with things."""

    test_tic = 189639080
    logging.basicConfig(level=logging.DEBUG)

    log_likelihood = LogLikelihood(test_tic)
    log_likelihood.save_jktebob_lc(f"tess{test_tic}_jktebob.dat")


if __name__ == "__main__":
    experiment()
