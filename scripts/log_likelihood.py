"""Define the log-likelihood function to use for MCMC."""

import logging

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import pandas
import numpy
from scipy.stats import norm, truncnorm
from sqlalchemy import select, delete

from tess_target import TESSTarget
from extinction_correction import Green19Correction
from binary import Binary
from paths import prsa_ebs
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

    def _get_lc_eclipses(self, lightcurve, which):
        """
        Filter the lightcure to leave only point per mask.

        See `self.calc_lc_log_likelihood()` for what masks are supported.
        """

        if which == "masked":
            period = self._best_fit_bls["masked_period"][0]
            window = 2.0 * (
                self._best_fit_bls["masked_duration"]
                + 2.0 * self._best_fit_bls["masked_period"][1]
            )
            time = (
                lightcurve["time"] - self._best_fit_bls["masked_transit_time"]
            )
        else:
            period = self._best_fit_bls["period"][0] * (
                1 if which == "both" else 2
            )
            window = 2.0 * (
                self._best_fit_bls["duration"]
                + 2.0 * self._best_fit_bls["period"][1]
            )
            time = lightcurve["time"] - self._best_fit_bls["transit_time"]
            if which == "odd":
                time -= self._best_fit_bls["period"][0]

        folded = time % period
        return lightcurve[numpy.minimum(folded, period - folded) < window]

    def _iter_lc_and_model(self, binary, lc_sys_err, eclipse_only):
        """Iterate over LC data and model given binary parameters."""

        for header, lightcurve in self._lcs:
            if eclipse_only:
                lightcurve = self._get_lc_eclipses(lightcurve, eclipse_only)
            lc_sq_errors = lightcurve["flux_err"] ** 2 + lc_sys_err**2

            model_lc = binary.get_lightcurve(
                lightcurve["time"],
                supersample_factor=100,
                exp_time=header["exptime"],
            )

            model_lc *= (model_lc * lightcurve["flux"] / lc_sq_errors).sum() / (
                model_lc**2 / lc_sq_errors
            ).sum()
            yield header, lightcurve, model_lc, lc_sq_errors

    @property
    def tic_id(self):
        """The TIC identifier of the EB being modeled."""

        return self._tic_id

    @property
    def best_fit_bls(self):
        """The best fit BLS parameters."""

        return self._best_fit_bls

    def get_range(self, param):
        """The support of the prior for a given parameter."""

        return getattr(self._range, param)

    def __init__(
        self,
        tic_id,
        overwrite_cache=False,
        ignore_extinction_flags=True,
        plot_bls=False,
    ):
        """Prepare to evaluate the log-likelihood for the given TIC ID."""

        super().__init__(tic_id)
        self._tic_id = tic_id

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
            per=(0.5, 300),
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
            lc_sys=(-10, 0),
            sed_sys=(-10, 0),
        )

        assert self._best_fit_bls["period"][0] > self._range.per[0]

        self._logger.debug("LCs: %s", repr(self._lcs))
        self._logger.debug("SED: %s", repr(self._sed))

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
                truncnorm.cdf(value, *self._range.meh, scale=0.5)
            )
        if param in self._log_uniform:
            value = numpy.log10(value)
        low, high = self.get_range(param)
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

    def plot_lc_model_comparison(
        self, sample_or_binary, pdf=None, extra_title=""
    ):
        """Create multi-page PDF showing the model over LC data for each LC."""

        if isinstance(sample_or_binary, Binary):
            binary = sample_or_binary
            lc_sys = 0.0
        elif isinstance(sample_or_binary, SampleParams):
            binary = Binary(from_mcmc=sample_or_binary)
            lc_sys = sample_or_binary.lc_sys
        else:
            sample_params = self.get_sample_params(sample_or_binary)
            binary = Binary(from_mcmc=sample_params)
            lc_sys = sample_params.lc_sys

        self._logger.debug(
            "Plotting LC model comparison for binary: %s", binary
        )

        #num_bins = 100
        for mask_name in [None, "deeper", "shallower"]:
            for header, lightcurve, model_lc, _ in self._iter_lc_and_model(
                binary, lc_sys, mask_name
            ):
                pyplot.figure(figsize=[4.8, 6.4])
                pyplot.subplot(211)
                pyplot.plot(lightcurve["time"], lightcurve["flux"], "-r")
                pyplot.plot(lightcurve["time"], model_lc, "-b")
                pyplot.xlabel("Time [days]")
                pyplot.ylabel("Flux [ppm]")
                pyplot.subplot(212)
                pyplot.plot(
                    (lightcurve["time"] % binary.per) / binary.per,
                    lightcurve["flux"],
                    ",",
                    zorder=10,
                )
                phase = (lightcurve["time"] % binary.per) / binary.per
                # binned_lc = self._bin_lightcurve(lightcurve, num_bins, phase)
                # pyplot.plot(
                #    (binned_lc["time"] % binary.per) / binary.per,
                #    binned_lc["flux"],
                #    "o",
                #    markersize=3,
                #    zorder=20,
                # )

                phase_sort = numpy.argsort(phase)
                pyplot.plot(
                    phase[phase_sort], model_lc[phase_sort], "-", zorder=30
                )
                pyplot.xlabel(f"Phase (Porb={binary.per!r})")
                pyplot.ylabel("Flux [ppm]")

                pyplot.suptitle(
                    f"TIC {self.tic_id}, sector {header['sector']}"
                    + f": {extra_title}"
                    if extra_title
                    else ""
                )
                if pdf is None:
                    pyplot.show()
                else:
                    pdf.savefig()
                pyplot.cla()
                pyplot.clf()

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

        if binary.a < 1 + binary.rp:
            return -numpy.inf
        result = 0.0
        for _, lightcurve, model_lc, lc_sq_errors in self._iter_lc_and_model(
            binary, lc_sys_err, eclipse_only
        ):
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
                error.args[0]
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


def manual_plot():
    """Plot the lightcurve for a given set of parameters."""

    params = SampleParams(
        mtotal=1.5,
        mratio=0.5,
        age_gyr=1.0,
        meh=0.0,
        per=numpy.pi,
        ecc=0.8,
        w=63.0,
        primary_impact_param=0.0,
        eclipse_time=1.0,
        primary_limb_dark_1=0.0,
        primary_limb_dark_2=0.0,
        secondary_limb_dark_1=0.0,
        secondary_limb_dark_2=0.0,
        primary_prot=100.0,
        secondary_prot=100.0,
        primary_reflection_coef=0.0,
        secondary_reflection_coef=0.0,
        primary_beaming_coef=0.0,
        secondary_beaming_coef=0.0,
        lc_sys=0.0,
        sed_sys=0.0,
    )
    binary = Binary(from_mcmc=params)

    plot_t = numpy.linspace(0.0, 3.0, 1000)
    pyplot.plot(
        plot_t, binary.get_lightcurve(plot_t), label="orig", linewidth=3
    )
    mod_params = params._replace(
        w=180.0 + params.w,
        eclipse_time=params.eclipse_time + binary.eclipse_time_difference,
    )
    pyplot.plot(
        plot_t,
        Binary(from_mcmc=mod_params).get_lightcurve(plot_t),
        label="180+w",
        linewidth=3,
    )
    mod_params = params._replace(
        w=180.0 - params.w,
        eclipse_time=params.eclipse_time + binary.eclipse_time_difference,
    )
    pyplot.plot(
        plot_t,
        Binary(from_mcmc=mod_params).get_lightcurve(plot_t),
        ":",
        label="180-w",
        linewidth=3,
    )

    pyplot.legend()
    pyplot.show()
    pyplot.cla()
    pyplot.clf()

def experiment():
    """Manually experiment with things."""

    # TODO: figure out why 323020176 crashes
    test_tic = 189639080
    eb_cat = pandas.read_csv(prsa_ebs, index_col="tess_id")
    print(f"Prsa EB params for TIC {test_tic}: {eb_cat.loc[test_tic]!r}")
    logging.basicConfig(level=logging.DEBUG)

    with PdfPages("best_fit_bls.pdf") as output_pdf:
        log_likelihood = LogLikelihood(test_tic, plot_bls=output_pdf)
    log_likelihood.save_jktebob_lc(f"tess{test_tic}_jktebob.dat")

    # use("PDF")
    with PdfPages("test.pdf") as output_pdf:
        log_likelihood.plot_lc_model_comparison(
            numpy.array(
                [
                    -1.22302493e00,
                    -1.69260939e00,
                    3.04427649e-01,
                    1.07357590e00,
                    -2.37418558e00,
                    -3.13671460e00,
                    2.49650346e00,
                    8.60379866e-05,
                    -2.52210796e-03,
                    6.62451853e-01,
                    -numpy.inf,
                    2.58420045e00,
                    -numpy.inf,
                    numpy.inf,
                    numpy.inf,
                    -numpy.inf,
                    -numpy.inf,
                    -numpy.inf,
                    -numpy.inf,
                    1.13667091e00,
                    2.40478964e00,
                ]
            ),
            output_pdf,
        )


if __name__ == "__main__":
    manual_plot()
