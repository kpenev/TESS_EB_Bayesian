"""Define the log-likelihood function to use for MCMC."""  # pylint: disable=too-many-lines

import logging
from functools import partial

import numpy
from numpy.lib.recfunctions import append_fields
from scipy.stats import norm, truncnorm
from sqlalchemy import select, delete

from tess_target import TESSTarget, get_bls_eclipse_mask
from detrending import (
    masked_detrend,
    get_moving_median,
    get_ooe_variability,
    detrend_with_gaps,
)
from extinction_correction import Green19Correction
from binary import Binary
from cache_interface import CacheSession, CachedSED, CachedBLS
from sample_params import SampleParams
from exclude_data import exclude_data
from catalog_interface import get_eb_info


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
    _precession_scale = 0.2

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
                    (cached_sed.bad_sed_threshold, cached_sed.bad_sed_penalty),
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
                            none_to_nan(getattr(cached_bls, column)),
                            none_to_nan(
                                getattr(cached_bls, column + "_uncertainty")
                            ),
                        )
                        if (column + "_uncertainty" in bls_columns)
                        else none_to_nan(getattr(cached_bls, column))
                    )
                    for column in bls_columns
                    if not column.endswith("_uncertainty")
                }
                cls._logger.debug("Loaded cached BLS: %s", repr(bls))
        return sed, bls

    def _cache(self, tic_id):
        """Add the SED and best fit BLS to cache (overwriting if necessary)."""

        # False positive
        # pylint: disable=no-member
        with CacheSession.begin() as cache_session:
            # pylint: enable=no-member
            cache_session.execute(delete(CachedSED).filter_by(tic_id=tic_id))
            cache_session.execute(delete(CachedBLS).filter_by(tic_id=tic_id))

            self._logger.debug("Caching SED: %s", repr(self._sed))
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
                    bad_sed_threshold=self._sed[2][0],
                    bad_sed_penalty=self._sed[2][1],
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

    def _get_folded_eclipse_mask(self, porb):
        """
        Return mask of near eclispe points in the combined all sector folded LC.
        """

        eclipse_mask = numpy.array([], dtype=bool)
        phase = numpy.array([], dtype=float)
        for header, lightcurve in self._lcs:
            detrended, detrend_mask = masked_detrend(
                lightcurve,
                header["exptime"],
                self,
                get_ooe_variability,
                return_mask=True,
            )
            rms = numpy.sqrt(
                numpy.mean((detrended["flux"] - 1)[detrend_mask] ** 2)
            )
            eclipse_mask = numpy.concatenate(
                (eclipse_mask, detrended["flux"] < 1 - 3.0 * rms)
            )
            phase = numpy.concatenate(
                (phase, (detrended["time"] % porb) / porb)
            )
        phase_sorter = numpy.argsort(phase)
        return phase[phase_sorter], eclipse_mask[phase_sorter]

    def _get_near_eclipse_ranges(
        self, porb, eclipse_fraction, bin_edges, ooe_factor
    ):
        """Return list of start times and near-eclipse durations."""

        if self.masked_is_significant():
            eclipse_phases = numpy.array(
                [
                    self._best_fit_bls["transit_time"],
                    self._best_fit_bls["masked_transit_time"],
                ]
            )
        else:
            eclipse_phases = numpy.array(
                [
                    self._best_fit_bls["transit_time"],
                    self._best_fit_bls["transit_time"]
                    + self._best_fit_bls["period"][0],
                ]
            )
        eclipse_phases = (eclipse_phases % porb) / porb

        print(
            "Up crossings"
            + repr(
                numpy.nonzero(
                    numpy.logical_and(
                        eclipse_fraction[:-1] < 0.5, eclipse_fraction[1:] >= 0.5
                    )
                )
            )
        )
        down_crossings = numpy.sort(
            bin_edges[
                numpy.nonzero(
                    numpy.logical_and(
                        eclipse_fraction[:-1] > 0.5, eclipse_fraction[1:] <= 0.5
                    )
                )[0]
                + 1
            ]
        )
        up_crossings = numpy.sort(
            bin_edges[
                numpy.nonzero(
                    numpy.logical_and(
                        eclipse_fraction[:-1] < 0.5, eclipse_fraction[1:] >= 0.5
                    )
                )[0]
            ]
        )

        result = []
        for phase in eclipse_phases:
            print(f"Searching place for phase: {phase} among {up_crossings!r}")
            print(f"Start ind: {numpy.searchsorted(up_crossings, phase)}")
            if up_crossings.size:
                start = numpy.searchsorted(up_crossings, phase)
                start = up_crossings[start - 1]
            else:
                start = 0
            end = numpy.searchsorted(down_crossings, phase, side="right")
            if end == down_crossings.size:
                end = 0
            else:
                end = down_crossings[end]
            if start >= end:
                duration = end + 1 - start
            else:
                duration = end - start
            start -= ooe_factor * duration
            result.append((start, duration * (1 + 2 * ooe_factor)))

        print(f"Near eclipse ranges: {result}")
        return result

    def _get_eclipse_fraction(self, porb):
        """Return the fraction of points near eclipse by phase."""

        phase, eclipse_mask = self._get_folded_eclipse_mask(porb)
        nbins = int(porb // min(header["exptime"] for header, _ in self._lcs))
        eclipse_fraction, bin_edges = numpy.histogram(
            phase[eclipse_mask], bins=nbins, range=(0, 1)
        )
        return (
            eclipse_fraction
            / numpy.histogram(phase, bins=nbins, range=(0, 1))[0]
        ), bin_edges

    def _add_eclipse_flags(self, ooe_factor):
        """
        Add flags to the LCs selecting only and all points near eclipses.

        Handles variability (real or instrumental) in the lightcurve using
        `get_ooe_variability()` detrending.
        """

        porb = self._best_fit_bls["period"][0]
        if not self.masked_is_significant():
            porb *= 2
        eclipse_fraction, bin_edges = self._get_eclipse_fraction(porb)
        for lc_ind, (header, lightcurve) in enumerate(self._lcs):
            lightcurve = append_fields(
                lightcurve,
                "eclipse_flags",
                numpy.zeros(lightcurve.size, dtype=int),
            )
            header["near_eclipse_ranges"] = self._get_near_eclipse_ranges(
                porb, eclipse_fraction, bin_edges, ooe_factor
            )
            header["bls_period"] = porb

            for (start_phase, duration), sign in zip(
                header["near_eclipse_ranges"],
                [1, -1],
            ):
                shifted_time = lightcurve["time"] - start_phase * porb
                near_eclipes = shifted_time % porb / porb < duration
                period_ind = numpy.floor(shifted_time / porb).astype(int)

                lightcurve["eclipse_flags"][near_eclipes] = (
                    sign * period_ind[near_eclipes]
                )
            self._lcs[lc_ind] = (header, lightcurve)

    def _prepare_lightcurves(self, tic_exclude, save_detrending):
        """Prepare the lightcurves for sampling."""

        detrend_kwargs = {
            "full_output": save_detrending,
        }
        cat_info = get_eb_info(self._tic_id)

        if "OOE" in tic_exclude or "BLSOOE" in tic_exclude:
            self.get_model = self.get_eclipse_model
            detrend_kwargs["get_trend"] = get_ooe_variability
            detrend_kwargs["spline_rejection"] = 5.0
            detrend_kwargs["eclipse_rejection"] = 2.0
            detrend_kwargs["half_porb"] = cat_info["period"] / 2
        else:
            self.get_model = self.get_full_model
            detrend_kwargs["get_trend"] = get_moving_median

        overwrite_cache = False
        original_lcs = self._lcs
        self._lcs = []
        while (not self._lcs) or (self._best_fit_bls is None):
            for header, lightcurve in original_lcs:
                if self._best_fit_bls is None:
                    self._logger.debug("Detrending without BLS information")
                    detrend = partial(
                        detrend_with_gaps,
                        min_gap=max(0.5, 30.0 * header["exptime"]),
                        **detrend_kwargs,
                    )
                else:
                    self._logger.debug("Detrending with BLS information")
                    detrend = partial(
                        masked_detrend,
                        log_likelihood=self,
                        get_trend=get_moving_median,
                        exptime=header["exptime"],
                        full_output=save_detrending,
                    )
                self._lcs.append((header, detrend(lightcurve)))
            if self._best_fit_bls is None:
                self._best_fit_bls = self.fit_bls(cat_info=cat_info)
                self._lcs = []
                overwrite_cache = True

        self._add_eclipse_flags(0.4)

        return overwrite_cache

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

    def check_logscale(self, param):
        """Return True iff the given parameter is sampled in log-scale."""

        return param in self._log_uniform

    def __init__(
        self,
        tic_id,
        *,
        overwrite_cache=(),
        ignore_extinction_flags=True,
        save_detrending=False,
    ):
        """Prepare to evaluate the log-likelihood for the given TIC ID."""

        super().__init__(tic_id)
        overwrite_cache = [item.upper() for item in overwrite_cache]

        # https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0-DataProductOverview-Table:CadenceQualityFlags

        self._sed, self._best_fit_bls = self.get_cached_sed_and_bls(tic_id)

        if self._sed is None or "SED" in overwrite_cache:
            self._sed = Green19Correction(
                ignore_extinction_flags
            ).get_absolute_magnitudes(tic_id)[0] + ((10.0, 1.0),)
        if "BLS" in overwrite_cache:
            self._best_fit_bls = None

        overwrite_cache = (
            self._prepare_lightcurves(
                exclude_data.get(tic_id, []), save_detrending
            )
            or overwrite_cache
        )

        if overwrite_cache:
            self._cache(tic_id)

        self._range = SampleParams(
            mtotal=(0.21, 50),
            mratio=(0.01, 2),
            age_gyr=(-3, 1.1),
            meh=Binary.meh_range,
            per=(
                self.bls_porb - 10.0 * self._best_fit_bls["period"][1],
                self.bls_porb + 10.0 * self._best_fit_bls["period"][1],
            ),
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

        self._logger.debug("LCs: %s", repr(self._lcs))
        self._logger.debug("SED: %s", repr(self._sed))

    def masked_is_significant(self):
        """
        Return True iff the masked BLS fit appears to fit real eclipses.

        To be marked as significant all of the following must be satisfied:

          * maked period should be close to unmasked period (within 5 unmasked
            uncertanties)

          * depth should exceed its uncertanity by at least a factor of 5

          * masked_harmonic_delta_log_likelihood < -5
        """

        return self.masked_bls_is_significant(self._best_fit_bls)

    @staticmethod
    def get_full_model(binary, header, lightcurve, lc_sys_err):
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
        return model_lc, lc_sq_errors, None

    def get_binary_near_eclipse_flags(self, lc_time, binary, duration_factor):
        """Return flags selecting only and all points near eclipses."""

        eval_ntimes = int(10 * binary.per / (lc_time[1:] - lc_time[:-1]).min())
        if self.masked_is_significant():
            bls_times = [
                self._best_fit_bls["transit_time"],
                self._best_fit_bls["masked_transit_time"],
            ]
        else:
            bls_times = [
                self._best_fit_bls["transit_time"],
                self._best_fit_bls["transit_time"]
                + self._best_fit_bls["period"][0],
            ]
        eval_time = numpy.concatenate(
            (numpy.linspace(0, binary.per, eval_ntimes), bls_times)
        )
        in_eclipse = binary.eclipse(eval_time) < 1
        if not in_eclipse[-2:].any():
            raise ValueError(
                f"BLS eclipse centers ({bls_times[-2:]!r}) not covered by "
                f"eclipse mask: {in_eclipse[-2:]!r}."
            )

        in_eclipse = in_eclipse[:-2]  # Remove the BLS check points
        if in_eclipse[0]:
            if in_eclipse.all():
                eclipse_start = eclipse_end = 0.0
            else:
                eclipse_start, eclipse_end = eval_time[
                    numpy.nonzero(numpy.logical_not(in_eclipse))[0][[-1, 0]]
                ]
            eclipse_duration = eclipse_end - eclipse_start + binary.per
        else:
            if not in_eclipse.any():
                eclipse_start = eclipse_end = 0.0
            else:
                eclipse_start, eclipse_end = numpy.nonzero(in_eclipse)[0][
                    [0, -1]
                ]
                eclipse_start -= 1
                eclipse_end += 1
                assert not in_eclipse[eclipse_start]
                assert not in_eclipse[eclipse_end]
                eclipse_start = eval_time[eclipse_start]
                eclipse_end = eval_time[eclipse_end]
            eclipse_duration = eclipse_end - eclipse_start

        mask_expand = eclipse_duration * (duration_factor - 1) / 2

        flag_time = lc_time - eclipse_start + mask_expand

        flags = (flag_time // binary.per).astype(int)
        flags -= flags.min()
        flags[flag_time % binary.per > eclipse_duration + 2 * mask_expand] = -1
        return flags

    def _evaluate_eclipse_model(
        self, binary, header, lightcurve, require_coverage=True
    ):
        """Evaluate the model near eclipses and list eclipse indices present."""

        eclipse_indices = numpy.unique(lightcurve["eclipse_flags"])

        bls_times = [
            (start + 0.5 * duration) * header["bls_period"]
            for start, duration in header["near_eclipse_ranges"]
        ]
        eclipse_model = binary.get_lightcurve(
            numpy.concatenate((lightcurve["time"], bls_times)),
            exclude=["beaming", "reflection", "ellipticity"],
            supersample_factor=100,
            exp_time=header["exptime"],
        )
        if require_coverage and (
            eclipse_model[-1] == 1 or eclipse_model[-2] == 1
        ):
            raise ValueError(
                f"BLS eclipse centers ({bls_times[-2:]!r}) not covered by "
                f"model eclipse: {eclipse_model[-2:]!r}."
            )
        return (eclipse_model[:-2], eclipse_indices)  # Remove BLS check points

    def get_eclipse_model(self, binary, header, lightcurve, lc_sys_err):
        """
        Same as `get_model()` but ignoring OOE variability.

        This method allows handling cases where there is significant non-binary
        related variability in the lightcurve (astrophysical or instrumental).

        A second order polynomial is fit to the out-of-eclipse points near each
        eclipse to estimate the background flux.
        """

        model_mask = lightcurve["eclipse_flags"] != 0
        if model_mask.sum() < 10:
            return (
                numpy.array([]),
                numpy.array([]),
                numpy.zeros(model_mask.size, dtype=bool),
            )
        lightcurve = lightcurve[model_mask]

        lc_sq_errors = lightcurve["flux_err"] ** 2 + lc_sys_err**2

        eclipse_model, eclipse_order = self._evaluate_eclipse_model(
            binary, header, lightcurve, False
        )

        for eclipse_ind in eclipse_order:
            if eclipse_ind == 0:
                continue

            eclipse_mask = lightcurve["eclipse_flags"] == eclipse_ind
            fit_mask = numpy.logical_and(
                eclipse_mask, numpy.abs(eclipse_model - 1) < 1e-10
            )
            if eclipse_mask.sum() < 20:
                LogLikelihood._logger.debug(
                    "Discarding eclipse %d", eclipse_ind
                )
                dont_discard = numpy.logical_not(eclipse_mask)
                model_mask[model_mask] = numpy.logical_and(
                    model_mask[model_mask], dont_discard
                )
                eclipse_model = eclipse_model[dont_discard]
                lc_sq_errors = lc_sq_errors[dont_discard]
                lightcurve = lightcurve[dont_discard]
                continue
            if fit_mask.sum() < 10 or not fit_mask[eclipse_mask][[0, -1]].all():
                LogLikelihood._logger.debug(
                    "For eclipse %d fitting OOE with eclipse.", eclipse_ind
                )
                ooe_poly = numpy.poly1d(
                    numpy.polyfit(
                        lightcurve["time"][eclipse_mask],
                        lightcurve["flux"][eclipse_mask]
                        / eclipse_model[eclipse_mask],
                        2,
                        w=1 / lc_sq_errors[eclipse_mask] ** 0.5,
                    )
                )
            else:
                LogLikelihood._logger.debug("Keeping eclipse %d", eclipse_ind)

                ooe_poly = numpy.poly1d(
                    numpy.polyfit(
                        lightcurve["time"][fit_mask],
                        lightcurve["flux"][fit_mask],
                        2,
                        w=1 / lc_sq_errors[fit_mask] ** 0.5,
                    )
                )
            eclipse_model[eclipse_mask] *= ooe_poly(
                lightcurve["time"][eclipse_mask]
            )

        return eclipse_model, lc_sq_errors, model_mask

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
        if isinstance(value, tuple):
            assert param == "w"
            assert len(value) == 2
            value, precession_rate = value
        else:
            precession_rate = None
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
            if value <= 0:
                return param_ind, -numpy.inf
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
            result = -numpy.inf
        if value > high:
            result = numpy.inf
        result = norm.ppf((value - low) / (high - low))
        if precession_rate is not None:
            return (param_ind, len(SampleParams._fields)), (
                result,
                precession_rate / self._precession_scale,
            )
        return param_ind, result

    def get_sample_params(self, mcmc_sample):
        """Return prior-transformed parameters given sample."""

        result = SampleParams(
            *[
                self.prior_transform(param, sample_entry)
                for param, sample_entry in zip(
                    SampleParams._fields, mcmc_sample
                )
            ]
        )
        if mcmc_sample.size == len(SampleParams._fields) + 1:
            result = result._replace(
                w=(result.w, self._precession_scale * mcmc_sample[-1])
            )
        return result

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

    def calc_lc_log_likelihood(  # pylint: disable=too-many-branches
        self, binary, lc_sys_err, bls_eclipse_only=(), return_residuals=False
    ):
        """
        Return log-likelihood of observing the TESS LCs for given binary.

        Args:
            binary(Binary):    The binary for which to evaluate the model to
                compare to the lightcurves.

            lc_sys_err(float):    The systematic error to assume for
                lightcurves. (added in quadrature to formal errors).

            bls_eclipse_only(str or False):    If not False, only lightcurve
                points near eclipse (per BLS) are considered:

                * even: Only the points in the vicinity of even eclipses (per
                  best fit BLS) are included.

                * odd: Only the points in the vicinity of even eclipses (per
                  best fit BLS) are included.

                * both: Points near both eclipses.

                * masked: Points near the eclipses reported in the masked BLS
                  are considered

            return_residuals(bool):    If True, return the residuals of between
                the model and LC (e.g. for use in least squares fit). WARNING:
                the correction term for ``lc_sys_err`` is not included in the
                residuals.
        """

        if binary.a < 1 + binary.rp or binary.out_of_range:
            self._logger.warning(
                "Unphysical binary parameters (assuming zero likelihood): %s",
                binary,
            )
            return -numpy.inf
        result = numpy.array([]) if return_residuals else 0.0
        for header, lightcurve in self._lcs:
            if lightcurve["good"].sum() < 10:
                continue
            lightcurve = lightcurve[lightcurve["good"]]
            if bls_eclipse_only:
                lightcurve = lightcurve[
                    get_bls_eclipse_mask(
                        self._best_fit_bls, lightcurve, bls_eclipse_only
                    )
                ]
            if lightcurve.size < 10:
                continue
            try:
                model_lc, lc_sq_errors, mask = self.get_model(
                    binary, header, lightcurve, lc_sys_err
                )
            except ValueError as error:
                self._logger.warning(
                    "Error while calculating model for TIC %d (assuming zero "
                    "likelihood): %s",
                    self.tic_id,
                    error.args[0],
                )
                return -numpy.inf
            if mask is None:
                observed_lc = lightcurve["flux"]
            elif mask.sum() < 10:
                continue
            else:
                observed_lc = lightcurve["flux"][mask]
            if return_residuals:
                result = numpy.concatenate(
                    (result, (observed_lc - model_lc) / lc_sq_errors**0.5)
                )
                if lc_sys_err > 0:
                    result = numpy.concatenate(
                        (
                            result,
                            numpy.sqrt(
                                numpy.log(
                                    lc_sq_errors
                                    / (lc_sq_errors - lc_sys_err**2)
                                )
                            ),
                        )
                    )
            else:
                result -= (
                    (observed_lc - model_lc) ** 2 / lc_sq_errors
                    + numpy.log(lc_sq_errors)
                ).sum()
            self._logger.debug("Log likelihood now: %s", repr(result / 2))

        if not numpy.isfinite(result).all():
            self._logger.error(
                "Non-finite log-likelihood for binary: %s", binary
            )
        if not return_residuals:
            result /= 2
        return result

    def calc_sed_log_likelihood(
        self, binary, sed_sys_err, return_residuals=False
    ):
        """Return log-likelihood of observed SED for given binary."""

        self._logger.debug("SED: %s", repr(self._sed))
        finite = numpy.isfinite(self._sed[0])
        self._logger.debug("Finite SED flags: %s", repr(finite))
        sed_errors = (self._sed[1][finite] ** 2 + sed_sys_err**2) ** 0.5
        result = (self._sed[0][finite] - binary.absmag[finite]) / sed_errors
        self._logger.debug("SED residuals: %s", repr(result))
        if self._sed[2][1]:
            nsigma = numpy.abs(
                self._sed[0][finite] - binary.absmag[finite]
            ) / numpy.maximum(self._sed[1][finite], 0.01)
            bad_sed = nsigma > self._sed[2][0]
            result[bad_sed] *= 10.0 ** (
                self._sed[2][1] * (nsigma[bad_sed] - self._sed[2][0]) ** 0.5
            )
            result[numpy.isinf(result)] = numpy.finfo(result.dtype).max
        self._logger.debug("After penalty, SED residuals: %s", repr(result))

        if return_residuals:
            if sed_sys_err > 0:
                result = numpy.concatenate(
                    (
                        result,
                        numpy.sqrt(
                            numpy.log(sed_errors**2 / self._sed[1][finite] ** 2)
                        ),
                    )
                )
            return result
        result = max(
            numpy.finfo(result.dtype).min,
            -(result**2 + numpy.log(sed_errors**2)).sum() / 2,
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

    @staticmethod
    def get_blob(sample_params):
        """Return blob to save in the emcee file for given sample parameters."""

        if isinstance(sample_params.w, float):
            return sample_params

        return sample_params._replace(
            w=sample_params.w[0],
        ) + (sample_params.w[1],)

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
            return (-numpy.inf,) + self.get_blob(sample_params)
        if binary.out_of_range:
            self._logger.warning(
                "Out of range parameters:\n\t%s",
                "\n\t".join(binary.out_of_range),
            )
            return (-numpy.inf,) + self.get_blob(sample_params)
        self._logger.debug("Binary: %s", binary)

        result = (
            self.calc_prior_loglikelihood(
                mcmc_sample[numpy.logical_not(exclude_priors)]
            )
            + self.calc_lc_log_likelihood(binary, sample_params.lc_sys)
            + self.calc_sed_log_likelihood(binary, sample_params.sed_sys)
        )

        blob = self.get_blob(sample_params)
        self._logger.debug(
            "Final log likelihood(%s): %s, blob size: %d, blob types: %s",
            repr(mcmc_sample),
            repr(result),
            len(self.get_blob(sample_params)),
            set(map(type, blob)),
        )
        result = (result,) + blob
        self._logger.debug(
            "Final result (size %d): %s", len(result), repr(result)
        )
        return result
