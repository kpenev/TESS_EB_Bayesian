"""Define a class for measuring ETVs from TESS lightcurves."""

from functools import partial

import numpy
from numpy.lib.recfunctions import append_fields
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.stats import rv_continuous

from binary import Binary
from log_likelihood import LogLikelihood
from detrending import (
    detrend_with_gaps,
    get_ooe_variability,
    get_lc_gap_indices,
)
import paths
from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams


class TimeShiftDistribution(  # pylint:disable=too-many-instance-attributes
    rv_continuous
):
    """Distribution of time shifts for a given set of eclipses."""

    def _cdf(self, x):  # pylint: disable=arguments-differ
        """The cumulative distribution up to the given time shift."""

        x = numpy.atleast_1d(x)
        result = numpy.empty_like(x)
        result[x < self.a] = 0.0
        result[x > self.b] = 1.0
        selection = (x >= self.a) & (x <= self.mode)
        if selection.any():
            result[selection] = (
                self._below_integral(x[selection])[0] - self._min_below
            ) / self._normalization
        selection = (x > self.mode) & (x <= self.b)
        if selection.any():
            result[selection] = (
                self._above_integral(x[selection])[0] - self._min_below
            ) / self._normalization
        return result

    def _pdf(self, x):  # pylint: disable=arguments-differ
        """The probability density at the given time shift."""

        return self._likelihood(x) / self._normalization

    def __init__(
        self,
        *args,
        mode,
        likelihood,
        above_integral,
        below_integral,
        **kwargs,
    ):
        """Define from likellihood and integrals above/below best fit O-C."""

        super().__init__(*args, **kwargs)
        self.mode = mode
        self._likelihood = likelihood
        self._above_integral = above_integral
        self._below_integral = below_integral
        self._min_below = float(below_integral(self.a))
        self._normalization = float(above_integral(self.b)) - self._min_below


class MeasureETV(LogLikelihood):
    """Class for fitting the shift in timing of eclipses."""

    def _get_eclipse_range(self, best_binary):
        """Return the range (start, duration) of primary & secondary eclipse.

        Determines eclipse ranges from the best-fit binary model lightcurve.
        Any point where the model flux (excluding beaming, reflection, and
        ellipticity) is below 1 is counted as being in eclipse.
        """

        porb = best_binary.per
        t0 = best_binary.t0

        n_points = 10000
        time_grid = numpy.linspace(
            t0 - porb / 2, t0 + porb / 2, n_points, endpoint=False
        )
        model_flux = best_binary.eclipse(time_grid)

        transitions = numpy.diff((model_flux < 1.0).astype(int))
        start_idx = numpy.where(transitions == 1)[0] + 1
        end_idx = numpy.where(transitions == -1)[0] + 1

        assert start_idx.size == 1
        assert end_idx.size == 1
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        dt = porb / n_points
        t_start = time_grid[start_idx]
        t_end = time_grid[end_idx - 1] + dt
        assert t_start <= t0 < t_end
        duration = (t_end - t_start) / porb
        start_phase = (t_start % porb) / porb
        return start_phase, duration

    def _find_near_eclipse_ranges(self, pad_duration):
        """Return the start phase & duration near primary/secondary eclipses."""

        near_eclipse_ranges = []
        for best_binary in self._best_binaries:
            max_etv_phase = self._max_abs_etv / best_binary.per
            start_phase, duration = self._get_eclipse_range(best_binary)
            start_phase -= pad_duration * duration + max_etv_phase
            duration += 2 * pad_duration * duration + 2 * max_etv_phase
            near_eclipse_ranges.append((start_phase, duration))
        return near_eclipse_ranges

    def _add_eclipse_flags(self, pad_duration):
        """Add flags to the LCs selecting only and all points near eclipses."""

        porb = self._best_binaries[0].per
        near_eclipse_ranges = self._find_near_eclipse_ranges(pad_duration)
        max_duration = 0.0

        for lc_ind, (header, lightcurve) in enumerate(self._lcs):
            lightcurve = append_fields(
                lightcurve,
                "eclipse_flags",
                numpy.zeros(lightcurve.size, dtype=int),
            )
            header["near_eclipse_ranges"] = near_eclipse_ranges
            header["porb"] = porb

            for (start_phase, duration), sign in zip(
                near_eclipse_ranges, [1, -1]
            ):
                max_duration = max(max_duration, duration)
                shifted_time = lightcurve["time"] - start_phase * porb
                near_eclipses = shifted_time % porb / porb < duration
                period_ind = numpy.floor(shifted_time / porb).astype(int)
                lightcurve["eclipse_flags"][near_eclipses] = (
                    sign * period_ind[near_eclipses]
                )
            self._lcs[lc_ind] = (header, lightcurve)
        return max_duration

    def _prepare_lightcurves(self, *_, **kwargs):
        """Remove the OOE variability of the lightcurve and flag eclipses."""

        max_eclipse_duration = self._add_eclipse_flags(
            kwargs.get("pad_duration", 0.0)
        )
        for lc_ind, (header, lightcurve) in enumerate(self._lcs):
            original = numpy.copy(lightcurve["flux"])
            lightcurve["flux"] /= self._best_binaries[0].get_lightcurve(
                lightcurve["time"],
                supersample_factor=100,
                exp_time=header["exptime"],
                exclude=["eclipse"],
            )
            detrended, _, good_mask = detrend_with_gaps(
                lightcurve,
                mask=lightcurve["eclipse_flags"] == 0,
                get_trend=get_ooe_variability,
                eclipse_rejection=numpy.inf,
                approx_node_spacing=max(
                    10.0 * max_eclipse_duration,
                    0.1 * self._best_binaries[0].per,
                ),
                min_gap=0.5,
                full_output=True,
                return_mask=True,
            )
            detrended["original"] = original[good_mask]

            self._lcs[lc_ind] = (header, detrended)

    def _get_observed_eclipses(self, eclipse_indices):
        """Return the lightcurve near selected eclipse indices."""

        eclipse_indices = set(eclipse_indices)
        if eclipse_indices == self._eclipse_indices:
            return self._observed_lc, self._exptime
        if eclipse_indices < self._eclipse_indices:
            keep = numpy.zeros(self._observed_lc.shape, dtype=bool)
            for eclipse_idx in eclipse_indices:
                keep = numpy.logical_or(
                    keep, self._observed_lc["eclipse_flags"] == eclipse_idx
                )
            self._observed_lc = self._observed_lc[keep]
            self._eclipse_indices = eclipse_indices
            return self._observed_lc, self._exptime

        self._eclipse_indices = set(eclipse_indices)

        self._observed_lc = None
        self._exptime = None

        for header, lightcurve in self._lcs:
            included = False
            for eclipse_idx in eclipse_indices:
                eclipse_lc = lightcurve[
                    lightcurve["eclipse_flags"] == eclipse_idx
                ]
                eclipse_lc = eclipse_lc[numpy.isfinite(eclipse_lc["flux"])]
                if eclipse_lc.size == 0:
                    continue
                included = True
                if self._observed_lc is None:
                    self._observed_lc = numpy.copy(eclipse_lc)
                else:
                    self._observed_lc = numpy.concatenate(
                        (self._observed_lc, eclipse_lc)
                    )
            if included:
                if self._exptime is None:
                    self._exptime = header["exptime"]
                else:
                    assert self._exptime == header["exptime"], (
                        "Attempting to combine eclipses from lightcurves with "
                        "different exposure times."
                    )
        assert self._observed_lc is not None
        return self._observed_lc, self._exptime

    def _get_model(self, observed_lc, time_shift, exptime, component):
        """Return a model assuming the given time shift."""

        eclipse = self._best_binaries[component].eclipse(
            observed_lc["time"] + time_shift,
            supersample_factor=100,
            exp_time=exptime,
        )
        secondary_flux_fraction = self._best_binaries[
            component
        ].secondary_flux_fraction()
        return (eclipse + secondary_flux_fraction) / (
            1.0 + secondary_flux_fraction
        )

    @staticmethod
    def _get_best_params(samples_fname):
        """Return the maximum likelihood parameters in given samples file."""

        backend = HDFBackend(
            samples_fname,
            name="mcmc",
            read_only=True,
        )
        log_prob = backend.get_log_prob()
        top_index = numpy.unravel_index(numpy.argmax(log_prob), log_prob.shape)
        return SampleParams(
            *backend.get_blobs(discard=0, thin=top_index[0] + 1)[0][
                top_index[1:]
            ]
        )

    def __init__(
        self, tic_id, max_abs_etv=None, samples_fname=paths.samples, **kwargs
    ):
        """
        Prepare ETV measurement using max-likelihood parameters.

        Create eclipse flags selecting primary eclipse with positive integers
        and secondary eclipses with negative.
        """

        if "pad_duration" not in kwargs:
            kwargs["pad_duration"] = 0.0
        best_params = self._get_best_params(samples_fname.format(tic_id=tic_id))
        if max_abs_etv is None:
            max_abs_etv = min(0.1, 0.05 * best_params.per)

        self._best_binaries = [
            Binary(from_mcmc=best_params),
            Binary(from_mcmc=best_params),
        ]
        self._best_binaries[1].swap_components()
        self._lc_sys_err = best_params.lc_sys
        self._max_abs_etv = max_abs_etv
        self._eclipse_indices = set()
        self._observed_lc = None
        self._exptime = None
        super().__init__(tic_id, **kwargs)

    def sum_sq_residuals(self, time_shift, eclipse_indices):
        """Sum-square residuals for selected eclipses given a time shift."""

        eclipse_indices = numpy.atleast_1d(eclipse_indices)
        assert (eclipse_indices > 0).all() or (eclipse_indices < 0).all(), (
            "Combining primary and secondary eclipses not supported by "
            "MeasureETV"
        )
        observed_lc, exptime = self._get_observed_eclipses(eclipse_indices)
        model_lc = self._get_model(
            observed_lc, time_shift, exptime, 0 if eclipse_indices[0] > 0 else 1
        )

        lc_sq_errors = observed_lc["flux_err"] ** 2 + self._lc_sys_err**2
        return ((observed_lc["flux"] - model_lc) ** 2 / lc_sq_errors).sum()

    def likelihood(self, time_shift, eclipse_indices):
        """Return the unnormalized likelihood (not log) of given timeshit."""

        if abs(time_shift) > self._max_abs_etv:
            return 0.0
        return numpy.exp(-self.sum_sq_residuals(time_shift, eclipse_indices))

    def fit_timeshift(self, eclipse_indices):
        """Find the best-fit common time shift for the given eclipses."""

        return minimize_scalar(
            self.sum_sq_residuals,
            (
                -self._max_abs_etv,
                0.0,
                self._max_abs_etv,
            ),
            args=(eclipse_indices,),
        )

    def get_timeshift_distro(self, eclipse_indices):
        """Calculate the CDF of the time shift for selected eclipses."""

        best_fit_shift = self.fit_timeshift(eclipse_indices).x
        solve_kwargs = {
            "y0": [0.0],
            "dense_output": True,
            "max_step": 0.1 * max(best_fit_shift, 1 / (24 * 60)),
        }
        print(f"Integrating with options: {solve_kwargs!r}")
        above_integral = solve_ivp(
            lambda x, y: [self.likelihood(x, eclipse_indices)],
            (best_fit_shift, self._max_abs_etv),
            **solve_kwargs,
        )
        below_integral = solve_ivp(
            lambda x, y: [self.likelihood(x, eclipse_indices)],
            (best_fit_shift, -self._max_abs_etv),
            **solve_kwargs,
        )
        print("Constructing distribution")
        return TimeShiftDistribution(
            name="ETV distribution",
            mode=best_fit_shift,
            likelihood=numpy.vectorize(
                partial(self.likelihood, eclipse_indices=eclipse_indices)
            ),
            above_integral=above_integral.sol,
            below_integral=below_integral.sol,
            a=-self._max_abs_etv,
            b=self._max_abs_etv,
        )

    def get_eclipse_indices(self, min_gap, allow_partial=False):
        """
        Return the eclipse indices split by sector by lightcurve segment.

        Args:
            min_gap(float):    The minimum gap, in days, which triggers a
                splitting of the lightcurve.

            allow_partial(bool):    If False (default), indices of eclipses whic
                touch the ends of a lightcurve segment are not included.

        Returns:
            Two lists one for primary eclipses and one for secondary eclipses.
            Each list is formatted as: ``[(sector1, [(piece 1 indices), (piece 2
            indices), ...]), (sector2, ...]''
        """

        result = [], []
        for header, lightcurve in self._lcs:
            start_index = 0
            sector_primaries = []
            sector_secondaries = []
            for end_index in get_lc_gap_indices(lightcurve["time"], min_gap):
                segment = lightcurve["eclipse_flags"][start_index:end_index]
                indices = numpy.unique(segment)
                split = numpy.searchsorted(indices, 0)
                primaries = indices[split + 1 :]
                secondaries = indices[:split]
                if not allow_partial:
                    if segment[0] < 0:
                        assert secondaries[-1] == segment[0]
                        secondaries = secondaries[:-1]
                    elif segment[0] > 0:
                        assert primaries[0] == segment[0]
                        primaries = primaries[1:]
                sector_primaries.append(primaries)
                sector_secondaries.append(secondaries)
                start_index = end_index
            result[0].append((header["sector"], sector_primaries))
            result[1].append((header["sector"], sector_secondaries))
        return result
