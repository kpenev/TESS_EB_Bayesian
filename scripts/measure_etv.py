"""Define a class for measuring ETVs from TESS lightcurves."""

from functools import partial

import numpy
from numpy.lib.recfunctions import append_fields
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.stats import rv_continuous

from binary import Binary
from log_likelihood import LogLikelihood
from detrending import detrend_with_gaps, get_ooe_variability


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
            self._lcs[lc_ind] = (
                header,
                detrend_with_gaps(
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
                ),
            )

    def _get_observed_eclipses(self, eclipse_indices):
        """Return the lightcurve near selected eclipse indices."""

        observed_lc = None
        exptime = None
        if set(eclipse_indices) <= self._last_eclipse_indices:
            lc_indices = sorted(self._last_lcs_indices)
        else:
            lc_indices = range(len(self._lcs))
        self._last_eclipse_indices = set(eclipse_indices)
        self._last_lcs_indices = set()

        for lc_ind in lc_indices:
            header, lightcurve = self._lcs[lc_ind]
            included = False
            for eclipse_idx in eclipse_indices:
                eclipse_lc = lightcurve[
                    lightcurve["eclipse_flags"] == eclipse_idx
                ]
                if eclipse_lc.size == 0:
                    continue
                self._last_lcs_indices.add(lc_ind)
                included = True
                if observed_lc is None:
                    observed_lc = numpy.copy(eclipse_lc)
                else:
                    observed_lc = numpy.concatenate((observed_lc, eclipse_lc))
            if included:
                if exptime is None:
                    exptime = header["exptime"]
                else:
                    assert exptime == header["exptime"], (
                        "Attempting to combine eclipses from lightcurves with "
                        "different exposure times."
                    )
        return observed_lc, exptime

    def _get_model(self, observed_lc, time_shift, exptime, component):
        """Return a model assuming the given time shift."""

        eclipse = self._best_binaries[component].eclipse(
            observed_lc["time"] - time_shift,
            supersample_factor=100,
            exp_time=exptime,
        )
        secondary_flux_fraction = self._best_binaries[
            component
        ].secondary_flux_fraction()
        return (eclipse + secondary_flux_fraction) / (
            1.0 + secondary_flux_fraction
        )

    def __init__(self, tic_id, best_params, max_abs_etv, **kwargs):
        """
        Prepare ETV measurement using max-likelihood parameters.

        Create eclipse flags selecting primary eclipse with positive integers
        and secondary eclipses with negative.
        """

        self._best_binaries = [
            Binary(from_mcmc=best_params),
            Binary(from_mcmc=best_params),
        ]
        self._best_binaries[1].swap_components()
        self._lc_sys_err = best_params.lc_sys
        self._max_abs_etv = max_abs_etv
        self._last_eclipse_indices = set()
        self._last_lcs_indices = None
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
            'y0': [0.0],
            'dense_output': True,
            'max_step': 0.1 * max(best_fit_shift, 1/(24 * 60)),

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
