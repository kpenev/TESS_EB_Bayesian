"""Define tools for ETVs analysis from TESS lightcurves."""

from functools import partial
from multiprocessing import Pool
import pickle
import logging
import os

import numpy
from numpy.lib.recfunctions import append_fields
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.stats import rv_continuous

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)

from binary import Binary
from log_likelihood import LogLikelihood
from detrending import (
    detrend_with_gaps,
    fit_ooe_model,
    get_lc_gap_indices,
)

from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams


_worker_measure_etv = None
_logger = logging.getLogger(__name__)


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
        assert numpy.isfinite(
            result
        ).all(), f"Non-finite CDF for x={x!r}: {result!r}"
        return result

    def _pdf(self, x):  # pylint: disable=arguments-differ
        """The probability density at the given time shift."""

        return self._likelihood(x) / self._normalization

    def to_dict(self):
        """Return a picklable dict of the distribution's state."""

        return {
            "mode": self.mode,
            "a": self.a,
            "b": self.b,
            "above_integral": self._above_integral,
            "below_integral": self._below_integral,
        }

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

        if "name" not in kwargs:
            kwargs["name"] = "ETV distribution"
        super().__init__(*args, **kwargs)
        print(f"Range: {self.a}, {self.b}")
        self.mode = mode
        self._likelihood = likelihood
        self._above_integral = above_integral
        self._below_integral = below_integral
        self._min_below = float(below_integral(self.a))
        print(f"Min below: {self._min_below}")
        self._normalization = float(above_integral(self.b)) - self._min_below
        print(f"Normalization: {self._normalization}")


class MeasureETV(LogLikelihood):
    """Class for fitting the shift in timing of eclipses."""

    _default_config = {
        "no_detrend": False,
        "eclipse_tweak_order": None,
        "max_abs_etv": None,
    }

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
            max_etv_phase = self._config["max_abs_etv"] / best_binary.per
            start_phase, duration = self._get_eclipse_range(best_binary)
            start_phase -= pad_duration * duration + max_etv_phase
            duration += 2 * pad_duration * duration + 2 * max_etv_phase
            near_eclipse_ranges.append((start_phase, duration))

        self._eclipse_durations[0] = near_eclipse_ranges[0][1]
        self._eclipse_durations[1] = near_eclipse_ranges[1][1]
        self._eclipse_durations *= self._best_binaries[0].per

        return near_eclipse_ranges

    def _flag_eclipses(self, near_eclipse_ranges):
        """Add flags to the LCs selecting only and all points near eclipses."""

        porb = self._best_binaries[0].per

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
                shifted_time = lightcurve["time"] - start_phase * porb
                near_eclipses = ((shifted_time % porb) / porb) < duration
                period_ind = numpy.floor(shifted_time / porb).astype(int)
                lightcurve["eclipse_flags"][near_eclipses] = (
                    sign * period_ind[near_eclipses]
                )
            self._lcs[lc_ind] = (header, lightcurve)

    def _get_ooe_model(self, best_binary, lightcurve, exp_time):
        """Return a best-fit scaled model of the OOE variability of the LC."""

        ooe_mask = lightcurve["eclipse_flags"] == 0
        model = best_binary.get_lightcurve(
            lightcurve["time"],
            supersample_factor=100,
            exp_time=exp_time,
            exclude=["eclipse"],
        )
        flux = lightcurve["flux"][ooe_mask]
        lc_sq_errors = (
            lightcurve["flux_err"][ooe_mask] ** 2 + self._lc_sys_err**2
        )
        ooe_model = model[ooe_mask]
        model *= (ooe_model * flux / lc_sq_errors).sum() / (
            ooe_model**2 / lc_sq_errors
        ).sum()
        return model, ooe_mask

    def _select_spline_nodes(self, lc_times, near_eclipse_ranges, node_spacing):
        """Return node times for OOE spline fitting, avoiding eclipses.

        Selects node locations within the times of a lightcurve segment such
        that:
          - No node falls within a primary or secondary eclipse (as defined by
            near_eclipse_ranges).
          - At least one node is placed in the OOE region between each
            consecutive pair of eclipses.
          - If the OOE data span within a region exceeds ``node_spacing``,
            additional evenly-spaced nodes are added, keeping the spacing no
            smaller than ``node_spacing``.

        Args:
            lc_times:    The  times of a single contiguous lightcurve segment.

            near_eclipse_ranges:    List of (start_phase, duration) tuples
                as returned by _find_near_eclipse_ranges, one per eclipse
                type (primary then secondary).

            node_spacing(float):    Minimum spacing between nodes in days.

        Returns:
            numpy.ndarray:  Sorted 1-D array of node times (days).
        """

        t_min = lc_times.min()
        t_max = lc_times.max()

        eclipse_intervals = self._get_eclipse_intervals(
            t_min, t_max, near_eclipse_ranges
        )

        # OOE regions: before first eclipse, between consecutive pairs,
        # and after the last eclipse
        if eclipse_intervals:
            ooe_bounds = (
                [(t_min, eclipse_intervals[0][0])]
                + [
                    (eclipse_intervals[i][1], eclipse_intervals[i + 1][0])
                    for i in range(len(eclipse_intervals) - 1)
                ]
                + [(eclipse_intervals[-1][1], t_max)]
            )
        else:
            return None

        nodes = []
        for region_start, region_end in ooe_bounds:
            region_times = lc_times[
                (lc_times >= region_start) & (lc_times <= region_end)
            ]
            if region_times.size == 0:
                continue
            t_lo = region_times.min()
            t_hi = region_times.max()
            num_nodes = int((t_hi - t_lo) / node_spacing) + 2
            nodes.extend(numpy.linspace(t_lo, t_hi, num_nodes))

        return numpy.unique(nodes)

    def _prepare_lightcurves(self, *_, **kwargs):
        """Remove the OOE variability of the lightcurve and flag eclipses."""

        near_eclipse_ranges = self._find_near_eclipse_ranges(
            kwargs.get("pad_duration", 0.0)
        )
        self._flag_eclipses(near_eclipse_ranges)
        max_eclipse_duration = self._eclipse_durations.max()
        if self._config["no_detrend"]:
            return
        for lc_ind, (header, lightcurve) in enumerate(self._lcs):
            original = numpy.copy(lightcurve["flux"])
            model, ooe_mask = self._get_ooe_model(
                self._best_binaries[0], lightcurve, header["exptime"]
            )
            lightcurve["flux"] /= model
            detrended, _, good_mask = detrend_with_gaps(
                lightcurve,
                mask=ooe_mask,
                get_trend=fit_ooe_model,
                eclipse_rejection=numpy.inf,
                spline_nodes=partial(
                    self._select_spline_nodes,
                    near_eclipse_ranges=near_eclipse_ranges,
                    node_spacing=max(
                        2.0 * max_eclipse_duration,
                        0.1 * self._best_binaries[0].per,
                    ),
                ),
                min_gap=0.5,
                full_output=True,
            )
            detrended = append_fields(
                detrended,
                ["model", "demodeled"],
                [model[good_mask], detrended["original"]],
                usemask=False,
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
                eclipse_lc = eclipse_lc[
                    numpy.logical_and(
                        numpy.isfinite(eclipse_lc["flux"]),
                        numpy.isfinite(eclipse_lc["flux_err"]),
                    )
                ]
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

    def _get_eclipse_intervals(self, t_min, t_max, near_eclipse_ranges):
        """Return sorted, non-overlapping eclipse time intervals in a segment.

        Args:
            t_min(float):           Start of the lightcurve segment (days).

            t_max(float):           End of the lightcurve segment (days).

            near_eclipse_ranges:    List of (start_phase, duration) tuples
                as returned by _find_near_eclipse_ranges, one per eclipse
                type (primary then secondary).

        Returns:
            List of [start, end] pairs (days) for each eclipse occurrence
            that overlaps [t_min, t_max], sorted by start time.
        """

        porb = self._best_binaries[0].per

        # Period-k eclipse of type (start_phase, duration) covers:
        #   [(k + start_phase) * porb,  (k + start_phase + duration) * porb)
        eclipse_intervals = []
        for start_phase, duration in near_eclipse_ranges:
            for k in range(
                int(numpy.floor(t_min / porb - start_phase - duration)),
                int(numpy.ceil(t_max / porb - start_phase)) + 2,
            ):
                eclipse_start = (k + start_phase) * porb
                eclipse_end = (k + start_phase + duration) * porb
                if eclipse_end > t_min and eclipse_start < t_max:
                    eclipse_intervals.append(
                        [max(eclipse_start, t_min), min(eclipse_end, t_max)]
                    )

        eclipse_intervals.sort()

        previous_end = -numpy.inf
        for start, end in eclipse_intervals:
            assert start > previous_end, "Near-eclipse ranges overlap!"
            previous_end = end

        return eclipse_intervals

    def _eclipse_covered_single(self, eclipse_idx, lightcurve):
        """Return True iff the given eclipse is well covered in the LC."""

        eclipse_lc = lightcurve[lightcurve["eclipse_flags"] == eclipse_idx]
        start = eclipse_lc["time"].min()
        end = eclipse_lc["time"].max()
        component_idx = 0 if eclipse_lc[0]["eclipse_flags"] > 0 else 1
        min_points = eclipse_lc.size / 4
        time_frac = (eclipse_lc["time"] - start) / self._eclipse_durations[
            component_idx
        ]
        # print(
        #    f"Eclipse {eclipse_idx}: duration = "
        #    f"{self._eclipse_durations[component_idx]}, start = {start}, "
        #    f"end = {end}, Np={eclipse_lc.size}. In first 1/3: "
        #    f"{(time_frac < 1 / 3).sum()}. In second 1/3: "
        #    f"{numpy.logical_and(time_frac < 1 / 3, time_frac < 2 / 3).sum()}."
        #    f" In last 1/3: {(time_frac > 2 / 3).sum()}"
        # )

        return (
            (end - start > 2 * self._eclipse_durations[component_idx] / 3)
            and ((time_frac < 1 / 3).sum() > min_points)
            and (
                numpy.logical_and(time_frac < 1 / 3, time_frac < 2 / 3).sum()
                > min_points
            )
            and ((time_frac > 2 / 3).sum() > min_points)
        )

    def _calc_eclipse_sum_sq_residuals(  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self, observed_lc, model_lc, lc_sq_errors, t0, eclipse_idx
    ):
        """Calc the contribution of a single eclipse to _`sum_sq_residuals()`"""

        mask = observed_lc["eclipse_flags"] == eclipse_idx
        flux = observed_lc["flux"][mask]
        sq_errors = lc_sq_errors[mask]
        eclipse_model = model_lc[mask]
        if self._config["eclipse_tweak_order"] is None:
            scaled_model = eclipse_model
        else:
            dt = observed_lc["time"][mask] - t0
            tweak_terms = numpy.column_stack(
                [
                    eclipse_model * dt**k
                    for k in range(self._config["eclipse_tweak_order"] + 1)
                ]
            )
            inv_sigma = 1.0 / numpy.sqrt(sq_errors)
            coeffs, _, _, _ = numpy.linalg.lstsq(
                tweak_terms * inv_sigma[:, None],
                flux * inv_sigma,
                rcond=None,
            )
            scaled_model = tweak_terms @ coeffs
        return ((flux - scaled_model) ** 2 / sq_errors).sum()

    def __init__(
        self,
        tic_id,
        samples_fname,
        **config,
    ):
        """
        Prepare ETV measurement using max-likelihood parameters.

        Create eclipse flags selecting primary eclipse with positive integers
        and secondary eclipses with negative.
        """

        if "pad_duration" not in config:
            config["pad_duration"] = 0.0
        best_params = self._get_best_params(samples_fname.format(tic_id=tic_id))
        self._lc_sys_err = best_params.lc_sys
        self._config = {
            k: config.pop(k) if k in config else default
            for k, default in self._default_config.items()
        }
        if self._config["max_abs_etv"] is None:
            self._config["max_abs_etv"] = min(0.1, 0.05 * best_params.per)

        self._best_binaries = [
            Binary(from_mcmc=best_params),
            Binary(from_mcmc=best_params),
        ]
        self._best_binaries[1].swap_components()

        self._eclipse_indices = set()
        self._observed_lc = None
        self._exptime = None
        self._eclipse_durations = numpy.array([numpy.nan, numpy.nan])
        super().__init__(tic_id, **config)

    def sum_sq_residuals(self, time_shift, eclipse_indices):
        """Sum-square residuals for selected eclipses given a time shift.

        Before computing residuals, finds the best-fit polynomial in time
        (measured from the eclipse center) by which to scale model_lc,
        using weighted least squares. The polynomial order is controlled by
        self._config['eclipse_tweak_order'].
        """

        eclipse_indices = numpy.atleast_1d(eclipse_indices)
        assert (eclipse_indices > 0).all() or (eclipse_indices < 0).all(), (
            "Combining primary and secondary eclipses not supported by "
            "MeasureETV"
        )
        component = 0 if eclipse_indices[0] > 0 else 1
        observed_lc, exptime = self._get_observed_eclipses(eclipse_indices)
        model_lc = self._get_model(observed_lc, time_shift, exptime, component)

        lc_sq_errors = observed_lc["flux_err"] ** 2 + self._lc_sys_err**2
        t0 = self._best_binaries[component].t0
        porb = self._best_binaries[0].per

        total_ssr = 0.0
        for eclipse_idx in eclipse_indices:
            total_ssr += self._calc_eclipse_sum_sq_residuals(
                observed_lc,
                model_lc,
                lc_sq_errors,
                (t0 + numpy.abs(eclipse_idx) * porb),
                eclipse_idx,
            )
        return total_ssr

    def likelihood(self, time_shift, eclipse_indices, min_sum_sq_residuals=0.0):
        """Return the unnormalized likelihood (not log) of given timeshit."""

        if abs(time_shift) > self._config["max_abs_etv"]:
            return 0.0
        return numpy.exp(
            min_sum_sq_residuals
            - self.sum_sq_residuals(time_shift, eclipse_indices)
        )

    def fit_timeshift(self, eclipse_indices):
        """Find the best-fit common time shift for the given eclipses."""

        try:
            return minimize_scalar(
                self.sum_sq_residuals,
                (
                    -self._config["max_abs_etv"],
                    0.0,
                    self._config["max_abs_etv"],
                ),
                args=(eclipse_indices,),
            )
        except ValueError:
            ssr = [
                self.sum_sq_residuals(offset, eclipse_indices)
                for offset in [
                    -self._config["max_abs_etv"],
                    0,
                    self._config["max_abs_etv"],
                ]
            ]
            print(
                "Minimizing sum squared residuals for eclispe indices "
                f"{eclipse_indices} failed, starting from "
                f"SSR({-self._config['max_abs_etv']}) = {ssr[0]},"
                f" SSR(0) = {ssr[1]}, "
                f"SSR({self._config['max_abs_etv']}) = {ssr[2]}"
            )
            raise

    def get_timeshift_distro(self, eclipse_indices):
        """Calculate the CDF of the time shift for selected eclipses."""

        shift_fit_result = self.fit_timeshift(eclipse_indices)
        best_fit_shift = shift_fit_result.x
        solve_kwargs = {
            "y0": [0.0],
            "dense_output": True,
            "max_step": 0.1 * max(best_fit_shift, 1 / (24 * 60)),
            "jac": numpy.array([[0.0]]),
        }
        print(f"Integrating with options: {solve_kwargs!r}")
        above_integral = solve_ivp(
            lambda x, y: [
                self.likelihood(x, eclipse_indices, shift_fit_result.fun)
            ],
            (best_fit_shift, self._config["max_abs_etv"]),
            **solve_kwargs,
        )
        below_integral = solve_ivp(
            lambda x, y: [
                self.likelihood(x, eclipse_indices, shift_fit_result.fun)
            ],
            (best_fit_shift, -self._config["max_abs_etv"]),
            **solve_kwargs,
        )
        print("Constructing distribution")
        return TimeShiftDistribution(
            mode=best_fit_shift,
            likelihood=numpy.vectorize(
                partial(
                    self.likelihood,
                    eclipse_indices=eclipse_indices,
                    min_sum_sq_residuals=shift_fit_result.fun,
                )
            ),
            above_integral=above_integral.sol,
            below_integral=below_integral.sol,
            a=-self._config["max_abs_etv"],
            b=self._config["max_abs_etv"],
        )

    def get_mean_time(self, eclipse_indices):
        """Return the mean of the times of all points for selected eclipses."""

        return numpy.mean(
            self._get_observed_eclipses(eclipse_indices)[0]["time"]
        )

    def get_eclipse_indices(  # pylint: disable=too-many-locals
        self, min_gap, allow_partial=False
    ):
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
            eclipse_covered = numpy.vectorize(
                partial(self._eclipse_covered_single, lightcurve=lightcurve)
            )
            start_index = 0
            sector_primaries = []
            sector_secondaries = []
            for end_index in get_lc_gap_indices(lightcurve["time"], min_gap):
                print(
                    "Getting eclipse indices in LC segment "
                    f"{start_index}:{end_index}"
                )
                segment = lightcurve["eclipse_flags"][start_index:end_index]
                indices = numpy.unique(segment)
                print(f"All indices: {indices}")
                split = numpy.searchsorted(indices, 0)
                primaries = indices[split + 1 :]
                secondaries = indices[:split]
                if not allow_partial:
                    if segment[0] < 0:
                        print(f"Dropping {segment[0]} from secondaries")
                        assert secondaries[-1] == segment[0]
                        secondaries = secondaries[:-1]
                        print(f"Clean secondaries: {secondaries}")
                    elif segment[0] > 0:
                        print(f"Dropping {segment[0]} from primaries")
                        assert primaries[0] == segment[0]
                        primaries = primaries[1:]
                    if segment[-1] < 0:
                        print(f"Dropping {segment[-1]} from secondaries")
                        assert secondaries[0] == segment[-1]
                        secondaries = secondaries[1:]
                        print(f"Clean secondaries: {secondaries}")
                    elif segment[-1] > 0:
                        print(f"Dropping {segment[-1]} from primaries")
                        assert primaries[-1] == segment[-1]
                        primaries = primaries[:-1]

                if primaries.size:
                    primaries = primaries[eclipse_covered(primaries)]
                if primaries.size:
                    sector_primaries.append(primaries)

                if secondaries.size:
                    secondaries = secondaries[eclipse_covered(secondaries)]
                if secondaries.size:
                    sector_secondaries.append(secondaries)
                start_index = end_index
            result[0].append((header["sector"], sector_primaries))
            result[1].append((header["sector"], sector_secondaries))
        return result


def _get_distro_data(eclipse_indices):
    """Compute time shift distribution and return serializable data."""

    try:
        shift_fit_result = _worker_measure_etv.fit_timeshift(eclipse_indices)
        distro_dict = _worker_measure_etv.get_timeshift_distro(
            eclipse_indices
        ).to_dict()
        distro_dict["min_sum_sq_residuals"] = shift_fit_result.fun
        return distro_dict
    except Exception as err:
        _logger.critical(
            "Failed to construct ETV distribution for TIC %d, "
            "eclipse_indices %s: %s",
            _worker_measure_etv.tic_id,
            repr(eclipse_indices),
            err,
        )
        raise


def compute_and_save_etv_distros(
    measure_etv, tasks, output_fname, config, **extra_pickle
):
    """
    Compute ETV time shift distributions in parallel and save to file.

    Uses fork-based multiprocessing so workers inherit the already-initialized
    MeasureETV instance without repeating expensive setup. Each worker
    computes get_timeshift_distro for one task and returns a serializable
    dict via TimeShiftDistribution.to_dict().

    Args:
        measure_etv:    Initialized MeasureETV instance.

        tasks:    List of eclipse-index arrays, one per distribution.

        num_parallel:    How many parallel processes to use.

        output_fname:    Path to write the pickle output file.

        extra_pickle:    Additional data to add to the pickle
    """

    global _worker_measure_etv  # pylint: disable=global-statement
    _worker_measure_etv = measure_etv
    config = vars(config)
    config.update(
        {
            "task": "compute_and_save_etv_distros",
            "tic_id": measure_etv.tic_id,
            "parent_pid": os.getpid(),
        }
    )
    with Pool(
        processes=config["num_parallel"],
        initializer=setup_process_map,
        initargs=[config],
    ) as pool:
        distro_data = pool.map(_get_distro_data, tasks)
    for task, distro_dict in zip(tasks, distro_data):
        distro_dict["eclipse_indices"] = task
    extra_pickle["tic_id"] = measure_etv.tic_id
    extra_pickle["distributions"] = distro_data
    with open(output_fname, "wb") as out_file:
        pickle.dump(extra_pickle, out_file)


def load_etv_distros(
    output_fname, restore_likelihood=True, **measure_etv_kwargs
):
    """
    Load ETV distributions previously saved by compute_and_save_distros.

    Args:
        output_fname:           Path to the pickle file written by
                                compute_and_save_distros.
        restore_likelihood:     If True (default), create a MeasureETV instance
                                from the saved tic_id, samples_fname_pattern,
                                and max_abs_etv so that the PDF is available on
                                the recovered distributions.
        **measure_etv_kwargs:   Extra keyword arguments forwarded to MeasureETV
                                when restore_likelihood is True.

    Returns:
        A dict with keys:
            ``num_primary_tasks``:  Number of leading entries in
                                    ``distributions`` that correspond to
                                    primary eclipses.
            ``distributions``:      List of dicts, each containing
                                    ``eclipse_indices`` (numpy array) and a
                                    ``TimeShiftDistribution`` under the key
                                    ``distro``.
    """

    with open(output_fname, "rb") as in_file:
        data = pickle.load(in_file)
    measure_etv = None
    if restore_likelihood:
        measure_etv = MeasureETV(
            tic_id=data["tic_id"],
            samples_fname=data["samples_fname_pattern"],
            max_abs_etv=data["max_abs_etv"],
            **measure_etv_kwargs,
        )
    non_init_keys = {"eclipse_indices", "min_sum_sq_residuals"}
    for entry in data["distributions"]:
        likelihood = None
        if measure_etv is not None:
            likelihood = numpy.vectorize(
                partial(
                    measure_etv.likelihood,
                    eclipse_indices=entry["eclipse_indices"],
                    min_sum_sq_residuals=entry["min_sum_sq_residuals"],
                )
            )
        distro_params = {
            k: v for k, v in entry.items() if k not in non_init_keys
        }
        entry["distro"] = TimeShiftDistribution(
            **distro_params, likelihood=likelihood
        )
    return data
