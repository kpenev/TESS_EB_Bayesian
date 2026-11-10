"""Define class representing TESS EBs for analysis."""

import logging

import batman
import numpy
from scipy.linalg import lstsq

from ebeer import EBEERBinary

from sample_params import SampleParams


class Binary(EBEERBinary):
    """Represent TESS eclipsing binaries."""

    _logger = logging.getLogger(__name__)

    # Settings for evaluating light curves on a grid of orbital phases (the
    # ``phase_grid`` argument of `get_lightcurve()` and `eclipse()`):
    #   step: the longest spacing of the eclipse grid [days];
    #   window_search_points: how many phases per orbit the sky separation is
    #       checked at, to find where the stars can overlap;
    #   ebeer_points: how many phases per orbit the out-of-eclipse effects are
    #       evaluated at.
    default_phase_grid = {
        "step": 2.0 / 86400.0,
        "window_search_points": 20000,
        "ebeer_points": 2000,
    }

    def _get_primary_lightcurve(
        self, times, true_anomaly, exclude=(), **eclipse_config
    ):
        """Return ratio of primary flux to that of identical isolated star."""

        flux = numpy.ones(true_anomaly.shape)
        if "ellipticity" not in exclude:
            flux += self.ellipticity(true_anomaly)

        if "reflection" not in exclude:
            flux += self.reflection(true_anomaly)

        if "beaming" not in exclude:
            flux += self.beaming(true_anomaly)

        if "eclipse" not in exclude:
            flux *= self.eclipse(times, **eclipse_config)

        return flux

    def _setup_fit_beer_coef_problem(
        self,
        times,
        lc_to_fit,
        *,
        true_anomaly,
        exclude,
    ):
        """Set up the linear algebra problem for `fit_ebeer_coefficients()`."""

        def add_primary_rhs(rhs, scaling, eclipse):
            """Update the RHS vector with scaled effects of current primary."""

            if "ellipticity" in exclude:
                rhs -= scaling * eclipse
            else:
                rhs -= (
                    scaling * (1.0 + self.ellipticity(true_anomaly)) * eclipse
                )

        def add_primary_lhs(lhs_matrix, scaling, eclipse, offset):
            """Update LHS matrix with scaled effects of current primary."""

            if "reflection" not in exclude:
                lhs_matrix[:, offset] = (
                    scaling * self.reflection(true_anomaly) * eclipse
                )
            if "beaming" not in exclude:
                lhs_matrix[:, beaming_ind + offset] = (
                    scaling * self.beaming(true_anomaly) * eclipse
                )

        if true_anomaly is None:
            true_anomaly = self.calc_true_anomaly(times)

        secondary_flux_fraction = self.secondary_flux_fraction()
        if "eclipse" in exclude:
            eclipse = 1
        else:
            eclipse = self.eclipse(times)

        rhs = numpy.copy(lc_to_fit)
        if secondary_flux_fraction:
            num_components = 2
            rhs *= 1.0 + secondary_flux_fraction
        else:
            num_components = 1

        num_coef = 0
        if "beaming" not in exclude:
            self.set_beaming_coef((1.0, 1.0))
            num_coef += num_components

        if "reflection" in exclude:
            beaming_ind = 0
        else:
            self.set_reflection_coef((1.0, 1.0))
            beaming_ind = num_components
            num_coef += num_components

        lhs_matrix = numpy.empty((true_anomaly.size, num_coef))
        add_primary_rhs(rhs, 1.0, eclipse)
        add_primary_lhs(lhs_matrix, 1.0, eclipse, 0)

        if secondary_flux_fraction > 0:
            self.swap_components()
            if "eclipse" not in exclude:
                eclipse = self.eclipse(times)
            add_primary_rhs(rhs, secondary_flux_fraction, eclipse)
            add_primary_lhs(lhs_matrix, secondary_flux_fraction, eclipse, 1)

            self.swap_components()

        return lhs_matrix, rhs, num_coef, secondary_flux_fraction

    def fit_ebeer_coefficients(  # pylint: disable=too-many-arguments
        self,
        times,
        lc_to_fit,
        *,
        true_anomaly=None,
        result="lc",
        exclude=(),
    ):
        """
        Set the beaming and reflection coefficients to their best fit values.

        Args:
            times(array):    The times at which the lightcurve is known.

            lc_to_fit(array):    The relative flux (ratio of flux to that of
                isolated stars) at each entry in ``times``.

            true_anomaly(array):    For efficincy, specify the true anomaly at
                each entry in ``times``. If not specified it will be calculated.

            result(str):    What to return. Should be one of: ``'lc'``,
                ``'rms'``, ``'residues'``

            exclude(iterable):    For testing purposes it is sometimes useful to
                disable some effects. Any effect listed here is not included in
                the lightcurve. Entries should be from: ``beaming``,
                ``reflection``, ``ellipticity``, ``eclipse``.

        Returns:
            One of the following:
                array: The best fit batman + eBEER lightcurve at
                ``true_anomaly`` if ``result`` is ``'lc'``.

                float: The RMS residuals of best fit beaming and or reflection
                model.

                array: The residuals of the best fit beaming and or reflection
                model if ``result`` is neither of the above.
        """

        lhs_matrix, rhs, num_coef, secondary_flux_fraction = (
            self._setup_fit_beer_coef_problem(
                times, lc_to_fit, true_anomaly=true_anomaly, exclude=exclude
            )
        )
        num_components = 2 if secondary_flux_fraction else 1

        if num_coef == 0:
            if result == "lc":
                return lc_to_fit - rhs / (1.0 + secondary_flux_fraction)
            if result == "rms":
                return numpy.mean(rhs**2) ** 0.5
            return rhs

        fit_result = lstsq(lhs_matrix, rhs)
        if "reflection" not in exclude:
            self.set_reflection_coef(fit_result[0][:num_components])
        if "beaming" not in exclude:
            beaming_ind = 0 if "reflection" in exclude else num_components
            self.set_beaming_coef(
                fit_result[0][beaming_ind : beaming_ind + num_components]
            )
        print(
            f"Coef: {fit_result[0]!r}\nRank: {fit_result[2]}\nSingular "
            f"vals:{fit_result[3]!r}"
        )
        residuals = (lhs_matrix.dot(fit_result[0]) - rhs) / (
            1.0 + secondary_flux_fraction
        )
        if result == "lc":
            return residuals + lc_to_fit

        if result == "rms":
            result = numpy.mean(residuals**2) ** 0.5
            print(f"Returning: {result!r}")
            return result
        return residuals

    def _sky_separation(self, times):
        """
        Return the projected separation of the centers in primary radii.

        Uses batman's own routine for both halves of the orbit (in front of
        and behind the primary), so the result is the actual separation
        everywhere.
        """

        times = numpy.ascontiguousarray(times, dtype=float)
        orbit = (
            self.t0,
            self.per,
            self.a,
            self.inc * numpy.pi / 180.0,
            self.ecc,
            self.w * numpy.pi / 180.0,
        )
        # pylint: disable=protected-access
        return numpy.minimum(
            batman._rsky._rsky(times, *orbit, 1, 1),
            batman._rsky._rsky(times, *orbit, 2, 1),
        )
        # pylint: enable=protected-access

    def _eclipse_on_phase_grid(
        self, times, exp_time, phase_grid, **transit_config
    ):
        """
        Same as `eclipse()` for a fixed orbit, computed on a grid of phases.

        The orbit is periodic, so the eclipse averaged over an exposure is a
        function of phase only. It is evaluated on nodes spaced by h at whole
        multiples of h from the eclipse time, only within the windows where
        the stars can overlap (from the sky separation at
        ``phase_grid["window_search_points"]`` phases, with a margin for the
        exposure and the search spacing), averaged over each exposure by the
        trapezoid rule, and interpolated linearly to the phase of each
        exposure. h is the exposure time divided by the smallest even number of
        steps no longer than ``phase_grid["step"]`` (for a zero exposure
        time, nothing is averaged and h is ``phase_grid["step"]``). The nodes
        do not move when the windows change and the flux is 1 at the window
        edges, so the result is continuous in all parameters.
        """

        key = (exp_time, self.inverted)
        if not hasattr(self, "_eclipse_grids"):
            # pylint: disable=attribute-defined-outside-init
            self._eclipse_grids = {}
        if key not in self._eclipse_grids:
            self._eclipse_grids[key] = self._compute_eclipse_grid(
                exp_time, phase_grid, **transit_config
            )
        phase = times - self.t0
        result = numpy.ones(times.shape)
        for window_start, window_end, node_offsets, flux in self._eclipse_grids[
            key
        ]:
            folded = numpy.mod(phase - window_start, self.per) + window_start
            inside = folded <= window_end
            result[inside] = numpy.interp(folded[inside], node_offsets, flux)
        return result

    # pylint: disable-next=too-many-locals
    def _compute_eclipse_grid(self, exp_time, phase_grid, **transit_config):
        """
        Return the windows, nodes and averaged eclipse of the phase grid.

        See `_eclipse_on_phase_grid()`, which stores the result for each
        exposure time and star, so every lightcurve with the same exposure time
        reuses it. A binary's parameters must not change after the first grid
        is computed.

        Returns:
            [(float, float, array, array)]:
                For each window: its first and last offset from the eclipse
                time, the offsets of its nodes and the averaged eclipse there.
        """

        num_search = phase_grid["window_search_points"]
        search_step = self.per / num_search
        max_speed = (
            2.0
            * numpy.pi
            * self.a
            / self.per
            * numpy.sqrt((1.0 + self.ecc) / (1.0 - self.ecc))
        )
        can_overlap = self._sky_separation(
            self.t0 + search_step * numpy.arange(num_search)
        ) < 1.0 + self.rp + max_speed * (exp_time + 2.0 * search_step)
        if not can_overlap.any():
            return []

        if can_overlap.all():
            # Only possible when the margin exceeds the range of separations
            # (long exposures, very short periods): one window, whole orbit.
            windows = [(0.0, self.per)]
        else:
            # Start from a phase where the stars cannot overlap, so no window
            # wraps around the end of the orbit.
            first_outside = int(numpy.argmin(can_overlap))
            changes = numpy.flatnonzero(
                numpy.diff(numpy.roll(can_overlap, -first_outside).astype(int))
            )
            ends = list(changes[1::2])
            if len(ends) < len(changes[0::2]):
                ends.append(num_search - 1)
            # One search step of margin beyond the first and last points.
            windows = [
                (
                    (start + first_outside) * search_step,
                    (end + first_outside + 1) * search_step,
                )
                for start, end in zip(changes[0::2], ends)
            ]

        if exp_time > 0:
            num_steps = 2 * int(
                numpy.ceil(exp_time / (2.0 * phase_grid["step"]))
            )
            node_step = exp_time / num_steps
        else:
            num_steps = 0
            node_step = phase_grid["step"]
        half = num_steps // 2
        result = []
        for window_start, window_end in windows:
            node_offsets = node_step * numpy.arange(
                int(numpy.floor(window_start / node_step)) - half,
                int(numpy.ceil(window_end / node_step)) + half + 1,
            )
            flux = batman.TransitModel(
                self,
                numpy.ascontiguousarray(self.t0 + node_offsets),
                **transit_config,
            ).light_curve(self)
            if num_steps:
                cumulative = numpy.concatenate(([0.0], numpy.cumsum(flux)))
                flux = (
                    cumulative[num_steps + 1 :]
                    - cumulative[: -num_steps - 1]
                    - 0.5 * (flux[num_steps:] + flux[:-num_steps])
                ) / num_steps
                node_offsets = node_offsets[half:-half]
            result.append((window_start, window_end, node_offsets, flux))
        return result

    def eclipse(
        self,
        times,
        *,
        exp_time=0.0,
        supersample_step=5.0 / 86400.0,
        extrapolate=True,
        phase_grid=None,
        **transit_config,
    ):
        """
        Return fraction of the primary flux observed due to eclipse.

        With ``phase_grid`` (settings like `default_phase_grid`) and a fixed
        orbit, the eclipse is computed on a grid of phases and interpolated
        (see `_eclipse_on_phase_grid()`). Otherwise, only exposures during
        which the stars can overlap are passed to batman;
        the flux is exactly 1 for the rest. An exposure can overlap if the
        projected separation of the centers at its midpoint (in primary radii)
        is less than 1 + R2/R1 plus the largest change of separation possible
        during a full exposure (twice what is needed).

        The error of averaging over an exposure with equally spaced points is
        nearly proportional to the step (the light curve has kinks at the
        contact points), so by default it is removed to first order by
        Richardson extrapolation: with the step at most ``supersample_step``,
        the flux is 2 * flux(step) - flux(2 * step). On TIC 101845679 the
        default 5 s step changes the log-likelihood by at most 0.25 relative to
        plain 0.1 s steps (vs 1.5 for the previous 100 steps per exposure) at
        a fraction of the cost.

        Args:
            times(float or array):    The (mid-exposure) times [days].

            exp_time(float):    The exposure time [days].

            supersample_step(float or None):    The longest time between
                supersampling points within an exposure [days]. None disables
                supersampling (and extrapolation).

            extrapolate(bool):    Whether to apply the Richardson extrapolation.

            phase_grid(dict or None):    Compute on a grid of phases with these
                settings (see `default_phase_grid`), or exactly if None.
                Evolving orbits are always computed exactly.

            transit_config:    Passed on to the batman initializer.
        """

        if not self._orbit_is_fixed:
            self._fix_evolving_orbit(times)
        times = numpy.asarray(times, dtype=float)
        flat_times = times.ravel()
        if phase_grid is not None and not self._evolving_orbit:
            return self._eclipse_on_phase_grid(
                flat_times, exp_time, phase_grid, **transit_config
            ).reshape(times.shape)
        # Largest relative orbital speed (at periastron), in primary radii per
        # day, times the exposure time.
        max_separation_change = (
            2.0
            * numpy.pi
            * self.a
            / self.per
            * numpy.sqrt((1.0 + self.ecc) / (1.0 - self.ecc))
            * exp_time
        )
        can_overlap = (
            self._sky_separation(flat_times)
            < 1.0 + self.rp + max_separation_change
        )
        result = numpy.ones(flat_times.shape)
        if not can_overlap.any():
            return result.reshape(times.shape)

        def get_flux(supersample_factor):
            """Return batman's flux for the overlapping exposures."""

            return batman.TransitModel(
                self,
                numpy.ascontiguousarray(flat_times[can_overlap]),
                exp_time=exp_time,
                supersample_factor=supersample_factor,
                **transit_config,
            ).light_curve(self)

        if supersample_step is None or exp_time <= 0:
            result[can_overlap] = get_flux(1)
        elif extrapolate:
            # Exactly twice as many points for the finer of the two fluxes.
            coarse_factor = max(
                1, int(numpy.ceil(exp_time / (2.0 * supersample_step)))
            )
            result[can_overlap] = 2.0 * get_flux(2 * coarse_factor) - get_flux(
                coarse_factor
            )
        else:
            result[can_overlap] = get_flux(
                max(1, int(numpy.ceil(exp_time / supersample_step)))
            )
        return result.reshape(times.shape)

    def get_lightcurve(
        self,
        times,
        secondary_flux_fraction=None,
        exclude=(),
        phase_grid=None,
        **eclipse_config,
    ):
        """
        Return the flux modulation at the given times for current binary.

        Args:
            times(float or numpy array):    time(s) in DAYS which the model will
                be computed for

            secondary_flux_fraction(float, None, or 'split'):    Allows
                overwriting the estimate of the ratio of the flux comes from
                the secondary to that of the primary. If the special value
                ``'split'`` is passed, the function return a 2-tuple of the
                calculated flux modulations (float or numpy array) from the
                primary and secondary respectively. Otherwise return is a single
                float or numpy array which is the combined flux modulation due
                to both stars. If left to the default value of ``None``, the
                ratio is calculated from the radii and effective temperatures of
                the two stars currently set.

            exclude(iterable):    For testing purposes it is sometimes useful to
                disable some effects. Any effect listed here is not included in
                the lightcurve. Entries should be from: ``beaming``,
                ``reflection``, ``ellipticity``, ``eclipse``.
                Effects are enabled or disabled for both stars simultaneously.

            phase_grid(dict or None):    Compute on a grid of orbital phases
                with these settings (see `default_phase_grid`) and interpolate,
                or exactly if None. Evolving orbits are always computed
                exactly. On the grid, the out-of-eclipse effects of each star
                are evaluated at ``phase_grid["ebeer_points"]`` phases and the
                eclipses as described in `_eclipse_on_phase_grid()`.

            eclip_config:    Extra parameters to pass to the BATMAN initializer
                for eclipse modeling.

        Returns:
            Total eBEER + eclipses flux modulation either split by star or
            combined (see ``secondary_flux_fraction`` argument).
        """

        on_grid = phase_grid is not None and not self._evolving_orbit
        true_anomaly = phase = None
        if on_grid:
            times = numpy.asarray(times, dtype=float)
            phase = numpy.mod(times - self.t0, self.per)
            with_ebeer = any(
                effect not in exclude
                for effect in ("ellipticity", "reflection", "beaming")
            )
            # The out-of-eclipse effects are evaluated at evenly spaced phases
            # from the eclipse time, once per binary and star, and reused for
            # every lightcurve (like the eclipse grids).
            if with_ebeer and not hasattr(self, "_ebeer_grids"):
                ebeer_times = self.t0 + self.per * numpy.linspace(
                    0.0, 1.0, phase_grid["ebeer_points"] + 1
                )
                # pylint: disable=attribute-defined-outside-init
                self._ebeer_grids = {
                    "times": ebeer_times,
                    "phases": ebeer_times - self.t0,
                    "true_anomaly": self.calc_true_anomaly(ebeer_times),
                }
                # pylint: enable=attribute-defined-outside-init
        else:
            true_anomaly = self.calc_true_anomaly(times)
        self._orbit_is_fixed = True

        def get_star_flux():
            """Return the light curve of the current primary star."""

            if not on_grid:
                return self._get_primary_lightcurve(
                    times, true_anomaly, exclude, **eclipse_config
                )
            flux = numpy.ones(times.shape)
            if with_ebeer:
                key = (self.inverted, tuple(sorted(exclude)))
                if key not in self._ebeer_grids:
                    self._ebeer_grids[key] = self._get_primary_lightcurve(
                        self._ebeer_grids["times"],
                        self._ebeer_grids["true_anomaly"],
                        tuple(exclude) + ("eclipse",),
                    )
                flux *= numpy.interp(
                    phase, self._ebeer_grids["phases"], self._ebeer_grids[key]
                )
            if "eclipse" not in exclude:
                flux *= self.eclipse(
                    times, phase_grid=phase_grid, **eclipse_config
                )
            return flux

        primary_flux = get_star_flux()

        if secondary_flux_fraction is None:
            secondary_flux_fraction = self.secondary_flux_fraction()

        self.swap_components()

        secondary_flux = get_star_flux()

        self.swap_components()
        self._orbit_is_fixed = False

        if secondary_flux_fraction == "split":
            return (primary_flux, secondary_flux)

        result = (primary_flux + secondary_flux_fraction * secondary_flux) / (
            1 + secondary_flux_fraction
        )
        return result


if __name__ == "__main__":
    print(
        Binary(
            from_mcmc=SampleParams(
                mtotal=numpy.pi / 2,
                mratio=0.5,
                age_gyr=4.6,
                feh=0.0,
                per=3.0,
                esinw=0.3,
                ecosw=0.3,
                inc=90.0,
                perpass_phase=0.23,
                primary_limb_dark_1=0.6,
                primary_limb_dark_2=0.3,
                secondary_limb_dark_1=0.6,
                secondary_limb_dark_2=0.3,
                primary_prot=3.0,
                secondary_prot=3.0,
                primary_reflection_coef=0.1,
                secondary_reflection_coef=0.1,
                primary_beaming_coef=0.1,
                secondary_beaming_coef=0.1,
            )
        )
    )
