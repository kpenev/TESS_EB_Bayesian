"""Define class representing TESS EBs for analysis."""

import batman
import numpy
from scipy.optimize import minimize
from scipy.linalg import lstsq

from ebeer import EBEERBinary
#from binary_parameters import InputParams


class Binary(EBEERBinary):
    """Represent TESS eclipsing binaries."""

    def _get_primary_lightcurve(self, times, true_anomaly, exclude=()):
        """Return ratio of primary flux to that of identical isolated star."""

        flux = numpy.ones(true_anomaly.shape)
        if "ellipticity" not in exclude:
            flux += self.ellipticity(true_anomaly)
        if "reflection" not in exclude:
            flux += self.reflection(true_anomaly)
        if "beaming" not in exclude:
            flux += self.beaming(true_anomaly)
        if "eclipse" not in exclude:
            flux *= self.eclipse(times)
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
            eclipse = batman.TransitModel(self, times).light_curve(self)

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
                eclipse = batman.TransitModel(self, times).light_curve(self)
            add_primary_rhs(rhs, secondary_flux_fraction, eclipse)
            add_primary_lhs(lhs_matrix, secondary_flux_fraction, eclipse, 1)

            self.swap_components()

        return lhs_matrix, rhs, num_coef, secondary_flux_fraction

    def fit_ebeer_coefficients(
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

    def eclipse(self, times, **transit_config):
        """Return fraction of the primary flux observed due to eclipse."""

        return batman.TransitModel(self, times, **transit_config).light_curve(
            self
        )

    def fit_lightcurve(self, times, lc_to_fit, **fit_coef_kwargs):
        """
        Set the parameters of the binary to best fit the given flux modulations.

        Args
            times(array):    The times at which the flux modulations are known.

            lc_to_fit(array):    The lightcurve to fit (ratio of flux to that of
                isolated stars). Should have the same size as times.

            fit_coef_kwargs:    Any arguments to pass to
                `fit_ebeer_coefficients()`.

        Returns:
            See `fit_ebeer_coefficients()`.
        """

        def to_minimize(t0_perpass):
            """The function to minimize."""

            self.t0_perpass = t0_perpass
            result = self.fit_ebeer_coefficients(
                times,
                lc_to_fit,
                result="rms",
                **fit_coef_kwargs,
            )
            print(f"To minimize result at t={t0_perpass}: {result!r}")
            return result

        orig_result = fit_coef_kwargs.pop("result", "fluxdiff")
        best_fit = None
        for t0_guess in numpy.linspace(0, self.per, 11):
            print(f"Minimizing with t0 = {t0_guess!r}")
            fit_result = minimize(to_minimize, x0=t0_guess)
            print(f"Minimize result: {fit_result!r}")
            if best_fit is None or fit_result.fun < best_fit.fun:
                best_fit = fit_result
        self.t0_perpass = best_fit.x
        fit_coef_kwargs["result"] = orig_result
        return self.fit_ebeer_coefficients(
            times,
            lc_to_fit,
            **fit_coef_kwargs,
        )

    def get_lightcurve(self, times, secondary_flux_fraction=None, exclude=()):
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

        Returns:
            Total eBEER + eclipses flux modulation either split by star or
            combined (see ``secondary_flux_fraction`` argument).
        """

        true_anomaly = self.calc_true_anomaly(times)
        primary_flux = self._get_primary_lightcurve(
            times, true_anomaly, exclude
        )

        if secondary_flux_fraction is None:
            secondary_flux_fraction = self.secondary_flux_fraction()

        self.swap_components()

        secondary_flux = self._get_primary_lightcurve(
            times, true_anomaly, exclude
        )

        self.swap_components()

        if secondary_flux_fraction == "split":
            return (primary_flux, secondary_flux)

        return (primary_flux + secondary_flux_fraction * secondary_flux) / (
            1 + secondary_flux_fraction
        )


if __name__ == "__main__":
    print(
        Binary(
            from_mcmc=InputParams(
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
