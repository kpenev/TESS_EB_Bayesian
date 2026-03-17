"""Class which visualizes the inner workings of MeasureETV."""

import logging

import numpy
from matplotlib import pyplot
from measure_etv import MeasureETV


class MeasureETVVisualize(MeasureETV):
    """Wrap methods of MeasureETV with plots showing their functioning."""

    stop_plotting = False

    def _add_eclipse_flags(self, *args, **kwargs):
        """Show color-coded plot of the eclipse flags."""

        result = super()._add_eclipse_flags(*args, **kwargs)
        if self.stop_plotting:
            return result

        figure, axes = pyplot.subplots(
            len(self._lcs),
            1,
            figsize=(12, 3 * len(self._lcs)),
            squeeze=False,
        )
        for ax, (header, lightcurve) in zip(axes[:, 0], self._lcs):
            ax.set_title(f"Sector {header['sector']} {header['provenance']}")

            out_of_eclipse = lightcurve["eclipse_flags"] == 0
            ax.plot(
                lightcurve["time"][out_of_eclipse],
                lightcurve["flux"][out_of_eclipse],
                ".",
                color="black",
                markersize=1,
            )

            ax.set_prop_cycle(None)
            for eclipse_idx in numpy.unique(lightcurve["eclipse_flags"]):
                if eclipse_idx == 0:
                    continue
                mask = lightcurve["eclipse_flags"] == eclipse_idx
                ax.plot(
                    lightcurve["time"][mask],
                    lightcurve["flux"][mask],
                    ".",
                    markersize=4,
                    label=str(eclipse_idx),
                )

        figure.suptitle("Eclipse flags")
        pyplot.tight_layout()
        pyplot.show()
        return result

    def _prepare_lightcurves(self, *args, **kwargs):
        """Show the removal of OOE variability from lightcurves."""

        result = super()._prepare_lightcurves(  # pylint: disable=assignment-from-no-return
            *args, **kwargs
        )
        if self.stop_plotting:
            return result

        figure, axes = pyplot.subplots(
            len(self._lcs),
            1,
            figsize=(12, 3 * len(self._lcs)),
            squeeze=False,
        )
        for ax, (header, lightcurve) in zip(axes[:, 0], self._lcs):
            ax.set_title(f"Sector {header['sector']} {header['provenance']}")
            ax.plot(
                lightcurve["time"],
                lightcurve["original"],
                ".",
                color="gray",
                markersize=4,
                label="original",
            )
            ax.plot(
                lightcurve["time"],
                lightcurve["model"],
                "-",
                color="cyan",
                linewidth=3,
                label="model",
            )
            ax.plot(
                lightcurve["time"],
                lightcurve["demodeled"],
                ".",
                color="green",
                linewidth=3,
                label="de-modeled",
            )
            ax.plot(
                lightcurve["time"],
                lightcurve["trend"],
                "-",
                color="red",
                linewidth=3,
                label="trend",
            )
            ax.plot(
                lightcurve["time"],
                lightcurve["flux"],
                ".",
                color="blue",
                markersize=4,
                label="detrended",
            )
            ax.axhline(y=1.0, color="black", linewidth=5)
            ax.legend()

        figure.suptitle("OOE variability removal")
        pyplot.tight_layout()
        pyplot.show()

        return result

    def _get_model(self, observed_lc, time_shift, *args, **kwargs):
        """Show the observed and model lightcurves at given time shift."""

        model_flux = super()._get_model(
            observed_lc, time_shift, *args, **kwargs
        )
        if self.stop_plotting:
            return model_flux

        eclipse_indices = numpy.unique(observed_lc["eclipse_flags"])
        eclipse_indices = eclipse_indices[eclipse_indices != 0]

        figure, axes = pyplot.subplots(
            len(eclipse_indices),
            1,
            figsize=(12, 3 * max(1, len(eclipse_indices))),
            squeeze=False,
        )
        for ax, eclipse_idx in zip(axes[:, 0], eclipse_indices):
            ax.set_title(f"Eclipse {eclipse_idx}")
            mask = observed_lc["eclipse_flags"] == eclipse_idx
            ax.plot(
                observed_lc["time"][mask],
                observed_lc["flux"][mask],
                ".",
                color="black",
                markersize=4,
                label="observed",
            )
            ax.plot(
                observed_lc["time"][mask],
                model_flux[mask],
                "-",
                color="red",
                label="model",
            )
            ax.legend()

        figure.suptitle(f"time shift = {time_shift:.6f}")
        pyplot.tight_layout()
        pyplot.show()

        return model_flux

    def get_timeshift_distro(self, eclipse_indices):
        """Show PDF and CDF of timeshifts from MCMC samples."""

        stop_plotting = self.stop_plotting
        self.stop_plotting = True
        timeshift_distro = super().get_timeshift_distro(eclipse_indices)
        if stop_plotting:
            self.stop_plotting = True
            return timeshift_distro

        figure, axes = pyplot.subplots(2, 1, figsize=(12, 12), squeeze=True)
        plot_x = numpy.linspace(
            timeshift_distro.ppf(1e-5),
            timeshift_distro.isf(1e-5),
            max(
                1000,
                int(
                    10.0
                    * (timeshift_distro.b - timeshift_distro.a)
                    / max(timeshift_distro.mode, 1 / (24 * 60))
                ),
            ),
        )
        pyplot.sca(axes[0])
        pyplot.plot(plot_x, timeshift_distro.pdf(plot_x), color="black")
        p_value = timeshift_distro.cdf(0.0)
        if p_value > 1e-6:
            pyplot.axvline(x=0.0)
            pyplot.axhline(y=timeshift_distro.pdf(0.0))
        pyplot.axvline(x=timeshift_distro.mode, color='red')
        pyplot.xlabel("O-C")
        pyplot.ylabel("PDF")
        pyplot.sca(axes[1])
        pyplot.plot(plot_x, timeshift_distro.cdf(plot_x), color="black")
        if p_value > 1e-6:
            pyplot.axvline(x=0.0)
            pyplot.axhline(y=p_value)
        pyplot.axvline(x=timeshift_distro.mode, color='red')
        pyplot.xlabel("O-C")
        pyplot.ylabel("CDF")
        figure.suptitle(
            f"Timeshift distribution for eclipses {eclipse_indices}, p-value: "
            f"{p_value}"
        )
        pyplot.show()

        self.stop_plotting = stop_plotting

        return timeshift_distro


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tic_id = 1045298

    measure_etv = MeasureETVVisualize(
        tic_id,
        samples_fname="/mnt/md2/TESS_EBs/first1000/tess{tic_id}_samples.h5",
    )

    primary_eclipse_indices, secondary_eclipse_indices = (
        measure_etv.get_eclipse_indices(0.5)
    )
    print(f"Primary eclipse indices: {primary_eclipse_indices}")
    print(f"Secondary eclipse indices: {secondary_eclipse_indices}")

    eclipses = [974]
    print(f"Testing with eclipses: {eclipses}")
    measure_etv.sum_sq_residuals(-0.0610584446328522, eclipses)
    measure_etv.stop_plotting = True
    timeshifts = numpy.linspace(-0.1, 0.1, 1000)
    logprob = [measure_etv.sum_sq_residuals(dt, eclipses) for dt in timeshifts]
    best_fit = measure_etv.fit_timeshift(eclipses)
    print(f"Best fit result: {best_fit!r}")
    pyplot.plot(timeshifts, logprob, "-k")
    pyplot.axvline(x=best_fit.x)
    pyplot.xlabel("O-C")
    pyplot.ylabel("log-likelihood")
    pyplot.show()
    measure_etv.stop_plotting = False
    measure_etv.get_timeshift_distro(eclipses)
