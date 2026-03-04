"""Class which visualizes the inner workings of MeasureETV."""

import logging

import numpy
from matplotlib import pyplot
from measure_etv import MeasureETV
from sample_params import SampleParams

from hacked_emcee_hdf5_backend import HDFBackend


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tic_id = 176591772
    backend = HDFBackend(
        f"/mnt/md2/TESS_EBs/first1000/tess{tic_id}_samples.h5",
        name="mcmc",
        read_only=True,
    )
    log_prob = backend.get_log_prob()
    top_index = numpy.unravel_index(numpy.argmax(log_prob), log_prob.shape)
    top_params = SampleParams(
        *backend.get_blobs(discard=0, thin=top_index[0] + 1)[0][top_index[1:]]
    )
    measure_etv = MeasureETVVisualize(tic_id, top_params, pad_duration=0.1)
    measure_etv.sum_sq_residuals(0.0, [189, 190])
    measure_etv.stop_plotting = True
    timeshifts = numpy.linspace(-0.01, 0.01, 100)
    logprob = [
        measure_etv.sum_sq_residuals(dt, [189, 190]) for dt in timeshifts
    ]
    best_fit = measure_etv.fit_timeshift([189, 190])
    print(f'Best fit result: {best_fit!r}')
    pyplot.plot(timeshifts, logprob, "-k")
    pyplot.axvline(x=best_fit.x)
    pyplot.xlabel("O-C")
    pyplot.ylabel("log-likelihood")
    pyplot.show()
