"""Define class for plotting lightcurves and models."""

from matplotlib import pyplot, rcParams
import numpy
from asteval import Interpreter

from tess_target import TESSTarget, get_bls_eclipse_mask
from log_likelihood import LogLikelihood


class LightCurvePlotter:
    """Plot lightcurves, models, and detrndeing of TESS targets."""

    @staticmethod
    def plot_full(lightcurve, model_lcs):
        """Plot the full unfolded lightcurve for a sector."""

        pyplot.plot(lightcurve["time"], lightcurve["flux"], ".")
        for y in model_lcs:
            pyplot.plot(
                lightcurve["time"], y, "-k", linewidth=3.0 / len(model_lcs)
            )
        pyplot.xlabel("Time [d]")
        pyplot.ylabel("Flux [ppm]")

    @staticmethod
    def plot_vs_phase(lightcurve, phase, model_lcs=(), xlabel="Phase"):
        """Create a plot of the lightcurve vs the given phase."""

        phase_order = numpy.argsort(phase)
        ordered_phase = phase[phase_order]
        pyplot.plot(ordered_phase, lightcurve["flux"][phase_order], ".")
        for y in model_lcs:
            pyplot.plot(
                ordered_phase,
                y[phase_order],
                "-k",
                linewidth=3.0 / len(model_lcs),
            )
        pyplot.xlabel(xlabel)
        pyplot.ylabel("Flux [ppm]")

    @classmethod
    def plot_folded(cls, lightcurve, log_likelihood, model_lcs=(), period=None):
        """Show sector lightcurve folded by the best fit BLS period."""

        if period is None:
            period = log_likelihood.best_fit_bls["period"][0]
            if not log_likelihood.masked_is_significant():
                period *= 2
        phase = (lightcurve["time"] % period) / period
        cls.plot_vs_phase(lightcurve, phase, model_lcs)
        pyplot.xlim(0, 1)
        return period

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    @classmethod
    def plot_zoomed(
        cls, lightcurve, bls, zoom, *, model_lcs=(), binaries=None
    ):
        """
        Show folded lightcurve near some BLS transit.

        Possible values for zoom are "default", "even", "odd", "masked",
        "primary", "secondary".
        """

        if zoom in ["primary", "secondary"]:
            binary_ind = 0
            while binary_ind < len(binaries):
                assert binaries is not None
                if zoom == "secondary":
                    binaries[binary_ind].swap_components()
                period = binaries[binary_ind].per
                time_reference = binaries[binary_ind].t0
                eval_t = numpy.linspace(
                    time_reference - period / 2,
                    time_reference + period / 2,
                    100,
                )
                eclipse_lc = binaries[binary_ind].eclipse(eval_t)
                eclipsed = eval_t[eclipse_lc < 1] - time_reference
                if zoom == "secondary":
                    binaries[binary_ind].swap_components()
                if eclipsed.size == 0:
                    binary_ind += 1
                    print(
                        f"No {zoom} eclipse found, trying binary {binary_ind}"
                    )
                    continue
                print(f"Found {zoom} eclipse for binary {binary_ind}")
                half_xrange = max(abs(eclipsed.min()), eclipsed.max())
                pyplot.xlim(-half_xrange, half_xrange)
                plot_lightcurve = lightcurve
                plot_model_lcs = model_lcs
                break
        else:
            if zoom == "default":
                period = bls["period"][0]
                time_reference = bls["transit_time"]
            elif zoom == "masked":
                period = bls["masked_period"][0]
                time_reference = bls["masked_transit_time"]
            else:
                period = 2 * bls["period"][0]
                if zoom == "even":
                    time_reference = bls["transit_time"]
                else:
                    assert zoom == "odd"
                    time_reference = bls["transit_time"] + bls["period"][0]
            mask = get_bls_eclipse_mask(bls, lightcurve, zoom)
            plot_lightcurve = lightcurve[mask]
            plot_model_lcs = [lc[mask] for lc in model_lcs]
            # half_xrange = (
            #    bls[("masked_" if zoom == "masked" else "") + "duration"]
            #    / period
            # )

        phase = (
            (plot_lightcurve["time"] - time_reference + period / 2) % period
        ) / period - 0.5
        cls.plot_vs_phase(
            plot_lightcurve, phase, plot_model_lcs, xlabel=r"$\Delta$Phase"
        )

    # pylint: enable=too-many-locals
    # pylint: enable=too-many-arguments

    def _setup_figure(self, num_lcs):
        """Create the figure and sub-figures for plotting."""

        full_figure = pyplot.figure(
            figsize=(
                rcParams["figure.figsize"][0],
                rcParams["figure.figsize"][1] * 1.5 * num_lcs,
            ),
            # layout="constrained",
        )
        print(f"Mosai spec str: {self._config.plot_lightcurve[1]}")
        return [
            subfig[0]
            for subfig in full_figure.subfigures(
                num_lcs, 1, hspace=0.03, squeeze=False
            )
        ]

    def __init__(self, config):
        """Prepare to plot lightcurves with given configuration."""

        self._config = config
        self._mosaic_spec = Interpreter(
            user_symbols={
                plot_type: plot_type
                for plot_type in [
                    "full",
                    "diff",
                    "folded",
                    "folded_diff",
                    "zoom_default",
                    "zoom_even",
                    "zoom_odd",
                    "zoom_masked",
                    "zoom_primary",
                    "zoom_secondary",
                    "zoom_ooe",
                ]
            }
        )(self._config.plot_lightcurve[1])

    def __call__(self, tic_id, binaries=None, detrend=None):
        """
        Plot lightcurves of TESS target together with detrending and model(s).

        Args:
            tic_id (int):    TIC ID of target to plot.

            binaries (Binary|None):    Collection of binaries configured with
                the model lightcurves to plot.

            detrend(callable):    Function to detrend the lightcurve (detrended
                lightcurve is shown the same way as models.

        Returns:
            None
        """

        tess_target = TESSTarget(tic_id)
        log_likelihood = None

        subfigures = self._setup_figure(len(tess_target.lcs))
        title_pre = f"TIC {self._config.tic_id}\n"
        for (header, lightcurve), subfig in zip(tess_target.lcs, subfigures):
            model_lcs = []
            lightcurve = numpy.copy(lightcurve)
            lightcurve["flux"] /= numpy.median(lightcurve["flux"])
            if binaries is not None:
                model_lcs = [
                    LogLikelihood.get_model(
                        bnry, header, lightcurve, bnry.lc_sys_err
                    )[0]
                    for bnry in binaries
                ]
            if detrend is not None:
                if log_likelihood is None:
                    log_likelihood = LogLikelihood(self._config.tic_id)
                model_lcs.append(detrend(lightcurve, log_likelihood)["flux"])
            print("Mosaic spec: " + repr(self._mosaic_spec))
            for plot_type, axis in subfig.subplot_mosaic(
                self._mosaic_spec,
                gridspec_kw={"wspace": 0.3, "hspace": 0.3},
                sharey=True,
            ).items():
                pyplot.sca(axis)
                if plot_type == "full":
                    self.plot_full(lightcurve, model_lcs)
                else:
                    if log_likelihood is None:
                        log_likelihood = LogLikelihood(self._config.tic_id)
                    if plot_type == "folded":
                        period = self.plot_folded(
                            lightcurve,
                            log_likelihood,
                            model_lcs,
                            None if binaries is None else binaries[0].per,
                        )
                        if title_pre:
                            title_pre = (
                                title_pre.strip()
                                + f": $P_{{orb}}$ = {period:.5f}\n"
                            )
                    else:
                        assert plot_type.startswith("zoom_")
                        self.plot_zoomed(
                            lightcurve,
                            log_likelihood.best_fit_bls,
                            plot_type[len("zoom_") :],
                            model_lcs=model_lcs,
                            binaries=binaries,
                        )
            subfig.suptitle(
                title_pre
                + f"Sector {header['sector']} {header['provenance']} lightcurve"
            )
            title_pre = ""
        if isinstance(self._config.plot_lightcurve[0], tuple):
            pyplot.savefig(
                self._config.plot_lightcurve[0][0],
                format=self._config.plot_lightcurve[0][1],
            )
        else:
            pyplot.savefig(self._config.plot_lightcurve[0])
