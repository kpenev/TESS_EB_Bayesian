"""Define class for plotting lightcurves and models."""

from matplotlib import pyplot, rcParams
import numpy
from asteval import Interpreter

from tess_target import TESSTarget, get_bls_eclipse_mask
from log_likelihood import LogLikelihood


class LightCurvePlotter:
    """Plot lightcurves, models, and detrndeing of TESS targets."""

    lc_plot_config = {
        "good": {
            "marker": ".",
            "markersize": 5,
            "linestyle": "none",
            "markerfacecolor": "green",
            "markeredgecolor": "none",
            "markeredgewidth": 0,
            "zorder": 10,
        },
        "bad": {
            "marker": ".",
            "markersize": 5,
            "linestyle": "none",
            "markerfacecolor": "red",
            "markeredgecolor": "none",
            "markeredgewidth": 0,
            "zorder": 10,
        },
    }
    model_lc_plot_config = {
        "first": {
            "marker": "none",
            "linestyle": "-",
            "color": "red",
            "zorder": 100,
        },
        "others": {
            "marker": "none",
            "linestyle": "-",
            "color": "black",
            "zorder": 20,
        },
    }

    sed_plot_config = {
        "marker": "o",
        "linestyle": "none",
        "markerfacecolor": "green",
        "markeredgecolor": "none",
        "markersize": 10,
        "ecolor": "green",
        "capsize": 5,
        "zorder": 10,
    }

    model_sed_plot_config = {
        "first": {
            "marker": "x",
            "linestyle": "none",
            "color": "red",
            "markersize": 10,
            "zorder": 100,
        },
        "others": {
            "marker": "x",
            "linestyle": "none",
            "color": "black",
            "markersize": 10,
            "zorder": 20,
        },
    }

    sed_filters = {
        "SDSS": {
            "u": 0.354,
            "g": 0.477,
            "r": 0.623,
            "i": 0.762,
            "z": 0.913,
            "y": 1.021,
        },
        "2MASS": {
            "J": 1.235,
            "H": 1.662,
            "Ks": 2.159,
        },
        "WISE": {
            "W1": 3.35,
            "W2": 4.6,
            "W3": 11.56,
            "W4": 22.09,
        },
        "PanSTARRS": {
            "g": 0.481,
            "r": 0.617,
            "i": 0.752,
            "z": 0.866,
            "y": 0.962,
        },
    }

    def plot_full(self, lightcurve, model_lcs):
        """Plot the full unfolded lightcurve for a sector."""

        for mask, plot_config in [
            (lightcurve["good"], self.lc_plot_config["good"]),
            (numpy.logical_not(lightcurve["good"]), self.lc_plot_config["bad"]),
        ]:
            pyplot.plot(
                lightcurve["time"][mask],
                lightcurve["flux"][mask],
                **plot_config,
            )
        ylim = pyplot.ylim()
        config_key = "first" if self._config.highlight_first_model else "others"
        for y in model_lcs:
            pyplot.plot(
                lightcurve["time"],
                y,
                linewidth=(
                    3.0 / (1 if config_key == "first" else len(model_lcs))
                ),
                **self.model_lc_plot_config[config_key],
            )
            config_key = "others"
        pyplot.xlabel("Time [d]")
        pyplot.ylabel("Flux")
        pyplot.ylim(ylim)

    def plot_vs_phase(self, lightcurve, phase, model_lcs=(), xlabel="Phase"):
        """Create a plot of the lightcurve vs the given phase."""

        phase_order = numpy.argsort(phase)
        print(f"Phase order: {phase_order!r}")
        print(f"Phase: {phase!r}")
        ordered_phase = phase[phase_order]
        for mask, plot_config in [
            (lightcurve["good"][phase_order], self.lc_plot_config["good"]),
            (
                numpy.logical_not(lightcurve["good"][phase_order]),
                self.lc_plot_config["bad"],
            ),
        ]:
            pyplot.plot(
                ordered_phase[mask],
                lightcurve["flux"][phase_order][mask],
                **plot_config,
            )
        ylim = pyplot.ylim()
        first = self._config.highlight_first_model
        for y in model_lcs:
            pyplot.plot(
                ordered_phase,
                y[phase_order],
                linewidth=3.0 / (1 if first else len(model_lcs)),
                **self.model_lc_plot_config["first" if first else "others"],
            )
            first = False
        pyplot.xlabel(xlabel)
        pyplot.ylabel("Flux [ppm]")
        pyplot.ylim(ylim)

    def plot_folded(self, lightcurve, model_lcs=()):
        """Show sector lightcurve folded by the best fit BLS period."""

        phase = (
            lightcurve["time"] % self._folding_period
        ) / self._folding_period
        self.plot_vs_phase(lightcurve, phase, model_lcs)
        pyplot.xlim(0, 1)

    @staticmethod
    def _get_phase(lightcurve, period, time_reference=0):
        """Return the phase of the lightcurve given the BLS results."""

        return (
            (lightcurve["time"] - time_reference + period / 2) % period
        ) / period - 0.5

    def _plot_phase_zoomed(self, lightcurve, time_reference, period, model_lcs):
        """Create zoomed plot on primary on secondary eclipse per binary."""

        phase = self._get_phase(lightcurve, period, time_reference)
        self.plot_vs_phase(
            lightcurve, phase, model_lcs, xlabel=r"$\Delta$Phase"
        )

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def _plot_ooe_zoomed(self, lightcurve, model_lcs, plot_x, ooe_mask, xlabel):
        """Create zoomed plot on out-of-eclipse per binary."""

        flux = lightcurve["flux"][ooe_mask]
        ymin = flux.min()
        ymax = flux.max()
        for y in model_lcs:
            flux = y[ooe_mask]
            ymin = min(flux.min(), ymin)
            ymax = max(flux.max(), ymax)
        self.plot_vs_phase(lightcurve, plot_x, model_lcs, xlabel)
        pad = 0.05 * (ymax - ymin)
        pyplot.ylim(ymin - pad, ymax + pad)

    # pylint: enable=too-many-arguments
    # pylint: enable=too-many-positional-arguments

    @staticmethod
    def _convert_binary_to_bsl_zoom(zoom, log_likelihood):
        """Return BLS zoom name that corresponds to given binary zoom name."""

        if log_likelihood.masked_is_significant():
            return "default" if zoom == "primary" else "masked"

        return "even" if zoom == "primary" else "odd"

    def plot_zoomed_binary(self, zoom, lightcurve, model_lcs, binaries):
        """Create zoomed plot on primary or secondary eclipse per binary."""

        binary_ind = 0
        ooe_mask = True
        ooe_timeref = 0
        for eclipse in (
            ["primary", "secondary"] if zoom.startswith("ooe") else [zoom]
        ):
            half_xrange = None
            while binary_ind < len(binaries):
                assert binaries is not None
                if eclipse == "secondary":
                    binaries[binary_ind].swap_components()
                period = binaries[binary_ind].per
                time_reference = binaries[binary_ind].t0
                ooe_timeref += time_reference
                eval_t = numpy.linspace(
                    time_reference - period / 2,
                    time_reference + period / 2,
                    100,
                )
                eclipse_lc = binaries[binary_ind].eclipse(eval_t)
                eclipsed = (eval_t[eclipse_lc < 1] - time_reference) / period
                if eclipse == "secondary":
                    binaries[binary_ind].swap_components()
                if eclipsed.size == 0:
                    binary_ind += 1
                    print(
                        f"No {eclipse} eclipse found, trying binary "
                        f"{binary_ind}"
                    )
                    continue
                print(f"Found {eclipse} eclipse for binary {binary_ind}")
                half_xrange = max(abs(eclipsed.min()), eclipsed.max())
                break
            if half_xrange is None:
                raise RuntimeError()
            if zoom.startswith("ooe"):
                phase = self._get_phase(lightcurve, period, time_reference)
                ooe_mask = numpy.logical_and(
                    ooe_mask,
                    numpy.logical_or(phase < -half_xrange, phase > half_xrange),
                )

        if zoom.startswith("ooe"):
            ooe_timeref /= 2
            phase = self._get_phase(lightcurve, period, ooe_timeref)
            self._plot_ooe_zoomed(
                lightcurve,
                model_lcs,
                lightcurve["time"] if zoom == "ooe" else phase,
                ooe_mask,
                "Time [d]" if zoom == "ooe" else "Phase",
            )
        else:
            half_xrange *= 1.5
            pyplot.xlim(-half_xrange, half_xrange)
            self._plot_phase_zoomed(
                lightcurve, time_reference, period, model_lcs
            )

    def plot_zoomed_bls(self, zoom, lightcurve, model_lcs, bls):
        """Create zoomed plot on BLS detected eclipses."""

        print(f"Creating zoom {zoom} LC plot")
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
                if zoom != "odd":
                    raise RuntimeError(f"Unrecognized BLS zoom label: {zoom}")
                time_reference = bls["transit_time"] + bls["period"][0]
        mask = get_bls_eclipse_mask(bls, lightcurve, zoom)
        self._plot_phase_zoomed(
            lightcurve[mask],
            time_reference,
            period,
            [lc[mask] for lc in model_lcs],
        )

    def plot_sed(self, tess_target, binaries):
        """Create a plot of the SED of the target along with model."""

        plot_x = numpy.array(
            [self.sed_filters["PanSTARRS"][filter] for filter in "grizy"]
            + [self.sed_filters["2MASS"][filter] for filter in ["J", "H", "Ks"]]
            + [self.sed_filters["WISE"][filter] for filter in ["W1", "W2"]]
        )
        pyplot.errorbar(
            plot_x,
            tess_target.sed[0],
            yerr=tess_target.sed[1],
            **self.sed_plot_config,
        )
        config_key = "first" if self._config.highlight_first_model else "others"
        for bnry in binaries or ():
            pyplot.plot(
                plot_x, bnry.absmag, **self.model_sed_plot_config[config_key]
            )
            config_key = "others"
        pyplot.xlabel(r"Wavelength [$\mu$]")
        pyplot.ylabel("Absolute magnitude")

    def plot_diff(self, mode, lightcurve, model_lcs):
        """Plot the difference between the LC and the first model vs time."""

        lightcurve = numpy.copy(lightcurve)
        lightcurve["flux"] -= model_lcs[0]
        getattr(self, f"plot_{mode}")(
            lightcurve, [lc - model_lcs[0] for lc in model_lcs[1:]]
        )
        pyplot.ylabel("Flux diff.")

    def _get_model_lcs(self, header, lightcurve, binaries, detrend):
        """Return the model lightcurves to add on top of the data."""

        model_lcs = []
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
        return model_lcs

    def _setup_figure(self, num_lcs):
        """Create the figure and sub-figures for plotting."""

        full_figure = pyplot.figure(
            figsize=(
                rcParams["figure.figsize"][0],
                rcParams["figure.figsize"][1] * 1.5 * num_lcs,
            ),
            layout="constrained",
        )
        print(f"Mosai spec str: {self._config.plot_lightcurve[1]}")
        return [
            subfig[0]
            for subfig in full_figure.subfigures(num_lcs, 1, squeeze=False)
        ]

    def __init__(self, config):
        """Prepare to plot lightcurves with given configuration."""

        self._config = config
        print(f"Setting mosaic from: {self._config.plot_lightcurve!r}")
        self._mosaic_spec = Interpreter(
            user_symbols={
                plot_type: plot_type
                for plot_type in [
                    "full",
                    "full_diff",
                    "folded",
                    "folded_diff",
                    "zoom_default",
                    "zoom_even",
                    "zoom_odd",
                    "zoom_masked",
                    "zoom_primary",
                    "zoom_secondary",
                    "zoom_ooe",
                    "zoom_ooe_folded",
                    "sed",
                    "empty",
                ]
            }
        )(self._config.plot_lightcurve[1])
        if self._mosaic_spec is None:
            raise ValueError(
                "Unable to parse mosaic specification: "
                + repr(self._config.plot_lightcurve[1])
            )
        plot_types = numpy.unique(numpy.array(self._mosaic_spec).flatten())
        self._full_log_likelihood = (
            plot_types.size != 1 or plot_types[0] != "full"
        )
        self._folding_period = getattr(config, "folding_period", None)
        if config.data_on_top:
            for cfg in self.lc_plot_config.values():
                cfg["zorder"] = 30
            self.sed_plot_config["zorder"] = 30

    def __call__(self, tic_id, binaries=None, detrend=None, title_info=None):
        """
        Plot lightcurves of TESS target together with detrending and model(s).

        Args:
            tic_id (int):    TIC ID of target to plot.

            binaries ([Binary]|None):    Collection of binaries configured with
                the model lightcurves to plot.

            detrend(callable):    Function to detrend the lightcurve (detrended
                lightcurve is shown the same way as models.

        Returns:
            None
        """

        title_pre = f"TIC {tic_id}\n"
        if self._full_log_likelihood:
            tess_target = LogLikelihood(tic_id)
            if self._folding_period is None:
                self._folding_period = tess_target.bls_porb

                if binaries and numpy.allclose(
                    binaries[0].per, self._folding_period, rtol=1e-3
                ):
                    self._folding_period = binaries[0].per
            title_pre = (
                title_pre.strip()
                + f": $P_{{orb}}$ = {self._folding_period:.5f}\n"
            )
        else:
            tess_target = TESSTarget(tic_id)

        title_pre += f"({title_info})"

        subfigures = self._setup_figure(len(tess_target.lcs))
        for (header, lightcurve), subfig in zip(tess_target.lcs, subfigures):
            lightcurve = numpy.copy(lightcurve)
            lightcurve["flux"] /= numpy.median(lightcurve["flux"])

            model_lcs = self._get_model_lcs(
                header, lightcurve, binaries, detrend
            )
            print("Mosaic spec: " + repr(self._mosaic_spec))
            for plot_type, axis in subfig.subplot_mosaic(
                self._mosaic_spec,
                empty_sentinel="empty",
                gridspec_kw={"hspace": 0.0},
            ).items():
                pyplot.sca(axis)
                if plot_type.startswith("zoom_"):
                    zoom_type = plot_type[len("zoom_") :]
                    try:
                        getattr(
                            self,
                            "plot_zoomed_"
                            + ("binary" if binaries is not None else "bls"),
                        )(
                            zoom_type,
                            lightcurve,
                            model_lcs,
                            binaries or tess_target.best_fit_bls,
                        )
                    except RuntimeError:
                        self.plot_zoomed_bls(
                            self._convert_binary_to_bsl_zoom(
                                zoom_type, tess_target
                            ),
                            lightcurve,
                            model_lcs,
                            tess_target.best_fit_bls,
                        )
                elif plot_type.endswith("_diff"):
                    self.plot_diff(
                        plot_type[: -len("_diff")], lightcurve, model_lcs
                    )
                elif plot_type == "sed":
                    self.plot_sed(tess_target, binaries)
                else:
                    getattr(self, f"plot_{plot_type}")(lightcurve, model_lcs)

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
