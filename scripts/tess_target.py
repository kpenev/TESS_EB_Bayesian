"""Define a class providing a uniform interface to TESS lightcurves."""

import logging

from matplotlib import pyplot

try:
    from matplotlib import colormaps
except ImportError:
    # For older matplotlib versions (< 3.5)
    from matplotlib import cm as colormaps
import numpy
from astropy.timeseries import BoxLeastSquares

# from foldedleastsquares import transitleastsquares, transit_mask

from download_lcs import get_astroquery as download_lcs
from exclude_data import exclude_data
from catalog_interface import get_eb_info


class FalsePositiveError(Exception):
    """Raised when the TLS results suggest a false positive."""


def get_bls_eclipse_mask(bls, lightcurve, which):
    """
    Filter the lightcure per best fit BLS to leave only points around eclipse.

    See `self.calc_lc_log_likelihood()` for what masks are supported.
    """

    if not isinstance(which, str):
        result = numpy.zeros(lightcurve["time"].size, dtype=bool)
        for entry in which:
            result |= get_bls_eclipse_mask(bls, lightcurve, entry)
        return result

    if which == "masked":
        period = bls["masked_period"][0]
        window = 0.75 * bls["masked_duration"] + min(
            bls["masked_period"][1], 0.5 * bls["masked_duration"]
        )
        time = lightcurve["time"] - bls["masked_transit_time"]
        print(
            f"Based on masked duration {bls['masked_duration']} and masked "
            f"P uncertainty {bls['masked_period'][1]}, window = {window}",
        )
    else:
        period = bls["period"][0] * (1 if which == "both" else 2)
        window = 0.75 * bls["duration"] + min(
            bls["period"][1], 0.5 * bls["duration"]
        )
        time = lightcurve["time"] - bls["transit_time"]
        if which == "odd":
            time -= bls["period"][0]

        print(
            f"Based on duration {bls['duration']} and "
            f"P uncertainty {bls['period'][1]}, window = {window}",
        )

    window = min(0.3 * period, window)

    folded = time % period
    return numpy.minimum(folded, period - folded) < window


def _evaluate_bls(times, period, duration, depth):
    """Evaluate a BLS model with the given parameters at the given times."""

    in_transit = (
        numpy.abs((times + period / 2) % period - period / 2) < 0.5 * duration
    )
    model = numpy.ones(times.size)
    model[in_transit] -= depth
    return model


def get_bls_model(lightcurve, log_likelihood):
    """Return the best fit BLS model evaluated at the LC times."""

    print("Adding BLS model per: %s", repr(log_likelihood.best_fit_bls))
    model = _evaluate_bls(
        lightcurve["time"] - log_likelihood.best_fit_bls["transit_time"],
        log_likelihood.best_fit_bls["period"][0],
        log_likelihood.best_fit_bls["duration"],
        log_likelihood.best_fit_bls["depth"][0],
    )
    if log_likelihood.masked_is_significant():
        model *= _evaluate_bls(
            lightcurve["time"]
            - log_likelihood.best_fit_bls["masked_transit_time"],
            log_likelihood.best_fit_bls["masked_period"][0],
            log_likelihood.best_fit_bls["masked_duration"],
            log_likelihood.best_fit_bls["masked_depth"][0],
        )

    return model


class TESSTarget:
    """
    Manage downloading and preparing TESS lightcurves for MCMC.

    Lightcurves are organized by sector.

    If both SPOC and QLP lightcurves are available for a sector, only the SPOC
    one is kept.
    """

    _logger = logging.getLogger(__name__)

    _bad_mask = 0

    for bad_ind in [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]:
        _bad_mask |= 1 << (bad_ind - 1)

    _tls_power_kwargs = {
        "R_star_min": 0.1,
        "R_star_max": 10.0,
        "M_star_max": 10.0,
        "period_min": 1.0,
        "period_max": 1.1,
        "transit_depth_min": 0.03,
        "per": 3.0,
        "rp": 0.1,
        "a": 11.0,
        "b": 0.0,
        # "transit_template": "grazing",
    }

    def _get_lc_format(self, sector, header, provenance):
        """Return the relevant column names and exposure time for gvien LC."""

        if provenance == "QLP":
            for existing_header, _ in self._lcs:
                if existing_header["sector"] == sector:
                    return None, None

            for err_col in ["KSPSAP_FLUX_ERR", "DET_FLUX_ERR"]:
                if err_col in header.values():
                    return ("SAP_FLUX", err_col), header["TIMEDEL"]
            self._logger.critical(
                "TIC %d, sector %d, %s lightcurve has no flux error column.",
                self._tic_id,
                sector,
                provenance,
            )
            assert False

        assert header["TIMEPIXR"] == 0.5
        # Iterable needs to be modified
        # pylint: disable=consider-using-enumerate
        for i in range(len(self._lcs)):
            if self._lcs[i][0]["sector"] == sector:
                assert self._lcs[i][0]["provenance"] == "QLP"
                del self._lcs[i]
                break
        # pylint: enable=consider-using-enumerate

        return ("PDCSAP_FLUX", "PDCSAP_FLUX_ERR"), header["INT_TIME"] * header[
            "NUM_FRM"
        ] / 86400

    def _format_lc(self, sector, header, provenance, observed_lc):
        """Return a lightcurve formatted uniformly regardless of provenance."""

        flux_columns, exptime = self._get_lc_format(sector, header, provenance)
        if exptime is None:
            assert flux_columns is None
            return None, None
        for column in flux_columns:
            if column not in observed_lc.columns.names:
                self._logger.critical(
                    "TIC %d, sector %d, %s lightcurve has no %s column.",
                    self._tic_id,
                    sector,
                    provenance,
                    column,
                )

        usable = numpy.logical_and(
            numpy.isfinite(observed_lc[flux_columns[0]]),
            numpy.isfinite(observed_lc[flux_columns[1]]),
        )
        usable = numpy.logical_and(
            usable,
            numpy.logical_not(observed_lc["QUALITY"] & self._bad_mask),
        )
        if not usable.any():
            self._logger.warning(
                "TIC %d, sector %d, %s lightcurve has 0/%d usable points.",
                self._tic_id,
                sector,
                provenance,
                usable.size,
            )
            return None, None
        observed_lc = observed_lc[usable]
        formatted_lc = numpy.empty(
            usable.sum(),
            dtype=[
                ("time", ">f8"),
                ("flux", ">f4"),
                ("flux_err", ">f4"),
                ("good", "bool"),
            ],
        )
        formatted_lc["time"] = observed_lc["TIME"]
        scaling = numpy.nanmedian(observed_lc[flux_columns[0]])
        formatted_lc["flux"] = observed_lc[flux_columns[0]] / scaling
        formatted_lc["flux_err"] = observed_lc[flux_columns[1]] / scaling
        formatted_lc["good"] = True
        return formatted_lc, {
            "exptime": exptime,
            "sector": sector,
            "provenance": provenance,
        }

    @property
    def lcs(self):
        """The lightcurves and minimal metadata for this target by sector."""

        return self._lcs

    @property
    def time_span(self):
        """The time span of the lightcurves for this target."""

        return self._time_span

    def _get_bls_period_step(self):
        """Return the inverse-period step for the BLS search."""

        return 0.02 / (
            30.0
            * min(
                (self._time_span[1] - self._time_span[0]),
                300,
            )
        )

    @staticmethod
    def _estimate_period_uncertainty(periodogram, best_index):
        """Estimate the period uncertainty per the given BLS periodogram."""

        index_range = [0, periodogram.power.size - 1]
        cutoff = 0.3 * periodogram.power[best_index]
        if best_index > 0 and periodogram.power[:best_index].min() < cutoff:
            index_range[0] = numpy.where(
                periodogram.power[:best_index] < cutoff
            )[0][-1]
        if (
            best_index < periodogram.power.size - 1
            and periodogram.power[best_index:].min() < cutoff
        ):
            index_range[1] = (
                numpy.where(periodogram.power[best_index:] < cutoff)[0][0]
                + best_index,
            )
        # pyplot.plot(periodogram.period, periodogram.power, "-k")
        # for color, i in zip(
        #    "rgb", [index_range[0], best_index, index_range[1]]
        # ):
        #    pyplot.axvline(x=periodogram.period[i], color=color)
        # pyplot.show()

        assert index_range[0] <= best_index
        assert index_range[1] >= best_index

        return max(
            periodogram.period[best_index] - periodogram.period[index_range[0]],
            periodogram.period[index_range[1]] - periodogram.period[best_index],
        )

    @staticmethod
    def _assemble_bls_result(model, periodogram, best_index):
        """Format the BLS result from the given periodogram and statistics."""

        result = {
            param: getattr(periodogram, param)[best_index]
            for param in ["period", "duration", "transit_time"]
        }
        result.update(model.compute_stats(**result))
        del result["transit_times"]
        del result["per_transit_count"]
        del result["per_transit_log_likelihood"]
        result["period"] = (
            result["period"],
            TESSTarget._estimate_period_uncertainty(periodogram, best_index),
        )

        return result

    @staticmethod
    def _fit_bls(lightcurve, periods, durations):
        """Fit BLS to the given lightcurve avoiding partial eclipses."""

        model = BoxLeastSquares(
            lightcurve["time"],
            lightcurve["flux"],
            dy=lightcurve["flux_err"],
        )
        periodogram = model.power(periods, durations)
        result = TESSTarget._assemble_bls_result(
            model, periodogram, numpy.argmax(periodogram.power)
        )
        in_eclipse = get_bls_eclipse_mask(result, lightcurve, "both")

        # pyplot.subplot(231)
        # pyplot.plot(periods, periodogram["power"], ".k")
        # pyplot.subplot(232)
        # pyplot.plot(periods, periodogram["duration"], ".k")
        # pyplot.subplot(233)
        # pyplot.plot(periods, periodogram["depth"], ".k")
        # pyplot.subplot(212)
        # pyplot.plot(lightcurve["time"], lightcurve["flux"], ".k")
        # pyplot.plot(
        #    lightcurve["time"],
        #    _evaluate_bls(
        #        lightcurve["time"] - result["transit_time"],
        #        result["period"][0],
        #        result["duration"],
        #        result["depth"][0],
        #    ),
        #    "-r",
        # )
        # pyplot.show()

        if in_eclipse[0] or in_eclipse[-1]:
            out_of_eclipse = numpy.argwhere(numpy.logical_not(in_eclipse))
            start = int(out_of_eclipse[0]) if in_eclipse[0] else 0
            end = int(out_of_eclipse[-1]) if in_eclipse[-1] else in_eclipse.size
            return TESSTarget._fit_bls(
                lightcurve[start:end], periods, durations
            )
        return result

    def _get_catalog_bls(self, lightcurve, cat_info):
        """Compute the BLS periodograms for P and P/2."""

        catalog_period = cat_info["period"]
        self._logger.debug(
            "Testing 2P, P, and P/2 for TIC %s, catalog P=%.4fd",
            self._tic_id,
            catalog_period,
        )

        bls_results = []
        for p_factor in [0.5, 1, 2]:
            if (
                2 * catalog_period / p_factor
                > lightcurve["time"][-1] - lightcurve["time"][0]
            ):
                bls_results.append(None)
                continue
            periods = 1.0 / numpy.arange(
                p_factor / (0.95 * catalog_period),
                p_factor / (1.05 * catalog_period),
                -self._get_bls_period_step(),
            )
            durations = numpy.linspace(
                min(periods[0] / 100, 0.02), periods[0] / 2, 1000
            )
            self._logger.debug(
                "Based on Porb = %s, BLS durations: %s",
                repr(periods[0]),
                repr(durations),
            )
            result = self._fit_bls(lightcurve, periods, durations)
            result.update(self._get_masked_best_fit_bls(lightcurve, result))

            self._logger.debug(
                "Best BLS for %s < P < %s periodogram with %d periods and "
                "%s <= duration <= %s: %s",
                periods[0],
                periods[-1],
                periods.size,
                durations[0],
                durations[-1],
                repr(result),
            )

            bls_results.append(result)

        assert (
            bls_results[0] is None
            or bls_results[0]["period"][0] > 10
            or TESSTarget.masked_bls_is_significant(bls_results[0])
        )
        assert not TESSTarget.masked_bls_is_significant(bls_results[2])

        if not TESSTarget.masked_bls_is_significant(bls_results[1]):
            self._logger.warning(
                "Catalog period (%s) appears to be half of the true orbital "
                "period: %s",
                repr(catalog_period),
                repr(bls_results[0]["period"]),
            )
            return bls_results[1]

        if (
            bls_results[1]["masked_depth"][0] > 0.5 * bls_results[1]["depth"][0]
            and abs(
                abs(
                    bls_results[1]["transit_time"]
                    - bls_results[1]["masked_transit_time"]
                )
                - bls_results[1]["period"][0] / 2
            )
            < bls_results[1]["period"][1]
        ):
            self._logger.debug(
                "Primary and secondary eclipses appear to have comparable "
                "depths (%s +- %s and %s +- %s) and are %s days apart, very "
                "close to P/2 (%s +- %s) apart. Using half-period BLS.",
                *bls_results[1]["depth"],
                *bls_results[1]["masked_depth"],
                (
                    bls_results[1]["transit_time"]
                    - bls_results[1]["masked_transit_time"]
                ),
                bls_results[1]["period"][0] / 2,
                bls_results[1]["period"][1] / 2,
            )
            return bls_results[2]

        self._logger.debug("Using catalog period BLS")
        return bls_results[1]

    def _get_best_fit_bls(self, lightcurve, periods=None, use_catalog=True):
        """Return the best fit orbital period and time of primary transit."""

        if use_catalog:
            assert periods is None
            cat_info = get_eb_info(self._tic_id)
            if cat_info is not None:
                return self._get_catalog_bls(lightcurve, cat_info)

        # Fallback to original logic for blind search or when catalog
        # unavailable
        self._logger.debug("Computing BLS for periods:\n%s", repr(periods))
        result = self._fit_bls(
            lightcurve,
            (
                1.0
                / numpy.arange(
                    1.0 / 0.4,
                    1.0 / 30.0,
                    -self._get_bls_period_step(),
                )
                if periods is None
                else periods
            ),
            numpy.linspace(
                min(periods[0] / 10, 0.02), max(periods[0] / 2, 0.2), 100
            ),
        )

        self._logger.debug(
            "Results:\n\t%s",
            "\n\t".join(
                [f"{param}: {value}" for param, value in result.items()]
            ),
        )
        if periods is None:
            if result["duration"] > 0.15:
                return self._fit_bls(
                    lightcurve,
                    1.0 / numpy.arange(1 / 2.0, 0.01, -0.1 / 30.0**2),
                    numpy.linspace(0.1, 1.0, 100),
                )
            elif periodogram.duration[best_index] < 0.04:
                return self._fit - bls(
                    lightcurve,
                    1.0 / numpy.arange(1 / 0.1, 0.01, -0.01 / 30.0**2),
                    numpy.linspace(0.01, 0.05, 100),
                )

        return result

    def _get_masked_best_fit_bls(self, lightcurve, bls_results):
        """Mask the eclispes detected by given BLS and fit BLS again."""

        mask = numpy.logical_not(
            get_bls_eclipse_mask(bls_results, lightcurve, "both")
        )
        self._logger.debug("After masking, %d points remain", mask.sum())
        period_range = (
            max(bls_results["period"][0] - 5 * bls_results["period"][1], 0.1),
            min(bls_results["period"][0] + 5 * bls_results["period"][1], 100),
        )
        periods = 1.0 / numpy.arange(
            1.0 / period_range[0],
            1.0 / period_range[1],
            -self._get_bls_period_step(),
        )
        if periods.size < 10:
            periods = numpy.linspace(*period_range, 10)
        self._logger.debug(
            "Covering period range %s < Porb < %s with %d points",
            *period_range,
            periods.size,
        )

        masked_bls_result = self._get_best_fit_bls(
            lightcurve[mask],
            periods=periods,
            use_catalog=False,
        )
        return {
            f"masked_{param}": value
            for param, value in masked_bls_result.items()
        }

    @staticmethod
    def _get_bls_plot_x(lightcurve, best_fit_bls, label):
        """Return x and sortind indices of the lightcurve for subplot."""

        if label in ["folded", "zoom_even", "zoom_odd"]:
            plot_x = lightcurve["time"] - best_fit_bls["transit_time"]
            if label == "zoom_even":
                plot_x += best_fit_bls["period"][0]
            elif label == "folded":
                plot_x += best_fit_bls["period"][0] / 2
            plot_x %= 2 * best_fit_bls["period"][0]
        else:
            plot_x = lightcurve["time"]

        sorter = numpy.argsort(plot_x)
        plot_x = plot_x[sorter]
        return plot_x, sorter

    @staticmethod
    def _get_bls_model(lightcurve, best_fit_bls, prefix, suffix):
        """Return the given masked/unmasked and both/even/odd BLS model."""

        folded = (
            lightcurve["time"]
            - best_fit_bls[prefix + "transit_time"]
            - (best_fit_bls[prefix + "period"][0] if suffix == "_odd" else 0)
        ) % (best_fit_bls[prefix + "period"][0] * (2 if suffix else 1))
        in_transit = (
            numpy.minimum(
                folded,
                (best_fit_bls[prefix + "period"][0] * (2 if suffix else 1))
                - folded,
            )
            < best_fit_bls[prefix + "duration"] / 2
        )

        return (
            1.0 - best_fit_bls[prefix + "depth" + suffix][0] * in_transit
        ) * numpy.nanmedian(lightcurve["flux"])

    def plot_best_fit_bls(self, combined_lc, best_fit_bls, pdf=None):
        """Create plots showing the best fit BLS paramaters on top of LCs."""

        cmap = dict(
            zip(
                [
                    f"{prefix}BLS{suffix}"
                    for prefix in ["", "masked "]
                    for suffix in ["", " odd", " even"]
                ],
                colormaps["Dark2"].colors,
            )
        )
        for header, lightcurve in self._lcs + [
            ({"sector": "all"}, combined_lc)
        ]:
            pyplot.figure(figsize=[4.8, 6.4])
            for label, axis in pyplot.subplot_mosaic(
                [
                    6 * ["full"],
                    3 * ["first"] + 3 * ["last"],
                    [
                        "folded",
                        "folded",
                        "zoom_even",
                        "zoom_even",
                        "zoom_odd",
                        "zoom_odd",
                    ],
                ]
            )[1].items():
                pyplot.sca(axis)
                plot_x, sorter = self._get_bls_plot_x(
                    lightcurve, best_fit_bls, label
                )
                pyplot.plot(
                    plot_x,
                    lightcurve["flux"][sorter],
                    ".k",
                    markersize=1,
                )
                for prefix in (
                    ["", "masked_"]
                    if label in ["full", "first", "last"]
                    else [""]
                ):
                    for suffix in ["", "_odd", "_even"]:
                        curve = f"{prefix}BLS{suffix}".replace("_", " ")
                        pyplot.plot(
                            plot_x,
                            self._get_bls_model(
                                lightcurve, best_fit_bls, prefix, suffix
                            )[sorter],
                            "-",
                            label=curve if label == "full" else None,
                            color=cmap[curve],
                            linewidth=1,
                        )
                if label == "first":
                    pyplot.xlim(
                        lightcurve["time"][0],
                        lightcurve["time"][0] + 3 * best_fit_bls["period"][0],
                    )
                elif label == "last":
                    pyplot.xlim(
                        lightcurve["time"][-1] - 3 * best_fit_bls["period"][0],
                        lightcurve["time"][-1],
                    )
                elif label.startswith("zoom"):
                    pyplot.xlim(
                        best_fit_bls["period"][0] - best_fit_bls["duration"],
                        best_fit_bls["period"][0] + best_fit_bls["duration"],
                    )
            pyplot.figlegend()

            pyplot.suptitle(f"Sector {header['sector']}")
            if pdf is None:
                pyplot.show()
            else:
                pdf.savefig()

    @property
    def tic_id(self):
        """The TIC identifier of the EB being modeled."""

        return self._tic_id

    def __init__(self, tic_id):
        """Download and organize the lightcurves for the given TIC ID."""

        self._tic_id = tic_id
        lcs = {
            provenance: download_lcs(tic_id, "all", provenance=provenance)
            for provenance in ["SPOC", "QLP"]
        }

        self._lcs = []
        self._time_span = [numpy.inf, -numpy.inf]

        for provenance, lc_collection in lcs.items():
            if provenance in exclude_data.get(tic_id, []):
                continue
            for sector, (header, observed_lc) in lc_collection.items():
                if sector in exclude_data.get(tic_id, []):
                    continue
                formatted_lc, formatted_header = self._format_lc(
                    sector, header, provenance, observed_lc
                )
                if formatted_lc is None:
                    continue
                self._time_span[0] = min(
                    formatted_lc["time"].min(), self._time_span[0]
                )
                self._time_span[1] = max(
                    formatted_lc["time"].max(), self._time_span[1]
                )

                self._lcs.append((formatted_header, formatted_lc))
        self._lcs.sort(key=lambda x: x[0]["sector"])

    def get_combined_lc(self):
        """Return LC combining all sectors after scaling by median."""

        combined_lc = None
        for _, formatted_lc in self.lcs:
            finite_lc = numpy.copy(
                formatted_lc[
                    numpy.logical_and(
                        numpy.isfinite(formatted_lc["flux"]),
                        numpy.isfinite(formatted_lc["flux_err"]),
                    )
                ]
            )
            med_flux = numpy.nanmedian(formatted_lc["flux"])
            if combined_lc is None:
                combined_lc = finite_lc
                combined_lc["flux"] /= med_flux
                combined_lc["flux_err"] /= med_flux
            else:
                combined_lc = numpy.concatenate([combined_lc, finite_lc])
                # False positive
                # pylint: disable=invalid-unary-operand-type
                combined_lc[-formatted_lc.size :]["flux"] /= med_flux
                combined_lc[-formatted_lc.size :]["flux_err"] /= med_flux
                # pylint: enable=invalid-unary-operand-type

        return combined_lc

    def fit_bls(self, use_catalog=True):
        """Fit BLS models to the given lightcurve."""

        combined_lc = self.get_combined_lc()
        best_fit_bls = self._get_best_fit_bls(
            combined_lc, use_catalog=use_catalog
        )
        if not use_catalog:
            best_fit_bls.update(
                self._get_masked_best_fit_bls(combined_lc, best_fit_bls)
            )

        self._logger.info(
            "Combined LC has %s usable points, BLS results:\n\t%s",
            repr(combined_lc.size),
            "\n\t".join(
                [f"{param}: {value}" for param, value in best_fit_bls.items()]
            ),
        )
        return best_fit_bls

    @staticmethod
    def masked_bls_is_significant(bls):
        """
        Return True iff the masked BLS fit appears to fit real eclipses.

        To be marked as significant all of the following must be satisfied:

          * maked period should be close to unmasked period (within 5 unmasked
            uncertanties)

          * depth should exceed its uncertanity by at least a factor of 5

          * masked_harmonic_delta_log_likelihood < -5
        """

        # bls = self._best_fit_bls
        return (
            bls["masked_depth"][0] > 10.0 * bls["masked_depth"][1]
            and bls["masked_harmonic_delta_log_likelihood"] < -10.0
            and abs(bls["depth_even"][0] - bls["depth_odd"][0])
            < max(
                10.0 * (bls["depth_even"][1] + bls["depth_odd"][1]),
                bls["masked_depth"][0],
            )
        )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    TESSTarget(33419790)
