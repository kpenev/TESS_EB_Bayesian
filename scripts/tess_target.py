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
from catalog_interface import (
    get_eb_info,
    get_catalog_period_range,
    is_in_catalog,
)


class FalsePositiveError(Exception):
    """Raised when the TLS results suggest a false positive."""


def get_bls_eclipse_mask(bls, lightcurve, which):
    """
    Filter the lightcure per best fit BLS to leave only point per mask.

    See `self.calc_lc_log_likelihood()` for what masks are supported.
    """

    if not isinstance(which, str):
        result = numpy.zeros(lightcurve["time"].size, dtype=bool)
        for entry in which:
            result |= get_bls_eclipse_mask(bls, lightcurve, entry)
        return result

    if which == "masked":
        period = bls["masked_period"][0]
        window = 1.5 * bls["masked_duration"] + 2.0 * bls["masked_period"][1]
        time = lightcurve["time"] - bls["masked_transit_time"]
    else:
        period = bls["period"][0] * (1 if which == "both" else 2)
        window = 1.5 * bls["duration"] + 2.0 * bls["period"][1]
        time = lightcurve["time"] - bls["transit_time"]
        if which == "odd":
            time -= bls["period"][0]

    window = min(0.3 * period, window)

    folded = time % period
    return numpy.minimum(folded, period - folded) < window


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
        """Return the period step for the BLS search."""

        return 0.02 / (
            30.0
            * min(
                (self._time_span[1] - self._time_span[0]),
                300,
            )
        )

    def _get_best_fit_bls(self, lightcurve, periods=None, use_catalog=True):
        """Return the best fit orbital period and time of primary transit."""

        model = BoxLeastSquares(
            lightcurve["time"],
            lightcurve["flux"],
            dy=lightcurve["flux_err"],
        )

        auto_period = periods is None

        # Comprehensive P/P2 search when using catalog
        if auto_period and use_catalog:
            cat_info = get_eb_info(self._tic_id)
            if cat_info is not None:
                return self._comprehensive_period_search(
                    lightcurve, model, cat_info
                )

        # Fallback to original logic for blind search or when catalog unavailable
        if auto_period and periods is None:
            periods = 1.0 / numpy.arange(
                1.0 / 0.4,
                1.0 / 30.0,
                -self._get_bls_period_step(),
            )
        self._logger.debug("Computing BLS for perieds:\n%s", repr(periods))

        periodogram = model.power(
            periods,
            numpy.linspace(
                min(periods[0] / 10, 0.02), min(periods[0] / 2, 0.2), 100
            ),
        )
        self._logger.debug(
            "Periodogram periods:\n%s",
            repr(periodogram.period),
        )
        best_index = numpy.argmax(periodogram.power)
        stats = model.compute_stats(
            **{
                param: getattr(periodogram, param)[best_index]
                for param in ["period", "duration", "transit_time"]
            }
        )
        TESSTarget._logger.debug(
            "Stats:\n\t%s",
            "\n\t".join(
                [f"{param}: {value}" for param, value in stats.items()]
            ),
        )
        if auto_period:
            if periodogram.duration[best_index] > 0.15:
                candidate = model.power(
                    1.0 / numpy.arange(1 / 2.0, 0.01, -0.1 / 30.0**2),
                    numpy.linspace(0.1, 1.0, 100),
                )
                candidate_best_index = numpy.argmax(candidate.power)
            elif periodogram.duration[best_index] < 0.04:
                candidate = model.power(
                    1.0 / numpy.arange(1 / 0.1, 0.01, -0.01 / 30.0**2),
                    numpy.linspace(0.01, 0.05, 100),
                )
                candidate_best_index = numpy.argmax(candidate.power)
            else:
                candidate = None

            if (
                candidate is not None
                # False positive
                # pylint: disable=possibly-used-before-assignment
                and candidate.depth[candidate_best_index]
                > periodogram.depth[best_index]
                # pylint: enable=possibly-used-before-assignment
            ):
                periodogram = candidate
                best_index = candidate_best_index

            assert best_index > 0

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
            max(
                result["period"] - periodogram.period[index_range[0]],
                periodogram.period[index_range[1]] - result["period"],
            ),
        )

        return result

    def _comprehensive_period_search(self, lightcurve, model, cat_info):
        """Search both P and P/2 periods and select best based on likelihood."""

        catalog_period = cat_info["period"]
        self._logger.debug(
            f"Testing P and P/2 for TIC {self._tic_id}, catalog P={catalog_period:.4f}d"
        )

        candidates = []
        p_range = (catalog_period * 0.95, catalog_period * 1.05)
        p2_range = (catalog_period * 0.475, catalog_period * 0.525)
        candidates.append(("P", numpy.linspace(p_range[0], p_range[1], 50)))
        candidates.append(("P/2", numpy.linspace(p2_range[0], p2_range[1], 50)))

        bls_results = {}
        for name, periods in candidates:
            periodogram = model.power(
                periods,
                numpy.linspace(
                    min(periods[0] / 10, 0.02), min(periods[0] / 2, 0.2), 100
                ),
            )

            best_index = numpy.argmax(periodogram.power)
            stats = model.compute_stats(
                **{
                    param: getattr(periodogram, param)[best_index]
                    for param in ["period", "duration", "transit_time"]
                }
            )

            result = {
                param: getattr(periodogram, param)[best_index]
                for param in ["period", "duration", "transit_time"]
            }
            result.update(stats)
            result["bls_power"] = periodogram.power[best_index]

            # Debug: print available keys
            self._logger.debug(
                f"Available stats keys for {name}: {list(stats.keys())}"
            )

            # Use the correct likelihood key from astropy BLS
            if "log_likelihood" in stats:
                result["log_likelihood"] = stats["log_likelihood"]
            elif "per_transit_log_likelihood" in stats:
                # Use the average log likelihood across transits
                result["log_likelihood"] = numpy.mean(
                    stats["per_transit_log_likelihood"]
                )
            else:
                # Fallback to using BLS power as a proxy
                result["log_likelihood"] = float(periodogram.power[best_index])

            result["candidate_type"] = name
            bls_results[name] = result

        best_candidate = self._select_best_period_candidate(
            bls_results, catalog_period
        )

        if best_candidate["candidate_type"] == "P/2":
            best_candidate = self._analyze_p2_eclipses(
                lightcurve, model, best_candidate, catalog_period
            )

        result = {
            "period": best_candidate["period"],
            "duration": best_candidate["duration"],
            "transit_time": best_candidate["transit_time"],
            "depth": best_candidate["depth"],
            "depth_even": best_candidate.get(
                "depth_even", best_candidate["depth"]
            ),
            "depth_odd": best_candidate.get(
                "depth_odd", best_candidate["depth"]
            ),
            "log_likelihood": best_candidate["log_likelihood"],
            "candidate_type": best_candidate["candidate_type"],
        }

        period_step = self._get_bls_period_step()
        result["period"] = (result["period"], period_step)

        for key in [
            "transit_times",
            "per_transit_count",
            "per_transit_log_likelihood",
        ]:
            if key in result:
                del result[key]

        return result

    def _select_best_period_candidate(self, bls_results, catalog_period):
        """Select best candidate using likelihood, with catalog consistency for tie-breaking."""

        candidates = list(bls_results.values())
        best_by_likelihood = max(candidates, key=lambda x: x["log_likelihood"])

        likelihood_diff = best_by_likelihood["log_likelihood"] - min(
            c["log_likelihood"] for c in candidates
        )

        if likelihood_diff < 5.0:
            catalog_diffs = []
            for candidate in candidates:
                if candidate["candidate_type"] == "P":
                    diff = (
                        abs(candidate["period"] - catalog_period)
                        / catalog_period
                    )
                else:
                    diff = (
                        abs(candidate["period"] * 2 - catalog_period)
                        / catalog_period
                    )
                catalog_diffs.append((diff, candidate))

            best_by_catalog = min(catalog_diffs, key=lambda x: x[0])[1]
            self._logger.debug(
                f"Close likelihoods, using catalog consistency: {best_by_catalog['candidate_type']}"
            )
            return best_by_catalog

        self._logger.debug(
            f"Clear likelihood winner: {best_by_likelihood['candidate_type']}"
        )
        return best_by_likelihood

    def _analyze_p2_eclipses(
        self, lightcurve, model, p2_result, catalog_period
    ):
        """Separate even/odd eclipses for P/2 candidate and identify primary/secondary."""

        folded_time = (
            lightcurve["time"] - p2_result["transit_time"]
        ) % p2_result["period"]
        even_mask = folded_time < p2_result["period"] / 2
        odd_mask = ~even_mask

        even_depth = numpy.nanmedian(lightcurve["flux"][even_mask])
        odd_depth = numpy.nanmedian(lightcurve["flux"][odd_mask])
        baseline = numpy.nanmedian(lightcurve["flux"])

        even_depth_pf = (baseline - even_depth) / baseline
        odd_depth_pf = (baseline - odd_depth) / baseline

        if even_depth_pf > odd_depth_pf:
            primary_depth = even_depth_pf
            primary_type = "even"
        else:
            primary_depth = odd_depth_pf
            primary_type = "odd"

        result = p2_result.copy()
        result["period"] = catalog_period
        result["depth"] = primary_depth
        result["depth_even"] = even_depth_pf
        result["depth_odd"] = odd_depth_pf
        result["primary_eclipse"] = primary_type

        return result

    def _get_masked_best_fit_bls(self, lightcurve, bls_results):
        """Mask the eclispes detected by given BLS and fit BLS again."""

        mask = numpy.logical_not(
            get_bls_eclipse_mask(bls_results, lightcurve, "both")
        )
        self._logger.debug("After masking, %d points remain", mask.sum())
        num_periods = max(
            10,
            int(
                (11.0 * bls_results["period"][1]) // self._get_bls_period_step()
            ),
        )
        period_range = (
            max(bls_results["period"][0] - 5 * bls_results["period"][1], 0.1),
            min(bls_results["period"][0] + 5 * bls_results["period"][1], 100),
        )
        self._logger.debug(
            "Covering period range %s < Porb < %s with %d points",
            *period_range,
            num_periods,
        )

        masked_bls_result = self._get_best_fit_bls(
            lightcurve[mask],
            periods=numpy.linspace(
                *period_range,
                num_periods,
            ),
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


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    TESSTarget(33419790)
