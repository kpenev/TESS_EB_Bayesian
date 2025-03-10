"""Define a class providing a uniform interface to TESS lightcurves."""

import logging

from matplotlib import pyplot, colormaps
import numpy
from astropy.timeseries import BoxLeastSquares

from download_lcs import get_astroquery as download_lcs


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

    def _get_lc_format(self, sector, header, provenance):
        """Return the relevant column names and exposure time for gvien LC."""

        if provenance == "QLP":
            for existing_header, _ in self._lcs:
                if existing_header["sector"] == sector:
                    return None, None

            return ("KSPSAP_FLUX", "KSPSAP_FLUX_ERR"), header["TIMEDEL"]

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
        usable = numpy.logical_and(
            numpy.isfinite(observed_lc[flux_columns[0]]),
            numpy.isfinite(observed_lc[flux_columns[1]]),
        )
        usable = numpy.logical_and(
            usable,
            numpy.logical_not(observed_lc["QUALITY"] & self._bad_mask),
        )
        observed_lc = observed_lc[usable]
        formatted_lc = numpy.empty(
            usable.sum(),
            dtype=[
                ("time", ">f8"),
                ("flux", ">f4"),
                ("flux_err", ">f4"),
            ],
        )
        formatted_lc["time"] = observed_lc["TIME"]
        formatted_lc["flux"] = observed_lc[flux_columns[0]]
        formatted_lc["flux_err"] = observed_lc[flux_columns[1]]
        return formatted_lc, {
            "exptime": exptime,
            "sector": sector,
            "provenance": provenance,
        }

    @property
    def lcs(self):
        """The lightcurves and minimal metadata for this target by sector."""

        return self._lcs

    @staticmethod
    def _get_best_fit_bls(lightcurve):
        """Return the best fit orbital period and time of primary transit."""

        model = BoxLeastSquares(
            lightcurve["time"],
            lightcurve["flux"],
            dy=lightcurve["flux_err"],
        )

        periodogram = model.power(
            1.0
            / numpy.arange(
                1 / 0.4,
                0.01,
                -0.02
                / (
                    100.0
                    * (lightcurve["time"].max() - lightcurve["time"].min())
                ),
            ),
            numpy.linspace(0.02, 0.2, 100),
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
            and candidate.depth[candidate_best_index]
            > periodogram.depth[best_index]
        ):
            periodogram = candidate
            best_index = candidate_best_index

        index_range = [0, periodogram.power.size - 1]
        cutoff = 0.3 * periodogram.power[best_index]
        if periodogram.power[:best_index].min() < cutoff:
            index_range[0] = numpy.where(
                periodogram.power[:best_index] < cutoff
            )[0][-1]
        if periodogram.power[best_index:].min() < cutoff:
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

        assert index_range[0] < best_index
        assert index_range[1] > best_index

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

    def _get_masked_best_fit_bls(self, lightcurve, bls_results):
        """Mask the eclispes detected by given BLS and fit BLS again."""

        folded = (
            lightcurve["time"] - bls_results["transit_time"]
        ) % bls_results["period"][0]
        mask = numpy.minimum(folded, bls_results["period"][0] - folded) > (
            bls_results["duration"] + 2.0 * bls_results["period"][1]
        )
        self._logger.debug("After masking, %d points remain", mask.sum())
        masked_bls_result = self._get_best_fit_bls(lightcurve[mask])
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

    def __init__(self, tic_id):
        """Download and organize the lightcurves for the given TIC ID."""

        lcs = {
            provenance: download_lcs(tic_id, "all", provenance=provenance)
            for provenance in ["SPOC", "QLP"]
        }

        self._lcs = []

        for provenance, lc_collection in lcs.items():
            for sector, (header, observed_lc) in lc_collection.items():
                formatted_lc, formatted_header = self._format_lc(
                    sector, header, provenance, observed_lc
                )
                if formatted_lc is None:
                    continue

                self._lcs.append((formatted_header, formatted_lc))

    def get_combined_lc(self):
        """Return LC combining all sectors after scaling by median."""

        combined_lc = None
        for _, formatted_lc in self.lcs:
            med_flux = numpy.nanmedian(formatted_lc["flux"])
            if combined_lc is None:
                combined_lc = numpy.copy(formatted_lc)
                combined_lc["flux"] /= med_flux
                combined_lc["flux_err"] /= med_flux
            else:
                combined_lc = numpy.concatenate([combined_lc, formatted_lc])
                # False positive
                # pylint: disable=invalid-unary-operand-type
                combined_lc[-formatted_lc.size :]["flux"] /= med_flux
                combined_lc[-formatted_lc.size :]["flux_err"] /= med_flux
                # pylint: enable=invalid-unary-operand-type
        return combined_lc

    def fit_bls(self, plot=False):
        """Fit BLS models to the given lightcurve."""

        combined_lc = self.get_combined_lc()
        best_fit_bls = self._get_best_fit_bls(combined_lc)
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
        if plot:
            self.plot_best_fit_bls(combined_lc, best_fit_bls, plot)
        return best_fit_bls
