"""Calculate lightcurve models using PHOEBE for testing purposes."""

from os import path

import numpy
import phoebe
from astropy import units
from astropy.io import fits

from sample_params import (
    SampleParams,
    param_descriptions,
    param_units,
    param_fits_keys,
)


def set_star_params(star, mass):
    """Set stellar properties per Eq. 8-10 of eBEER paper."""

    if mass < 0.43:
        luminosity = 0.23 * mass**2.3
    elif mass < 2.0:
        luminosity = mass**4
    else:
        luminosity = 1.5 * mass**3.5
    radius = max(0.1, mass**0.8)
    star["mass"].set_value(mass * units.Msun)
    star["requiv"].set_value(radius * units.Rsun)
    star["teff"].set_value(6000.0 * luminosity**0.25 / radius**0.5)
    star["ld_func_bol"].set_value("linear")
    star["ld_mode_bol"].set_value("lookup")


def create_phoebe_binary(**parameters):
    """Create a PHOEBE binary given values for all parameters from cmdline."""

    result = phoebe.default_binary()
    result.add_dataset("lc", times=0, label="lc01")
    result.add_dataset("rv")
    result.flip_constraint("mass@primary", solve_for="period")
    result.flip_constraint("mass@secondary", solve_for="q")

    result["lc"]["passband"] = "Bolometric:900-40000"
    result.set_value_all("ld_func*", "linear")
    result.set_value_all("atm", "phoenix")

    orbit = result["orbit"]["component"]
    for param_name in ["ecc", "incl", "per0"]:
        orbit[param_name] = parameters[param_name]

    set_star_params(result["primary"]["component"], parameters["mprimary"])
    set_star_params(result["secondary"]["component"], parameters["msecondary"])
    roche_total = (
        result["primary"]["component"]["requiv_max"].quantity
        + result["secondary"]["component"]["requiv_max"].quantity
    )
    orbit["sma"].set_value(
        parameters["peridistance_factor"]
        * roche_total
        / (1.0 - parameters["ecc"])
    )

    orbit["t0_supconj"] = (
        orbit["period"].quantity * parameters["t0_supconj_factor"]
    )
    result["primary"]["rv_method"] = "dynamical"
    result["secondary"]["rv_method"] = "dynamical"

    return result


def make_mock_tess_lc(
    phoebe_binary,
    num_sectors=1,
    start_time=1325.3,
    flux_err_frac=1e-3,
    exclude=(),
    sector_gap=None,
):
    """
    Generate a mock TESS lightcurve from a PHOEBE binary.

    Computes the PHOEBE model flux at TESS 2-minute cadence times, normalizes
    by the reference flux (all effects off), and returns lightcurve data in the
    same format used internally by TESSTarget: a list of
    ``(header_dict, structured_array)`` tuples, one per sector.

    Args:
        phoebe_binary:  A fully configured ``phoebe.Bundle`` (e.g. from
            ``create_phoebe_binary()``). The lc dataset ``"lc01"`` must already
            exist. The bundle's times will be overwritten.

        num_sectors:    How many 27.4-day TESS sectors to simulate.

        start_time:     Start time in TESS BJD (BJD - 2457000). Default
            corresponds roughly to TESS sector 1.

        flux_err_frac:  Fractional flux uncertainty assigned to each point
            (constant). Set to 0 to get a noise-free lightcurve.

        exclude:        Tuple of effects to disable when computing the PHOEBE
            model. Supported values: ``"reflection"``, ``"ellipticity"``,
            ``"eclipse"``.

        sector_gap:     Gap in days between sectors. If None, defaults to the
            standard ~2-day gap.

    Returns:
        list:   A list of ``(header, lightcurve)`` tuples matching the format of
            ``TESSTarget.lcs``.  Each ``header`` is a dict with keys
            ``"exptime"`` (float, days), ``"sector"`` (int), and
            ``"provenance"`` (str).  Each ``lightcurve`` is a numpy structured
            array with dtype
            ``[("time", ">f8"), ("flux", ">f4"), ("flux_err", ">f4"),
            ("good", "bool")]``.
    """

    sector_duration = 27.4
    cadence_days = 2.0 / (24.0 * 60.0)
    if sector_gap is None:
        sector_gap = 2.0

    all_times = []
    sector_slices = []
    offset = 0
    for sector_index in range(num_sectors):
        t_start = start_time + sector_index * (sector_duration + sector_gap)
        times = numpy.arange(t_start, t_start + sector_duration, cadence_days)
        all_times.append(times)
        sector_slices.append(slice(offset, offset + len(times)))
        offset += len(times)
    combined_times = numpy.concatenate(all_times)

    phoebe_binary.set_value_all("times", combined_times * units.day)
    reference_flux = get_phoebe_reference_flux(phoebe_binary)

    phoebe_binary["irrad_method"].set_value(
        "none" if "reflection" in exclude else "horvat"
    )
    phoebe_binary.set_value_all(
        "distortion_method",
        value="sphere" if "ellipticity" in exclude else "roche",
    )
    phoebe_binary["eclipse_method"].set_value(
        "only_horizon" if "eclipse" in exclude else "native"
    )
    phoebe_binary.run_compute(overwrite=True)
    all_fluxes = (
        phoebe_binary["lc01@latest@model@fluxes"].quantity / reference_flux
    ).to_value()

    result = []
    for sector_index in range(num_sectors):
        sl = sector_slices[sector_index]
        times = combined_times[sl]
        fluxes = all_fluxes[sl]

        lightcurve = numpy.empty(
            len(times),
            dtype=[
                ("time", ">f8"),
                ("flux", ">f4"),
                ("flux_err", ">f4"),
                ("good", "bool"),
            ],
        )
        lightcurve["time"] = times
        lightcurve["flux"] = fluxes
        lightcurve["flux_err"] = flux_err_frac
        if flux_err_frac > 0:
            lightcurve["flux"] += numpy.random.normal(
                scale=flux_err_frac, size=len(times)
            ).astype(">f4")
        lightcurve["good"] = True

        header = {
            "exptime": cadence_days,
            "sector": sector_index + 1,
            "provenance": "MOCK",
        }
        result.append((header, lightcurve))

    return result


def save_mock_tess_lc(mock_lcs, output_dir, sample_params, tic_id=0):
    """
    Save mock TESS lightcurves to FITS files readable by ``get_astroquery``.

    Writes one FITS file per sector in the SPOC naming convention, with the
    binary table and header keywords that ``TESSTarget._format_lc`` expects.
    The ``SampleParams`` used to generate the lightcurve are recorded in the
    primary header of every file.

    Args:
        mock_lcs:       The list of ``(header, lightcurve)`` tuples as returned
            by ``make_mock_tess_lc()``.

        output_dir:     Directory in which to write the FITS files.

        sample_params:  A ``SampleParams`` instance with the parameter values
            the mock lightcurve was generated for.

        tic_id:         TIC identifier to embed in the filename and primary
            header. Defaults to 0.

    Returns:
        list:   Paths to the written FITS files.
    """

    cadence_sec = mock_lcs[0][0]["exptime"] * 86400.0

    written = []
    for header, lightcurve in mock_lcs:
        sector = header["sector"]

        col_time = fits.Column(
            name="TIME", format="D", array=lightcurve["time"]
        )
        col_flux = fits.Column(
            name="PDCSAP_FLUX", format="E", array=lightcurve["flux"]
        )
        col_err = fits.Column(
            name="PDCSAP_FLUX_ERR",
            format="E",
            array=lightcurve["flux_err"],
        )
        col_quality = fits.Column(
            name="QUALITY",
            format="J",
            array=numpy.zeros(len(lightcurve), dtype=numpy.int32),
        )

        table_hdu = fits.BinTableHDU.from_columns(
            [col_time, col_flux, col_err, col_quality]
        )
        table_hdu.header["INT_TIME"] = (
            cadence_sec,
            "Integration time per frame [s]",
        )
        table_hdu.header["NUM_FRM"] = (1, "Number of frames per exposure")
        table_hdu.header["TIMEPIXR"] = (0.5, "Bin time beginning=0, mid=0.5")
        table_hdu.header["TIMEDEL"] = (
            header["exptime"],
            "Time resolution of data [day]",
        )

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["TICID"] = (tic_id, "TIC identifier")
        primary_hdu.header["SECTOR"] = (sector, "TESS sector number")

        for field in SampleParams._fields:
            fits_key = param_fits_keys[field]
            comment = param_descriptions[field]
            unit_str = str(param_units[field])
            if unit_str and unit_str != "":
                comment += f" [{unit_str}]"
            primary_hdu.header[fits_key] = (
                float(getattr(sample_params, field)),
                comment,
            )

        hdu_list = fits.HDUList([primary_hdu, table_hdu])
        fname = (
            f"mock_tess_s{sector:04d}-{tic_id:016d}_lc.fits"
        )
        out_path = path.join(output_dir, fname)
        hdu_list.writeto(out_path, overwrite=True)
        written.append(out_path)

    return written


def read_mock_tess_lc(fits_fnames):
    """
    Read mock TESS lightcurves previously saved by ``save_mock_tess_lc``.

    Args:
        fits_fnames:    A list of paths to FITS files (one per sector) as
            returned by ``save_mock_tess_lc``.

    Returns:
        tuple:  ``(mock_lcs, sample_params)`` where ``mock_lcs`` is a list of
            ``(header, lightcurve)`` tuples in the same format as
            ``TESSTarget.lcs`` and ``sample_params`` is the ``SampleParams``
            reconstructed from the primary header of the first file.
    """

    fits_key_to_field = {v: k for k, v in param_fits_keys.items()}

    sample_params = None
    mock_lcs = []
    for fname in fits_fnames:
        with fits.open(fname, "readonly") as hdu_list:
            primary_header = hdu_list[0].header
            table_header = hdu_list[1].header
            table_data = hdu_list[1].data

            if sample_params is None:
                param_values = {}
                for fits_key, field in fits_key_to_field.items():
                    param_values[field] = primary_header[fits_key]
                sample_params = SampleParams(**param_values)

            lightcurve = numpy.empty(
                len(table_data),
                dtype=[
                    ("time", ">f8"),
                    ("flux", ">f4"),
                    ("flux_err", ">f4"),
                    ("good", "bool"),
                ],
            )
            lightcurve["time"] = table_data["TIME"]
            lightcurve["flux"] = table_data["PDCSAP_FLUX"]
            lightcurve["flux_err"] = table_data["PDCSAP_FLUX_ERR"]
            lightcurve["good"] = table_data["QUALITY"] == 0

            header = {
                "exptime": table_header["INT_TIME"]
                * table_header["NUM_FRM"]
                / 86400.0,
                "sector": primary_header["SECTOR"],
                "provenance": "MOCK",
            }
            mock_lcs.append((header, lightcurve))

    mock_lcs.sort(key=lambda x: x[0]["sector"])
    return mock_lcs, sample_params


def get_phoebe_fluxmod(phoebe_binary, reference_flux):
    """Return flux modulation, given reference, for fully configured binary."""

    phoebe_binary.run_compute(overwrite=True)
    return (
        (phoebe_binary["lc01@latest@model@fluxes"].quantity - reference_flux)
        / reference_flux
    ).to_value()


def get_phoebe_reference_flux(phoebe_binary):
    """Return the reference flux (with all effects off) for given binary."""

    phoebe_binary["irrad_method"].set_value("none")
    phoebe_binary.set_value_all("distortion_method", value="sphere")
    phoebe_binary["eclipse_method"].set_value("only_horizon")
    phoebe_binary.run_compute(overwrite=True)
    return phoebe_binary["lc01@latest@model@fluxes"].quantity


def get_phoebe_ellipticity_mod(
    phoebe_binary, reference_flux, combined_only=False
):
    """Return the primary, secondary, combined ellipticity flux modulations."""

    phoebe_binary["irrad_method"].set_value("none")
    phoebe_binary["eclipse_method"].set_value("only_horizon")
    phoebe_binary.set_value_all("distortion_method", value="roche")
    result = {"combined": get_phoebe_fluxmod(phoebe_binary, reference_flux)}
    if combined_only:
        return result["combined"]

    phoebe_binary["secondary"]["distortion_method"] = "sphere"
    result["primary"] = get_phoebe_fluxmod(phoebe_binary, reference_flux)
    phoebe_binary["secondary"]["distortion_method"] = "roche"
    phoebe_binary["primary"]["distortion_method"] = "sphere"
    result["secondary"] = get_phoebe_fluxmod(phoebe_binary, reference_flux)

    return result


def get_phoebe_reflection_mod(
    phoebe_binary, reference_flux, combined_only=False
):
    """Return the primary, secondary, combined reflection flux modulations."""

    phoebe_binary["irrad_method"].set_value("horvat")
    phoebe_binary["eclipse_method"].set_value("only_horizon")
    phoebe_binary.set_value_all("distortion_method", value="sphere")
    reflection_frac = {
        component: phoebe_binary[component]["irrad_frac_refl_bol"].get_value()
        for component in ["primary", "secondary"]
    }

    result = {"combined": get_phoebe_fluxmod(phoebe_binary, reference_flux)}
    if combined_only:
        return result["combined"]

    phoebe_binary["secondary"]["irrad_frac_refl_bol"] = 0.0
    result["primary"] = get_phoebe_fluxmod(phoebe_binary, reference_flux)

    phoebe_binary["secondary"]["irrad_frac_refl_bol"] = reflection_frac[
        "secondary"
    ]
    phoebe_binary["primary"]["irrad_frac_refl_bol"] = 0.0
    result["secondary"] = get_phoebe_fluxmod(phoebe_binary, reference_flux)
    phoebe_binary["primary"]["irrad_frac_refl_bol"] = reflection_frac["primary"]

    return result


def get_phoebe_eclipse_mod(phoebe_binary, reference_flux, combined_only=False):
    """Return the primary, secondary, combined eclipse flux modulations."""

    phoebe_binary["eclipse_method"].set_value("native")
    phoebe_binary.set_value_all("distortion_method", value="sphere")
    phoebe_binary["irrad_method"].set_value("none")
    result = get_phoebe_fluxmod(phoebe_binary, reference_flux)
    if combined_only:
        return result
    return {"combined": result}


def get_phoebe_multieffect_mod(
    phoebe_binary, reference_flux, combined_only=False, exclude=()
):
    """Return the primary, secondary, combined total flux modulations."""

    phoebe_binary["irrad_method"].set_value(
        "none" if "reflection" in exclude else "horvat"
    )
    phoebe_binary.set_value_all(
        "distortion_method",
        value="sphere" if "ellipticity" in exclude else "roche",
    )
    phoebe_binary["eclipse_method"].set_value(
        "only_horizon" if "eclipse" in exclude else "native"
    )

    reflection_frac = {
        component: phoebe_binary[component]["irrad_frac_refl_bol"].get_value()
        for component in ["primary", "secondary"]
    }

    result = {"combined": get_phoebe_fluxmod(phoebe_binary, reference_flux)}
    if combined_only:
        return result["combined"]

    phoebe_binary["secondary"]["distortion_method"] = "sphere"
    phoebe_binary["secondary"]["irrad_frac_refl_bol"] = 0.0
    result["primary"] = get_phoebe_fluxmod(phoebe_binary, reference_flux)

    phoebe_binary["secondary"]["distortion_method"] = "roche"
    phoebe_binary["secondary"]["irrad_frac_refl_bol"] = reflection_frac[
        "secondary"
    ]
    phoebe_binary["primary"]["distortion_method"] = "sphere"
    phoebe_binary["primary"]["irrad_frac_refl_bol"] = 0.0
    result["secondary"] = get_phoebe_fluxmod(phoebe_binary, reference_flux)
    phoebe_binary["primary"]["irrad_frac_refl_bol"] = reflection_frac["primary"]

    return result
