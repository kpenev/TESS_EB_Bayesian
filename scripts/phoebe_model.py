"""Calculate lightcurve models using PHOEBE for testing purposes."""

import phoebe
from astropy import units


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
