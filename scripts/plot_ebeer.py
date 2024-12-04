#!/usr/bin/env python3

"""Plot everything eBEER calculates for a given binary."""

from itertools import product
import pickle

from matplotlib import pyplot, use
import phoebe
from astropy import units
import numpy

from binary_parameters import BinaryParams

# False positive due to calling from string name
# pylint: disable=unused-import
from ebeer import (
    fit_ebeer_time_and_coef,
    calc_true_anomaly,
    ellipticity,
    reflection,
    beaming,
)

# pylint: enable=unused-import

_curves = list(
    product(
        ["primary", "secondary"],
        ["ellipticity", "reflection"],
    )
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

    binary = phoebe.default_binary()
    binary.add_dataset("lc", times=0, label="lc01")
    binary.add_dataset("rv")
    binary.flip_constraint("mass@primary", solve_for="period")
    binary.flip_constraint("mass@secondary", solve_for="q")

    binary["lc"]["passband"] = "Bolometric:900-40000"
    binary.set_value_all("ld_func*", "linear")
    binary["eclipse_method"].set_value("only_horizon")
    binary.set_value_all("atm", "phoenix")

    orbit = binary["orbit"]["component"]
    for param_name in ["ecc", "incl", "per0"]:
        orbit[param_name] = parameters[param_name]

    set_star_params(binary["primary"]["component"], parameters["mprimary"])
    set_star_params(binary["secondary"]["component"], parameters["msecondary"])
    roche_total = (
        binary["primary"]["component"]["requiv_max"].quantity
        + binary["secondary"]["component"]["requiv_max"].quantity
    )
    orbit["sma"].set_value(
        parameters["peridistance_factor"]
        * roche_total
        / (1.0 - parameters["ecc"])
    )

    orbit["t0_supconj"] = (
        orbit["period"].quantity * parameters["t0_supconj_factor"]
    )
    binary["primary"]["rv_method"] = "dynamical"
    binary["secondary"]["rv_method"] = "dynamical"

    return binary


def get_flux_modulations(phoebe_binary, times, *fit_args, **fit_kwargs):
    """
    Return best-fit eBEER flux modulation components for the given binary.
    """

    binary = BinaryParams(from_phoebe=phoebe_binary)
    fit_ebeer_time_and_coef(binary, times, *fit_args, **fit_kwargs)
    print(f"LC Best fit binary: {binary}")
    true_anomaly = calc_true_anomaly(binary, times)
    reflection_coef = numpy.zeros(2)
    result = {}
    secondary_flux_fraction = binary.secondary_flux_fraction()
    for component, modulation in _curves:
        if modulation not in result:
            result[modulation] = {}
        component_i = 0 if component == "primary" else 1
        if fit_kwargs.get("include_" + modulation, True):
            result[modulation][component] = (
                globals()[modulation](binary, true_anomaly)
                * (
                    1
                    if modulation == "ellipticity"
                    else getattr(binary, modulation + "_coef")
                )
                * (secondary_flux_fraction if component == "secondary" else 1)
                / (1.0 + secondary_flux_fraction)
            )
        if modulation == "reflection":
            reflection_coef[component_i] = binary.reflection_coef
            binary.swap_components()
    everything = None
    for modulation in result.values():
        if modulation:
            modulation["combined"] = (
                modulation["primary"] + modulation["secondary"]
            )
            if everything is None:
                everything = {
                    component: numpy.copy(modulation[component])
                    for component in ["primary", "secondary", "combined"]
                }
            else:
                for component in ["primary", "secondary", "combined"]:
                    everything[component] += modulation[component]
    result["everything"] = everything
    result["reflection_coef"] = reflection_coef
    result['t0_perpass'] = binary.t0_perpass
    return result


def get_rvs(phoebe_binary, times, rvs_to_fit, *fit_args, **fit_kwargs):
    """Return best-fit eBEER beaming modulation per component to given RVs."""

    binary = BinaryParams(from_phoebe=phoebe_binary)
    result = {
        't0_perpass': numpy.zeros(2)
    }

    fit_kwargs["include_beaming"] = True
    fit_kwargs["include_reflection"] = False
    fit_kwargs["include_ellipticity"] = False
    teff_ratio = binary.teff_ratio
    binary.teff_ratio = 0.0

    fit_ebeer_time_and_coef(
        binary, times, rvs_to_fit["primary"], *fit_args, **fit_kwargs
    )
    print(f"RV Best fit binary (primary): {binary}")
    result['t0_perpass'][0] = binary.t0_perpass

    true_anomaly = calc_true_anomaly(binary, times)

    primary_beaming_coef = binary.beaming_coef
    result["primary"] = primary_beaming_coef * beaming(binary, true_anomaly)

    binary.teff_ratio = teff_ratio
    binary.swap_components()
    binary.teff_ratio = 0.0
    fit_ebeer_time_and_coef(
        binary, times, rvs_to_fit["secondary"], *fit_args, **fit_kwargs
    )
    print(f"RV Best fit binary (secondary): {binary}")

    binary.teff_ratio = 1.0 / teff_ratio
    binary.swap_components()
    secondary_beaming_coef = binary.beaming_coef
    result['t0_perpass'][1] = binary.t0_perpass
    binary.swap_components()
    result["secondary"] = secondary_beaming_coef * beaming(binary, true_anomaly)

    result["beaming_coef"] = numpy.array(
        [primary_beaming_coef, secondary_beaming_coef]
    )

    return result


def plot_modulations(times, flux_modulations, label_fmt, **plot_kwargs):
    """Plot flux modulations calculated using `get_flux_modulations()`."""

    for component in ("primary", "secondary", "combined"):
        if component not in flux_modulations:
            continue
        pyplot.plot(
            times,
            flux_modulations[component],
            label=label_fmt.format(component=component),
            **plot_kwargs.get(component, {}),
        )


def main():
    """Avoid polluting global scope."""

    use("PDF")
    plot_data = {}
    with open("ebeer_test_data.pkl", "rb") as pickle_f:
        try:
            while True:
                assert pickle.load(pickle_f) == "START RECORD"
                param_names = pickle.load(pickle_f)
                param_values = tuple(pickle.load(pickle_f))
                plot_data[param_values] = pickle.load(pickle_f)
                assert pickle.load(pickle_f) == "END RECORD"
        except EOFError:
            pass

    parameters = {
        "mprimary": 0.4,
        "msecondary": 0.4,
        "peridistance_factor": 3.0,
        "ecc": 0.0,
        "incl": 30.0,
        "per0": 0.0,
        "t0_supconj_factor": 0.0,
    }
    param_values = tuple(parameters[name] for name in param_names)
    times, _, phoebe_flux = plot_data[param_values]

    pyplot.plot(times, phoebe_flux["everything"], label="target")

    plot_colors = {
        "ellipticity": "green",
        "reflection": "red",
        "beaming": "blue",
        "everything": "black",
    }

    for modulation, to_plot in get_flux_modulations(
        create_phoebe_binary(**parameters),
        times,
        fluxdiff_to_fit=phoebe_flux["everything"],
        include_beaming=False,
        include_reflection=True,
        include_ellipticity=True,
    ).items():
        if to_plot:
            plot_modulations(
                times,
                to_plot,
                f"{{component}} {modulation}",
                primary={"linestyle": "--", "color": plot_colors[modulation]},
                secondary={"linestyle": ":", "color": plot_colors[modulation]},
                combined={"linestyle": "-", "color": plot_colors[modulation]},
            )

    #    parameters["per0"] = 90.0
    #    plot(
    #        *get_flux_modulations(
    #            parameters,
    #            times,
    #            fluxdiff_to_fit=phoebe_flux["ellipticity"],
    #            include_beaming=False,
    #            include_reflection=True,
    #            include_ellipticity=True,
    #        )
    #    )

    pyplot.xlabel("Time [days]")
    pyplot.ylabel(r"$\Delta F$")
    pyplot.figlegend()
    pyplot.savefig("ebeer_breakdown.pdf")


if __name__ == "__main__":
    main()
