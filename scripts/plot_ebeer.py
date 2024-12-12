#!/usr/bin/env python3

"""Plot everything eBEER calculates for a given binary."""

from itertools import product
import pickle

from matplotlib import pyplot, use
import numpy

from binary import Binary
from phoebe_model import create_phoebe_binary

_curves = list(
    product(
        ["primary", "secondary"],
        ["ellipticity", "reflection", "eclipse"],
    )
)


def get_individual_flux_modulations(binary, times, true_anomaly):
    """Return the flux modulations by effect and by star."""

    result = {"reflection_coef": numpy.zeros(2)}
    secondary_flux_fraction = binary.secondary_flux_fraction()
    for component, modulation in _curves:
        if modulation not in result:
            result[modulation] = {}
        component_i = 0 if component == "primary" else 1
        if modulation == "eclipse":
            result["eclipse"][component] = binary.eclipse(times)
        else:
            result[modulation][component] = (
                getattr(binary, modulation)(true_anomaly)
                * (secondary_flux_fraction if component == "secondary" else 1)
                / (1.0 + secondary_flux_fraction)
            )
        if modulation == "eclipse":
            # False positive
            # pylint: disable=no-member
            result["reflection_coef"][component_i] = binary.reflection_coef
            # pylint: enable=no-member
            binary.swap_components()
    return result, secondary_flux_fraction


def get_flux_modulations(phoebe_binary, times, *fit_args, **fit_kwargs):
    """
    Return best-fit eBEER flux modulation components for the given binary.
    """

    binary = Binary(from_phoebe=phoebe_binary)
    binary.fit_ebeer_coefficients(times, *fit_args, **fit_kwargs)
    print(f"LC Best fit binary: {binary}")
    true_anomaly = binary.calc_true_anomaly(times)
    result, secondary_flux_fraction = get_individual_flux_modulations(
        binary, times, true_anomaly
    )
    result["out_of_eclipse"] = {
        component: numpy.zeros(times.shape)
        for component in ["primary", "secondary", "combined"]
    }
    for effect in ["ellipticity", "reflection"]:
        if result[effect]:
            result[effect]["combined"] = (
                result[effect]["primary"] + result[effect]["secondary"]
            )
            for component in ["primary", "secondary", "combined"]:
                result["out_of_eclipse"][component] += result[effect][component]

    result["eclipse"]["combined"] = (
        result["eclipse"]["primary"]
        + secondary_flux_fraction * result["eclipse"]["secondary"]
    ) / (1.0 + secondary_flux_fraction)

    result["t0_perpass"] = binary.t0_perpass % binary.per

    result["everything"] = {}
    for component in ["primary", "secondary"]:
        result["everything"][component] = (
            (1.0 if component == "primary" else secondary_flux_fraction)
            / (1.0 + secondary_flux_fraction)
            + result["out_of_eclipse"][component]
        ) * result["eclipse"][component]
    result["everything"]["combined"] = (
        result["everything"]["primary"] + result["everything"]["secondary"]
    )
    for modulation in result["everything"].values():
        modulation -= 1.0
    return result


def get_rvs(phoebe_binary, times, rvs_to_fit, *fit_args, **fit_kwargs):
    """Return best-fit eBEER beaming modulation per component to given RVs."""

    binary = Binary(from_phoebe=phoebe_binary)
    result = {"t0_perpass": numpy.zeros(2)}

    fit_kwargs["exclude"] = ("reflection", "ellipticity", "eclipse")
    teff_ratio = binary.teff_ratio
    binary.teff_ratio = 0.0

    binary.fit_ebeer_coefficients(
        times, rvs_to_fit["primary"], *fit_args, **fit_kwargs
    )
    print(f"RV Best fit binary (primary): {binary}")
    result["t0_perpass"][0] = binary.t0_perpass % binary.per

    true_anomaly = binary.calc_true_anomaly(times)

    # False positive
    # pylint: disable=no-member
    primary_beaming_coef = binary.beaming_coef
    # pylint: enable=no-member
    result["primary"] = binary.beaming(true_anomaly)

    binary.teff_ratio = teff_ratio
    binary.swap_components()
    binary.teff_ratio = 0.0
    binary.fit_ebeer_coefficients(
        times, rvs_to_fit["secondary"], *fit_args, **fit_kwargs
    )
    print(f"RV Best fit binary (secondary): {binary}")
    # False positive
    # pylint: disable=no-member
    secondary_beaming_coef = binary.beaming_coef
    # pylint: enable=no-member
    result["secondary"] = binary.beaming(true_anomaly)

    binary.teff_ratio = 1.0 / teff_ratio
    binary.swap_components()
    result["t0_perpass"][1] = binary.t0_perpass % binary.per

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
        fluxdiff_to_fit=phoebe_flux["everything"] + 1,
        exclude=("beaming", "eclipse"),
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

    pyplot.xlabel("Time [days]")
    pyplot.ylabel(r"$\Delta F$")
    pyplot.figlegend()
    pyplot.savefig("ebeer_breakdown.pdf")


if __name__ == "__main__":
    main()
