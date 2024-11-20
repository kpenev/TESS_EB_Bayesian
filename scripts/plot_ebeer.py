#!/usr/bin/env python3

"""Plot everything eBEER calculates for a given binary."""

from itertools import product
import pickle

from matplotlib import pyplot
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
from test_ebeer import create_phoebe_binary

_curves = list(
    product(["primary", "secondary"], ["ellipticity", "reflection", "beaming"])
)


def get_flux_modulations(parameters, times, *fit_args, **fit_kwargs):
    """
    Return best-fit eBEER flux modulation components for the given parameters.
    """

    binary = BinaryParams(from_phoebe=create_phoebe_binary(**parameters))
    fit_ebeer_time_and_coef(binary, times, *fit_args, **fit_kwargs)
    print(f'iest fit binary: {binary}')
    true_anomaly = calc_true_anomaly(binary, times)
    flux_mod = {"primary": {}, "secondary": {}}
    secondary_flux_fraction = binary.secondary_flux_fraction()
    for component, modulation in _curves:
        component_i = 0 if component == "primary" else 1
        if fit_kwargs.get("include_" + modulation, True):
            flux_mod[component][modulation] = (
                globals()[modulation](binary, true_anomaly)
                * (
                    1
                    if modulation == "ellipticity"
                    else getattr(binary, modulation + "_coef")[component_i]
                )
                * (secondary_flux_fraction if component == "secondary" else 1)
                / (1.0 + secondary_flux_fraction)
            )
        if modulation == "beaming":
            binary.swap_components()
    return times, flux_mod


def plot(times, flux_modulations):
    """Plot flux modulations calculated using `get_flux_modulations()`."""

    for component, modulation in _curves:
        if modulation in flux_modulations[component]:
            pyplot.plot(
                times,
                flux_modulations[component][modulation],
                label=f"{component} {modulation}",
            )


if __name__ == "__main__":
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
        "peridistance_factor": 5.0,
        "ecc": 0.0,
        "incl": 135.0,
        "per0": 270.0,
        "t0_supconj_factor": 0.75,
    }
    param_values = tuple(parameters[name] for name in param_names)
    times, _, phoebe_flux = plot_data[param_values]
    plot(
        *get_flux_modulations(
            parameters,
            times,
            fluxdiff_to_fit=phoebe_flux["ellipticity"],
            include_beaming=False,
            include_reflection=True,
            include_ellipticity=True,
        )
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
    pyplot.legend()
    pyplot.show()
