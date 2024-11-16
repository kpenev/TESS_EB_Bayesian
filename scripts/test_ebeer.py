#!/usr/bin/env python3

"""A collection of tests comparing ebeer to PHOEBE."""

from itertools import product

from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
from matplotlib import pyplot
import numpy
import phoebe

from binary_parameters import BinaryParams
from ebeer import fit_ebeer_time_and_coef


def parse_command_line():
    """Return the command line configuration."""

    def add_param(arg_name, default, description):
        """Add a parameter that will be varied for tests."""

        parser.add_argument(
            "--" + arg_name,
            type=float,
            nargs="+",
            default=default,
            help=description,
        )

    parser = ArgumentParser(
        description=__doc__ + " Tests are run with all possible combinations "
        "of parameters specified.",
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    add_param(
        "mstar",
        [0.4, 0.7, 1.0],
        "Values to try for the masses of the stars. The primary and "
        "secondary star masses are independently taken from this list.",
    )
    add_param(
        "peridistance-factor",
        [3.0, 5.0, 10.0],
        "Values to try for the periapsis distance in units of the sum of "
        "the roche radii of the two stars.",
    )
    add_param(
        "ecc", [0.0, 0.3, 0.6, 0.8], "Values to try for the eccentricity."
    )
    add_param(
        "incl",
        [0.0, 30.0, 60.0, 90.0, 135.0, 180.0],
        "Values to try for the inclination in degrees.",
    )
    add_param(
        "per0",
        [0.0, 30.0, 60.0, 90.0, 135.0, 180.0, 240.0, 270.0, 315.0],
        "Values to try for the argument of periapsis in degrees.",
    )
    add_param(
        "t0-supconj-factor",
        numpy.linspace(0.0, 1.0, 5),
        "Values to try for the time of superior conjunction in units of the "
        "orbital period.",
    )
    parser.add_argument(
        "--ntimes",
        type=int,
        default=101,
        help="The number of lightcurve points to generate (evenly spaced from "
        "0 to the orbital period).",
    )
    parser.add_argument(
        "--ntriangles",
        type=int,
        default=10000,
        help="The number of triangles to tell PHOEBE to use when calculating "
        "its lightcurve.",
    )
    parser.add_argument(
        "--plots",
        default=None,
        help="If specified, a PDF file is generated with one page for each "
        "test case comparing the eBEER and PHOEBE flux changes. Probably "
        "undesirable for the full set of test cases.",
    )
    return parser.parse_args()


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
    binary.flip_constraint("mass@primary", solve_for="period")
    binary.flip_constraint("mass@secondary", solve_for="q")

    binary["passband"] = "Bolometric:900-40000"
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
    return binary


def run_tests(configuration):
    """Run the tests specified on the command line."""

    #TODO: Why is best fit time of periastron passage not as expected for first
    #test case.

    param_names = [
        "mprimary",
        "msecondary",
        "peridistance_factor",
        "ecc",
        "incl",
        "per0",
        "t0_supconj_factor",
    ]
    scenarios = product(
        *(
            [configuration.mstar, configuration.mstar]
            + [getattr(configuration, param) for param in param_names[2:]]
        )
    )
    for param_values in scenarios:
        phoebe_binary = create_phoebe_binary(
            **dict(zip(param_names, param_values))
        )
        phoebe_binary.set_value_all("ntriangles", configuration.ntriangles)
        times = numpy.linspace(
            0.0,
            phoebe_binary["orbit"]["component"]["period"].get_value(units.day),
            configuration.ntimes,
        )
        phoebe_binary["dataset"]["lc01"]["times"].set_value(times * units.day)

        phoebe_binary["irrad_method"].set_value("none")
        phoebe_binary.set_value_all("distortion_method", value="sphere")
        phoebe_binary.run_compute(overwrite=True)
        reference_flux = phoebe_binary["lc01@latest@model@fluxes"].quantity

        phoebe_binary.set_value_all("distortion_method", value="roche")
        phoebe_binary.run_compute(overwrite=True)
        phoebe_distort_only = (
            (
                phoebe_binary["lc01@latest@model@fluxes"].quantity
                - reference_flux
            )
            / reference_flux
        ).to_value() * 1e6

        ebeer_binary = BinaryParams(from_phoebe=phoebe_binary)
        ebeer_distort_only = fit_ebeer_time_and_coef(
            ebeer_binary,
            times,
            phoebe_distort_only,
            include_beaming=False,
            include_reflection=False,
        )
        print(ebeer_binary)

        pyplot.plot(times, phoebe_distort_only, "-k")
        pyplot.plot(times, ebeer_distort_only, "-r")
        pyplot.show()

        return


if __name__ == "__main__":
    run_tests(parse_command_line())
