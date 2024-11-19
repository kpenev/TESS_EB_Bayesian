#!/usr/bin/env python3

"""A collection of tests comparing ebeer to PHOEBE."""

from itertools import product
from multiprocessing import Pool, Manager
from os import path
import pickle

from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy
import phoebe

from binary_parameters import BinaryParams
from ebeer import fit_ebeer_time_and_coef

# TODO: Suspcious behavior:
# * Difference in ellipticity modulation when primary and secondary are swapped
#   for:
#   - mprimary: 0.7
#   - msecondary: 0.4
#   - peridistance_factor: 3.0
#   - ecc: 0.4
#   - incl: 0.0
#   - per0: 0.0
#   - t0_supconj_factor: 0.0
#   AND
#   - mprimary: 0.4
#   - msecondary: 0.7
#   - peridistance_factor: 5.0
#   - ecc: 0.4
#   - incl: 0.0
#   - per0: 0.0
#   - t0_supconj_factor: 0.0


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
    add_param("ecc", [0.0, 0.4, 0.8], "Values to try for the eccentricity.")
    add_param(
        "incl",
        [0.0, 30.0, 60.0, 90.0, 135.0, 180.0],
        "Values to try for the inclination in degrees.",
    )
    add_param(
        "per0",
        [0.0, 30.0, 90.0, 135.0, 180.0, 240.0, 270.0],
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
        default=301,
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
    parser.add_argument(
        "--nthreads",
        default=16,
        help="The number of parallel threads to use to do the calculations.",
    )
    parser.add_argument(
        "--check-progress",
        action="store_true",
        help="If passed, just counts how many scenarios are stored in the "
        "pickle and plots those.",
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


def get_phoebe_fluxmod(phoebe_binary, reference_flux):
    """Return flux modulation, given reference, for fully configured binary."""

    phoebe_binary.run_compute(overwrite=True)
    return (
        (phoebe_binary["lc01@latest@model@fluxes"].quantity - reference_flux)
        / reference_flux
    ).to_value() * 1e6


class CalculateScenario:
    """Callable that calculates eBEER and PHOEBE models given binary params."""

    def __init__(self, param_names, ntriangles, ntimes, pool_manager):
        """Prepare."""

        self._param_names = param_names
        self._ntriangles = ntriangles
        self._ntimes = ntimes
        self._pickle_lock = pool_manager.Lock()
        self.plot_data = pool_manager.dict()
        if path.exists("ebeer_test_data.pkl"):
            with open("ebeer_test_data.pkl", "rb") as pickle_f:
                try:
                    while True:
                        assert pickle.load(pickle_f) == "START RECORD"
                        assert pickle.load(pickle_f) == self._param_names
                        param_values = tuple(pickle.load(pickle_f))
                        print(f"Unpickling {param_values!r}")
                        self.plot_data[param_values] = pickle.load(pickle_f)
                        assert pickle.load(pickle_f) == "END RECORD"
                except EOFError:
                    pass

    def __call__(self, param_values):
        """Return the eBEER and PHOEBE flux modifications for parameters."""

        if tuple(param_values) in self.plot_data:
            print(f"Skipping pre-computed: {param_values!r}")
            return

        parameters = dict(zip(self._param_names, param_values))
        phoebe_binary = create_phoebe_binary(**parameters)
        # print(f'Phoebe binary:\n{phoebe_binary}')
        phoebe_binary.set_value_all("ntriangles", self._ntriangles)
        times = numpy.linspace(
            0.0,
            phoebe_binary["orbit"]["component"]["period"].get_value(units.day),
            self._ntimes,
        )
        phoebe_binary["dataset"]["lc01"]["times"].set_value(times * units.day)

        phoebe_binary["irrad_method"].set_value("none")
        phoebe_binary.set_value_all("distortion_method", value="sphere")
        phoebe_binary.run_compute(overwrite=True)
        reference_flux = phoebe_binary["lc01@latest@model@fluxes"].quantity

        phoebe_binary.set_value_all("distortion_method", value="roche")
        phoebe_flux = {
            "ellipticity": get_phoebe_fluxmod(phoebe_binary, reference_flux),
        }
        phoebe_binary.set_value_all("distortion_method", value="sphere")
        phoebe_binary["irrad_method"].set_value("horvat")
        phoebe_flux["reflection"] = get_phoebe_fluxmod(
            phoebe_binary, reference_flux
        )
        phoebe_binary.set_value_all("distortion_method", value="roche")
        phoebe_flux["everything"] = get_phoebe_fluxmod(
            phoebe_binary, reference_flux
        )

        ebeer_binary = BinaryParams(from_phoebe=phoebe_binary)

        ebeer_flux = {
            flux_key: fit_ebeer_time_and_coef(
                ebeer_binary,
                times,
                phoebe_flux[flux_key],
                include_beaming=False,
                include_reflection=(flux_key in ["reflection", "everything"]),
                include_ellipticity=(flux_key in ["ellipticity", "everything"]),
            )
            for flux_key in ["ellipticity", "reflection", "everything"]
        }
        result = (times, ebeer_flux, phoebe_flux)
        self.plot_data[tuple(param_values)] = result
        self._pickle_lock.acquire()
        with open("ebeer_test_data.pkl", "ab") as pickle_f:
            pickle.dump("START RECORD", pickle_f)
            pickle.dump(self._param_names, pickle_f)
            pickle.dump(param_values, pickle_f)
            pickle.dump(result, pickle_f)
            pickle.dump("END RECORD", pickle_f)
        self._pickle_lock.release()
        # print(ebeer_binary)


def run_tests(configuration):
    """Run the tests specified on the command line."""

    param_names = [
        "mprimary",
        "msecondary",
        "peridistance_factor",
        "ecc",
        "incl",
        "per0",
        "t0_supconj_factor",
    ]
    scenarios = list(
        product(
            *(
                [configuration.mstar, configuration.mstar]
                + [getattr(configuration, param) for param in param_names[2:]]
            )
        )
    )
    if configuration.plots:
        pdf = PdfPages(configuration.plots)

    calculate_scenario = CalculateScenario(
        param_names=param_names,
        ntriangles=configuration.ntriangles,
        ntimes=configuration.ntimes,
        pool_manager=Manager(),
    )
    print(
        f"Found {len(calculate_scenario.plot_data)} / "
        f"{len(scenarios)} pre-calculated scenarios"
    )

    if not configuration.check_progress:
        with Pool(configuration.nthreads) as workers:
            workers.map(calculate_scenario, scenarios)
    elif not configuration.plots:
        return

    for param_values, (
        times,
        ebeer_flux,
        phoebe_flux,
    ) in calculate_scenario.plot_data.items():
        for subplot, flux_key in enumerate(
            ["ellipticity", "reflection", "everything"]
        ):
            pyplot.subplot(2, 2, subplot + 1)
            pyplot.plot(times, phoebe_flux[flux_key], "-k")
            pyplot.plot(times, ebeer_flux[flux_key], "-r")
            pyplot.title(flux_key)

        pyplot.gcf().text(
            0.6,
            0.1,
            "Scenario:\n - "
            + "\n - ".join(
                f"{name}: {value!r}"
                for name, value in zip(param_names, param_values)
            ),
        )
        pdf.savefig()
        pyplot.close()

    if configuration.plots:
        pdf.close()


if __name__ == "__main__":
    run_tests(parse_command_line())
