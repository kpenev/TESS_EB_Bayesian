#!/usr/bin/env python3

"""A collection of tests comparing ebeer to PHOEBE."""

from itertools import product
from multiprocessing import Pool, Manager
from os import path
import pickle

from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
from matplotlib import pyplot, use
from matplotlib.backends.backend_pdf import PdfPages
import numpy

from plot_ebeer import (
    get_flux_modulations as get_ebeer_flux_modulations,
    get_rvs as get_ebeer_rvs,
    plot_modulations,
)

# False positive
# pylint: disable=unused-import
from phoebe_model import (
    create_phoebe_binary,
    get_phoebe_reference_flux,
    get_phoebe_ellipticity_mod,
    get_phoebe_reflection_mod,
    get_phoebe_eclipse_mod,
    get_phoebe_multieffect_mod,
)

# pylint: enable=unused-import


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
        [0.4, 0.7],
        "Values to try for the masses of the stars. The primary and "
        "secondary star masses are independently taken from this list.",
    )
    add_param(
        "peridistance-factor",
        [3.0, 7.0],
        "Values to try for the periapsis distance in units of the sum of "
        "the roche radii of the two stars.",
    )
    add_param("ecc", [0.0, 0.4, 0.8], "Values to try for the eccentricity.")
    add_param(
        "incl",
        [0.0, 30.0, 85.0, 90.0, 135.0],
        "Values to try for the inclination in degrees.",
    )
    add_param(
        "per0",
        [0.0, 30.0, 150.0],
        "Values to try for the argument of periapsis in degrees.",
    )
    add_param(
        "t0-supconj-factor",
        numpy.linspace(0.0, 1.0, 3),
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
        type=int,
        default=16,
        help="The number of parallel threads to use to do the calculations.",
    )
    parser.add_argument(
        "--check-progress",
        action="store_true",
        help="If passed, just counts how many scenarios are stored in the "
        "pickle and plots those.",
    )
    parser.add_argument(
        "--pickle-fname",
        default="ebeer_test_data.pkl",
        help="The filename to store/load pre-computed lightcurves in.",
    )
    parser.add_argument(
        "--refit-ebeer",
        default=False,
        action="store_true",
        help="Ignore pickled eBEER flux modulations and fit those from scratch."
        " Since PHOEBE is orders of magnitude more computationally intensive "
        "this is useful when experimenting with how to fit eBEER.",
    )
    return parser.parse_args()


# Intended to function just as a callable for multiprocessing
# pylint: disable=too-few-public-methods
class CalculateScenario:
    """Callable that calculates eBEER and PHOEBE models given binary params."""

    def __init__(
        self,
        *,
        param_names,
        ntriangles,
        ntimes,
        pool_manager,
        pickle_fname,
    ):
        """Prepare."""

        self.param_names = param_names
        self._ntriangles = ntriangles
        self._ntimes = ntimes
        self._pickle_lock = pool_manager.Lock()
        self._pickle_fnaame = pickle_fname
        self.plot_data = pool_manager.dict()
        if path.exists(pickle_fname):
            with open(pickle_fname, "rb") as pickle_f:
                try:
                    while True:
                        assert pickle.load(pickle_f) == "START RECORD"
                        assert pickle.load(pickle_f) == self.param_names
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

        parameters = dict(zip(self.param_names, param_values))
        phoebe_binary = create_phoebe_binary(**parameters)
        # print(f'Phoebe binary:\n{phoebe_binary}')
        phoebe_binary.set_value_all("ntriangles", self._ntriangles)
        times = numpy.linspace(
            0.0,
            phoebe_binary["orbit"]["component"]["period"].get_value(units.day),
            self._ntimes,
        )
        phoebe_binary.set_value_all("times", times * units.day)

        reference_flux = get_phoebe_reference_flux(phoebe_binary)

        phoebe_modulation = {
            modulation: globals()[f"get_phoebe_{modulation}_mod"](
                phoebe_binary, reference_flux
            )
            for modulation in ["ellipticity", "reflection", "eclipse"]
        }
        phoebe_modulation["out_of_eclipse"] = get_phoebe_multieffect_mod(
            phoebe_binary, reference_flux, exclude=("eclipse", "beaming")
        )
        phoebe_modulation["everything"] = get_phoebe_multieffect_mod(
            phoebe_binary, reference_flux, exclude=("beaming",)
        )
        phoebe_rv = {
            component: phoebe_binary[component]["rvs"]["rv01"][
                "model"
            ].get_value("km/s")
            for component in ["primary", "secondary"]
        }

        approx_modulation = get_ebeer_flux_modulations(
            phoebe_binary,
            times,
            phoebe_modulation["out_of_eclipse"]["combined"] + 1.0,
            exclude=("beaming", "eclipse"),
        )
        ebeer_rv = get_ebeer_rvs(phoebe_binary, times, phoebe_rv)
        result = (
            times,
            approx_modulation,
            phoebe_modulation,
            ebeer_rv,
            phoebe_rv,
        )
        self.plot_data[tuple(param_values)] = result
        self._pickle_lock.acquire()
        with open(self._pickle_fnaame, "ab") as pickle_f:
            pickle.dump("START RECORD", pickle_f)
            pickle.dump(self.param_names, pickle_f)
            pickle.dump(param_values, pickle_f)
            pickle.dump(result, pickle_f)
            pickle.dump("END RECORD", pickle_f)
        self._pickle_lock.release()
        # print(ebeer_binary)


# pylint: disable=too-few-public-methods


def get_plot_data(configuration):
    """Calculate everything to make the plots specified by configuration."""

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

    calculate_scenario = CalculateScenario(
        param_names=param_names,
        ntriangles=configuration.ntriangles,
        ntimes=configuration.ntimes,
        pool_manager=Manager(),
        pickle_fname=configuration.pickle_fname,
    )
    print(
        f"Found {len(calculate_scenario.plot_data)} / "
        f"{len(scenarios)} pre-calculated scenarios"
    )

    if not configuration.check_progress:
        with Pool(configuration.nthreads) as workers:
            workers.map(calculate_scenario, scenarios)

    return (
        param_names,
        {
            param_values: calculate_scenario.plot_data[param_values]
            for param_values in map(tuple, scenarios)
            if param_values in calculate_scenario.plot_data
        },
    )


def run_tests(configuration):
    """Run the tests specified on the command line."""

    param_names, plot_data = get_plot_data(configuration)

    if not configuration.plots:
        return

    use("PDF")
    pdf = PdfPages(configuration.plots)

    plot_param_translate = {
        "peridistance_factor": r"$\frac{r_{per}}{r_{roche}}$",
        "mprimary": r"$\frac{M_1}{M_{\odot}}$",
        "msecondary": r"$\frac{M_2}{M_{\odot}}$",
        "ecc": "e",
        "incl": "i",
        "per0": r"$\omega$",
        "t0_supconj_factor": r"$\frac{t_{sup. conj.}}{P_{orb}}$",
    }

    print(f"Plotting {len(plot_data)} scenarios")
    model = {}
    for param_values, (
        model["times"],
        model["ebeer_modulation"],
        model["phoebe_modulation"],
        model["ebeer_rv"],
        model["phoebe_rv"],
    ) in sorted(plot_data.items()):
        pyplot.subplots_adjust(left=0.15, right=0.95, top=0.8, bottom=0.1)

        if configuration.refit_ebeer:
            phoebe_binary = create_phoebe_binary(
                **dict(zip(param_names, param_values))
            )
            model["ebeer_modulation"] = get_ebeer_flux_modulations(
                phoebe_binary,
                model["times"],
                model["phoebe_modulation"]["out_of_eclipse"]["combined"] + 1.0,
                exclude=("beaming", "eclipse"),
            )
            model["ebeer_rv"] = get_ebeer_rvs(
                phoebe_binary, model["times"], model["phoebe_rv"]
            )

        plot_config = {
            "primary": {"color": "blue"},
            "secondary": {"color": "red"},
            "combined": {"color": "black"},
        }

        for subplot, modulation_key in enumerate(
            [
                "ellipticity",
                "reflection",
                "beaming",
                "eclipse",
                "out_of_eclipse",
                "everything",
            ]
        ):
            pyplot.subplot(2, 3, subplot + 1)

            for component_config in plot_config.values():
                component_config["linestyle"] = "-"
                component_config["alpha"] = 0.6

            phoebe_modulation = model["phoebe_modulation"].get(
                modulation_key, model["phoebe_rv"]
            )
            if modulation_key == "eclipse":
                phoebe_modulation = {
                    "combined": phoebe_modulation["combined"] + 1
                }
            plot_modulations(
                model["times"],
                phoebe_modulation,
                "PHOEBE {component}",
                **plot_config,
            )
            for component_config in plot_config.values():
                component_config["linestyle"] = ":"
                component_config["alpha"] = 1.0

            plot_modulations(
                model["times"],
                model["ebeer_modulation"].get(
                    modulation_key, model["ebeer_rv"]
                ),
                "eBEER {component}",
                **plot_config,
            )

            pyplot.title(modulation_key)

        pyplot.gcf().text(
            0.1,
            0.87,
            "Scenario: "
            + ", ".join(
                f"{plot_param_translate[name]}: {value}"
                for name, value in zip(param_names, param_values)
            )
            + "\nBest fit coef: "
            + ", ".join(
                [
                    r"$\alpha_{{refl,1}}="
                    f"{model['ebeer_modulation']['reflection_coef'][0]:0.3g}$",
                    r"$\alpha_{{refl,2}}="
                    f"{model['ebeer_modulation']['reflection_coef'][1]:0.3g}$",
                    r"$\alpha_{{beam,1}}="
                    f"{model['ebeer_rv']['beaming_coef'][0]:0.3g}$",
                    r"$\alpha_{{beam,2}}="
                    f"{model['ebeer_rv']['beaming_coef'][1]:0.3g}$",
                ]
            )
            + "\nBest fit $t_{0,perpass}$: "
            + ", ".join(
                [
                    "Ellip.+Refl.: ",
                    f"{float(model['ebeer_modulation']['t0_perpass']):0.3g}",
                    f"Primary RV: {model['ebeer_rv']['t0_perpass'][0]:0.3g}",
                    f"Secondary RV: {model['ebeer_rv']['t0_perpass'][1]:0.3g}",
                ]
            ),
        )
        pdf.savefig()
        pyplot.close()
        print("Added plot")

    if configuration.plots:
        pdf.close()


if __name__ == "__main__":
    run_tests(parse_command_line())
