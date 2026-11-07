#!/usr/bin/env python3

"""
Find the minima of the SED alone over a grid of stellar parameters.

Only the four parameters the SED depends on are fitted: total mass, mass
ratio, [M/H] and age. A least squares fit is started from every point of a
grid in these, and the distinct minima are reported. They are meant as
starting points for fits that add the light curve.

The fit is over log(total mass), mass ratio, [M/H] and log(age), within the
prior ranges. Outside the isochrones the masses are clipped to them, and one
residual per star of how far outside its mass is pushes the fit back in (see
`utils.fit_least_squares`), so the cost stays finite and continuous. The
starting ages are chosen per evolutionary phase of the primary (pre-main
sequence, main sequence, post main sequence) using the PARSEC ``label``
column.

The bad SED penalty is disabled and ``sed_sys`` is fixed: the penalty exists to
keep the light curve from drowning the SED, which cannot happen here.
"""

from multiprocessing import get_context
from types import SimpleNamespace
import logging

from astropy.table import Table
import numpy
from scipy import optimize

from command_line_util import create_parser
from binary_parameters import BinaryParams
from log_likelihood import LogLikelihood

_logger = logging.getLogger(__name__)

# The log-likelihood used by the worker processes (set before forking).
_target = None


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument("tic_id", type=int, help="The TIC ID to search.")
    parser.add_argument(
        "--mtotal-grid",
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "NUM"),
        default=(0.3, 10.0, 10),
        help="Total masses [Msun] to start from, log-spaced.",
    )
    parser.add_argument(
        "--mratio-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Mass ratios to start from.",
    )
    parser.add_argument(
        "--meh-grid",
        type=float,
        nargs="+",
        default=[-1.5, -0.8, -0.1, 0.5],
        help="[M/H] values to start from.",
    )
    parser.add_argument(
        "--ages-per-phase",
        type=int,
        nargs=3,
        metavar=("PRE_MS", "MS", "POST_MS"),
        default=(2, 3, 3),
        help="How many starting ages to place in each evolutionary phase of "
        "the primary.",
    )
    parser.add_argument(
        "--sed-sys",
        type=float,
        default=0.01,
        help="The fixed systematic SED uncertainty [mag].",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=8,
        help="How many least squares fits to run in parallel.",
    )
    parser.add_argument(
        "--output",
        default="sed_grid_search_{tic_id:d}.ecsv",
        help="Where to save all fit results.",
    )
    return parser.parse_args()


def get_star_log_age_range(mass, meh):
    """Return the log10(age/Gyr) range the isochrones cover for one star."""

    return BinaryParams.get_star_log_age_range(mass, meh)


def get_log_age_range(target, mtotal, mratio, meh):
    """Return the log10(age/Gyr) range valid for both stars and the prior."""

    mprimary = mtotal / (1.0 + mratio)
    ranges = [
        get_star_log_age_range(mprimary, meh),
        get_star_log_age_range(mprimary * mratio, meh),
        target.get_range("age_gyr"),
    ]
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


class SEDModel:
    """Evaluate the model SED of a binary from the isochrones only."""

    def __init__(self, target, sed_sys, grid_excess_scale=1e-6):
        """
        Prepare to evaluate SED residuals for the given log-likelihood.

        Args:
            target(LogLikelihood):    Provides the SED and its likelihood.

            sed_sys(float):    The fixed systematic SED uncertainty [mag].

            grid_excess_scale(float):    The distance of the masses outside the
                isochrones [Msun] that adds 1/2 to the cost.
        """

        BinaryParams.prepare_class()
        self._target = target
        self._sed_sys = sed_sys
        self.grid_excess_scale = grid_excess_scale
        # pylint: disable=protected-access
        self._interpolators = BinaryParams._cmd_interpolators
        self._passbands = BinaryParams()._passbands
        # pylint: enable=protected-access
        # The lowest cost and its fit vector among evaluations inside the
        # isochrones.
        self.best_cost = numpy.inf
        self.best_x = None

    def get_binary(self, mtotal, mratio, meh, age_gyr):
        """
        Return what `LogLikelihood.get_model_sed()` needs from a binary.

        The masses are clipped to the isochrones; the returned `grid_excess`
        says how far outside them each star's mass is.
        """

        mprimary = mtotal / (1.0 + mratio)
        component_teff = {}
        component_absmag = {}
        grid_excess = {}
        for component, mass in [
            ("primary", mprimary),
            ("secondary", mprimary * mratio),
        ]:
            values = []
            star_excess = 0.0
            for index, (photsys, interpolator) in enumerate(
                self._interpolators
            ):
                photsys_values, excess = interpolator(
                    (("logTe",) if index == 0 else ())
                    + self._passbands[photsys],
                    clip=True,
                    Mini=mass,
                    MH=meh,
                    logAge=9.0 + numpy.log10(age_gyr),
                )
                values.extend(photsys_values)
                star_excess = max(star_excess, excess)
            grid_excess[component] = star_excess
            component_teff[component] = 10.0 ** values[0]
            component_absmag[component] = numpy.array(values[1:])
        return SimpleNamespace(
            component_teff=component_teff,
            component_absmag=component_absmag,
            grid_excess=grid_excess,
        )

    @staticmethod
    def unpack(fit_vector):
        """Return (mtotal, mratio, meh, age_gyr) for a fit vector."""

        log_mtotal, mratio, meh, log_age = fit_vector
        return 10.0**log_mtotal, mratio, meh, 10.0**log_age

    def residuals(self, fit_vector):
        """Return the whitened SED residuals, log-det and grid excess terms."""

        binary = self.get_binary(*self.unpack(fit_vector))
        grid_excess = numpy.array(list(binary.grid_excess.values()))
        result = numpy.append(
            self._target.calc_sed_log_likelihood(
                binary, self._sed_sys, return_residuals=True
            ),
            grid_excess / self.grid_excess_scale,
        )
        cost = 0.5 * numpy.dot(result, result)
        if not grid_excess.any() and cost < self.best_cost:
            self.best_cost = cost
            self.best_x = numpy.array(fit_vector)
        return result


def get_phase_log_ages(mass, meh, log_age_range, ages_per_phase):
    """
    Return starting log10(age/Gyr) values in each evolutionary phase.

    The phases of a star of the given mass are found from the PARSEC
    ``label`` column (0: pre-main sequence, 1: main sequence, >1: later) along
    its track, and the requested number of ages is spread evenly in log(age)
    within each phase that exists inside ``log_age_range``.
    """

    # pylint: disable=protected-access
    interpolator = BinaryParams._cmd_interpolators[0][1]
    # pylint: enable=protected-access
    log_ages = numpy.linspace(*log_age_range, 200)
    labels = []
    for log_age in log_ages:
        try:
            labels.append(
                interpolator(("label",), Mini=mass, MH=meh, logAge=9 + log_age)[
                    0
                ]
            )
        except (ValueError, IndexError):
            labels.append(numpy.nan)
    labels = numpy.array(labels)
    result = []
    for phase_mask, num_ages in zip(
        [labels < 0.5, (labels >= 0.5) & (labels < 1.5), labels >= 1.5],
        ages_per_phase,
    ):
        if phase_mask.sum() < 2 or num_ages == 0:
            continue
        phase_ages = log_ages[phase_mask]
        result.extend(
            numpy.linspace(phase_ages[0], phase_ages[-1], num_ages + 2)[1:-1]
        )
    return result


def get_grid(target, config):
    """Return the starting fit vectors (log mtotal, mratio, meh, log age)."""

    min_mtotal, max_mtotal, num_mtotal = config.mtotal_grid
    result = []
    for mtotal in numpy.logspace(
        numpy.log10(min_mtotal), numpy.log10(max_mtotal), int(num_mtotal)
    ):
        for mratio in config.mratio_grid:
            for meh in config.meh_grid:
                try:
                    log_age_range = get_log_age_range(
                        target, mtotal, mratio, meh
                    )
                except (ValueError, IndexError):
                    continue
                if log_age_range[0] >= log_age_range[1]:
                    continue
                for log_age in get_phase_log_ages(
                    mtotal / (1.0 + mratio),
                    meh,
                    log_age_range,
                    config.ages_per_phase,
                ):
                    result.append((numpy.log10(mtotal), mratio, meh, log_age))
    return numpy.array(result)


def fit_one(arguments):
    """Run the SED-only least squares fit from one grid point."""

    start, sed_sys = arguments
    model = SEDModel(_target, sed_sys)
    meh_range = BinaryParams.meh_range
    log_age_range = _target.get_range("age_gyr")
    lower = [numpy.log10(0.21), 0.01, meh_range[0], log_age_range[0]]
    upper = [numpy.log10(50.0), 1.0, meh_range[1], log_age_range[1]]
    try:
        result = optimize.least_squares(
            model.residuals,
            numpy.clip(start, lower, upper),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
        )
    except ValueError:
        _logger.warning("SED fit failed from %s", repr(start), exc_info=True)
        return None
    if result.fun[-2:].any():
        # Ended just outside the isochrones (pulled out by the SED against
        # the finite grid excess residuals).
        _logger.info(
            "SED fit from %s ended %s Msun outside the isochrones; using the "
            "lowest cost evaluated inside.",
            repr(start),
            repr(result.fun[-2:] * model.grid_excess_scale),
        )
        if model.best_x is None:
            return None
        result.x, result.cost = model.best_x, model.best_cost
    mtotal, mratio, meh, age_gyr = model.unpack(result.x)
    binary = model.get_binary(mtotal, mratio, meh, age_gyr)
    model_sed = _target.get_model_sed(binary)[0]
    return {
        "start": start,
        "x": result.x,
        "cost": result.cost,
        "status": result.status,
        "nfev": result.nfev,
        "mtotal": mtotal,
        "mratio": mratio,
        "meh": meh,
        "age_gyr": age_gyr,
        "teff_primary": binary.component_teff["primary"],
        "teff_secondary": binary.component_teff["secondary"],
        "residuals": _target.sed[0] - model_sed,
    }


def distinct(results, tolerance=1e-3):
    """Return results sorted by cost with duplicate end points removed."""

    kept = []
    for result in sorted(results, key=lambda entry: entry["cost"]):
        if all(
            numpy.abs(result["x"] - other["x"]).max() > tolerance
            for other in kept
        ):
            kept.append(result)
    return kept


def main(config):
    """Run the grid search and report the distinct minima."""

    global _target  # pylint: disable=global-statement
    _target = LogLikelihood(config.tic_id)
    # pylint: disable=protected-access
    threshold, _, sharpness = _target._sed[2]
    _target._sed = (
        _target._sed[:2] + ((threshold, 0.0, sharpness),) + _target._sed[3:]
    )
    # pylint: enable=protected-access

    grid = get_grid(_target, config)
    _logger.info("Starting %d SED-only fits.", len(grid))
    with get_context("fork").Pool(config.num_parallel) as pool:
        results = [
            result
            for result in pool.map(
                fit_one, [(start, config.sed_sys) for start in grid]
            )
            if result is not None
        ]

    output = Table(
        {
            name: [result[name] for result in results]
            for name in [
                "start",
                "x",
                "cost",
                "status",
                "nfev",
                "mtotal",
                "mratio",
                "meh",
                "age_gyr",
                "teff_primary",
                "teff_secondary",
                "residuals",
            ]
        }
    )
    output.write(config.output.format(tic_id=config.tic_id), overwrite=True)

    minima = distinct(results)
    print(
        f"{len(grid)} grid points, {len(results)} fits, {len(minima)} "
        "distinct end points. Lowest cost first:"
    )
    for result in minima[:25]:
        print(
            f"cost {result['cost']:10.4g}  Mtot {result['mtotal']:6.3f}  "
            f"q {result['mratio']:5.3f}  [M/H] {result['meh']:6.2f}  "
            f"age {result['age_gyr']:8.4g} Gyr  Teff "
            f"{result['teff_primary']:5.0f}/{result['teff_secondary']:5.0f}  "
            f"residuals {numpy.round(result['residuals'], 3)}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
