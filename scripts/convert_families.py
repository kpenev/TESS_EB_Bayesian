#!/usr/bin/env python3

"""
Convert families saved under an older sed_sys prior range to the current one.

Family best points are latent samples, so the meaning of their sed_sys entry
depends on the prior range of sed_sys. The batch-1 preparations used [1e-10,
0.1] mag; since 2026-11-18 sed_sys is at most 0.03 mag, so the same latent
value would mean a smaller sed_sys. Each point is converted by value instead:
its sed_sys under the old range, capped just below the new maximum, back to a
latent under the current range. The other parameters keep their latents (their
ranges did not change).

Then (unless --no-refit) each point is refit under the current likelihood
(`utils.fit_least_squares()`), so initial_walkers.py estimates each mode around
an optimum (a capped sed_sys moves the optimum), and its eclipse time is folded
mid-prior as select_families.py does. Refits that end within --duplicate-delta
of a better family are reported; all families are kept.

The output is the .npz that initial_walkers.py reads, with the families'
isochrone interpolation (linear for a .npy from before it was recorded).
"""

from argparse import ArgumentParser
from functools import partial
from multiprocessing import get_context
from os import path
import logging

import numpy

import select_families
from log_likelihood import LogLikelihood
from sample_params import SampleParams
from utils import default_give_up, fit_least_squares

_logger = logging.getLogger(__name__)
_target = None


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC ID of the families.")
    parser.add_argument(
        "families",
        help="The families file: the .npz of select_families.py or an older "
        ".npy (linear isochrone interpolation).",
    )
    parser.add_argument(
        "output",
        help="Where to save the converted families (.npz; must not exist).",
    )
    parser.add_argument(
        "--old-sed-sys-range",
        type=float,
        nargs=2,
        default=(-10.0, -1.0),
        metavar=("LOW", "HIGH"),
        help="The log10 prior range of sed_sys [mag] the families were saved "
        "under.",
    )
    parser.add_argument(
        "--cap-fraction",
        type=float,
        default=0.99,
        help="Larger sed_sys values become this fraction of the current "
        "maximum (at the maximum itself the latent is infinite).",
    )
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Only convert sed_sys; do not refit the points.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help="Give up a refit after this long [s], keeping its best point.",
    )
    parser.add_argument(
        "--duplicate-delta",
        type=float,
        default=1e-3,
        help="Report refit families whose latent points end closer than this "
        "(largest coordinate difference) to a better family.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=8,
        help="How many refits to run in parallel.",
    )
    return parser.parse_args()


def load_families(fname):
    """Return the family points and their isochrone interpolation."""

    families = numpy.load(fname)
    if isinstance(families, numpy.ndarray):
        return numpy.atleast_2d(families).astype(float), "linear"
    with families:
        return (
            numpy.atleast_2d(families["points"]).astype(float),
            str(families["isochrone_interpolation"]),
        )


def convert_sed_sys(point, old_range, cap_fraction):
    """
    Return the point with sed_sys converted by value, and both values.

    Returns:
        (array, float, float):    The converted point, and sed_sys [mag]
            under the old range and after capping.
    """

    index = SampleParams._fields.index("sed_sys")
    with _target.changed_range("sed_sys", old_range):
        old_value = _target.prior_transform("sed_sys", point[index])
    maximum = 10.0 ** _target.get_range("sed_sys")[1]
    new_value = min(old_value, cap_fraction * maximum)
    converted = point.copy()
    converted[index] = _target.inverse_prior("sed_sys", new_value)[1]
    return converted, old_value, new_value


def refit(arguments):
    """Refit one point under the current likelihood (worker process)."""

    point, time_limit = arguments
    result = fit_least_squares(
        _target,
        point,
        give_up=(partial(default_give_up, time_limit=time_limit),),
    )
    folded = select_families.fold_periodic(result.x)
    return folded, int(result.status), int(result.nfev)


def convert_all(points, config):
    """Convert every point; return them and a report entry for each."""

    old_range = tuple(config.old_sed_sys_range)
    converted, report = [], []
    for point in points:
        with _target.changed_range("sed_sys", old_range):
            old_log_prob = _target(point)[0]
        new_point, old_value, new_value = convert_sed_sys(
            point, old_range, config.cap_fraction
        )
        converted.append(new_point)
        report.append(
            {
                "old sed_sys": old_value,
                "new sed_sys": new_value,
                "log_prob (old range)": old_log_prob,
                "log_prob (converted)": _target(new_point)[0],
            }
        )
    return numpy.array(converted), report


def refit_all(converted, report, config):
    """Refit every converted point; add the outcome to the report."""

    with get_context("fork").Pool(config.num_parallel) as pool:
        refits = pool.map(
            refit, [(point, config.time_limit) for point in converted]
        )
    for entry, (point, status, nfev) in zip(report, refits):
        entry["log_prob (refit)"] = _target(point)[0]
        entry["refit sed_sys"] = _target.get_sample_params(point).sed_sys
        entry["refit status/nfev"] = f"{status}/{nfev}"
    return numpy.array([point for point, _, _ in refits])


def print_report(report, final, config):
    """Print each family, best first, and any that end on a better one."""

    rank_by = "log_prob (converted)" if config.no_refit else "log_prob (refit)"
    order = numpy.argsort([-entry[rank_by] for entry in report])
    for rank, index in enumerate(order):
        print(
            f"family {index}: "
            + ", ".join(
                (
                    f"{key} {value:.2f}"
                    if key.startswith("log_prob")
                    else (
                        f"{key} {value:.4g}"
                        if isinstance(value, float)
                        else f"{key} {value}"
                    )
                )
                for key, value in report[index].items()
            )
        )
        close = [
            int(other)
            for other in order[:rank]
            if numpy.max(numpy.abs(final[index] - final[other]))
            < config.duplicate_delta
        ]
        if close:
            print(f"    ends on the same point as better families {close}")


def main(config):
    """Convert the families and save them for initial_walkers.py."""

    global _target  # pylint: disable=global-statement
    if not config.output.endswith(".npz"):
        raise ValueError(f"The output must be a .npz file: {config.output!r}")
    if path.exists(config.output):
        raise FileExistsError(f"{config.output!r} already exists.")
    points, isochrone_interpolation = load_families(config.families)
    _target = LogLikelihood(
        config.tic_id, isochrone_interpolation=isochrone_interpolation
    )
    # fold_periodic() uses select_families' likelihood for the prior ranges.
    select_families._target = _target  # pylint: disable=protected-access

    final, report = convert_all(points, config)
    if not config.no_refit:
        final = refit_all(final, report, config)
    print_report(report, final, config)

    numpy.savez(
        config.output,
        points=final,
        isochrone_interpolation=isochrone_interpolation,
    )
    print(f"Saved {len(final)} families to {config.output!r}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
