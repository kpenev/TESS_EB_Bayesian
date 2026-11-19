#!/usr/bin/env python3

"""
Create the parameter summaries of already rendered convergence plots.

One-time script (TEC-117) for convergence plots rendered before the summary
was saved along with them. The burn-in is the slow part of the convergence
plot, so rather than solving for it again, it is taken from the logs of the
rendering run (``Burnin is <burn-in> out of <steps> steps.``), along with the
number of steps the plot used. The last that many steps of the chain are used,
as rendering again with that limit would. If sampling has continued since the
plot, the logged burn-in is then conservative.

Every convergence plot without a summary must have exactly one burn-in and
number of steps in the logs (repeated runs must agree), otherwise nothing is
created.
"""

from argparse import ArgumentParser
from glob import glob
from os import path
import re
from types import SimpleNamespace

from chain_analysis import save_parameter_summary
from paths import samples as samples_fname_template
from visualize import get_plot_data

_rendering_re = re.compile(r"^Rendering convergence for (\d+) ", re.MULTILINE)
_burnin_re = re.compile(r"^Burnin is (\d+) out of (\d+) steps\.$", re.MULTILINE)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "plot_dir",
        help="The directory containing the rendered convergence plots. The "
        "summaries are created there.",
    )
    parser.add_argument(
        "log_dirs",
        nargs="+",
        help="The directories containing the ``*.outerr`` logs of the "
        "convergence rendering.",
    )
    parser.add_argument(
        "--tic-ids",
        type=int,
        nargs="+",
        default=None,
        help="Only summarize the plots of these TICs (default: all plots "
        "without a summary).",
    )
    parser.add_argument(
        "--samples-fname-template",
        default=samples_fname_template,
        help="The samples files, with ``{tic_id}`` substituted.",
    )
    return parser.parse_args()


def get_logged_burnin(log_dirs):
    """Return dict of TIC ID -> set of (burn-in, steps) found in the logs."""

    result = {}
    for log_dir in log_dirs:
        for log_fname in glob(path.join(log_dir, "*.outerr")):
            with open(log_fname, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()
            rendered = _rendering_re.findall(log_text)
            burnin = _burnin_re.findall(log_text)
            if not burnin:
                continue
            if len(rendered) != 1 or len(burnin) != 1:
                raise ValueError(
                    f"Expected one TIC and one burn-in in {log_fname}, found "
                    f"TICs {rendered!r} and burn-ins {burnin!r}."
                )
            result.setdefault(int(rendered[0]), set()).add(
                tuple(int(value) for value in burnin[0])
            )
    return result


def get_targets(config):
    """Return dict of TIC ID -> (burn-in, steps) for the plots to summarize."""

    logged = get_logged_burnin(config.log_dirs)
    targets = {}
    problems = []
    for plot_fname in sorted(glob(path.join(config.plot_dir, "tess*.png"))):
        tic_id = path.basename(plot_fname)[len("tess") : -len(".png")]
        if not tic_id.isdigit():
            continue
        tic_id = int(tic_id)
        if config.tic_ids is not None and tic_id not in config.tic_ids:
            continue
        if path.exists(path.splitext(plot_fname)[0] + "_parameters.json"):
            continue
        found = logged.get(tic_id, set())
        if len(found) != 1:
            problems.append(f"TIC {tic_id}: burn-in, steps = {found!r}")
        else:
            targets[tic_id] = found.pop()
    if problems:
        raise ValueError(
            "Need exactly one logged burn-in per plot:\n\t"
            + "\n\t".join(problems)
        )
    return targets


def summarize(tic_id, burnin, num_steps, config):
    """Create the parameter summary of one TIC from the logged burn-in."""

    plot_data, _, _, _, backend = get_plot_data(
        SimpleNamespace(
            samples_fname=config.samples_fname_template.format(tic_id=tic_id),
            chain_name="mcmc",
            burn_in=0,
            thin=1,
            sample_condition=None,
            # The last as many steps as the plot used.
            max_plot_steps=num_steps,
        ),
        None,
    )
    num_walkers = backend.shape[0]
    if plot_data.shape[0] != num_steps * num_walkers:
        raise ValueError(
            f"TIC {tic_id}: the plot used {num_steps} steps, but "
            f"{plot_data.shape[0] // num_walkers} were read from the chain."
        )
    if backend.iteration != num_steps:
        print(
            f"TIC {tic_id}: the chain now has {backend.iteration} steps, using "
            f"the last {num_steps} as the plot would."
        )
    save_parameter_summary(
        path.join(config.plot_dir, f"tess{tic_id}_parameters.json"),
        plot_data,
        num_walkers,
        burnin,
    )
    print(f"TIC {tic_id}: burn-in {burnin} of {num_steps} steps, summarized.")


def main(config):
    """Avoid polluting the global namespace."""

    targets = get_targets(config)
    print(f"Summarizing {len(targets)} convergence plots.")
    for tic_id, (burnin, num_steps) in targets.items():
        summarize(tic_id, burnin, num_steps, config)


if __name__ == "__main__":
    main(parse_command_line())
