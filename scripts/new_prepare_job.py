#!/usr/bin/env python3

"""
Create a SLURM job that finds the starting walkers for a list of TIC IDs.

Each TIC ID runs prepare_sampling.sh (SED search, seeded fits, families,
initial walkers and their plots) through launcher, a few at a time per node
(``LAUNCHER_PPN`` in the SLURM template), in the order of the list. When one
finishes the next starts, so the list may be longer than one job can process;
a TIC ID left unfinished resumes in the next job, since finished steps are
skipped. Unlike new_sampling.py, the jobs database is not changed.
"""

from datetime import datetime
from os import makedirs, path

from command_line_util import (
    add_slurm_config,
    create_parser,
    identify_hpc,
    tic_per_node,
)
from paths import launcher_fname, slurm_fname


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument(
        "tic_list",
        help="A file listing the TIC IDs (separated by whitespace) in the "
        "order in which to process them. Repeats are ignored.",
    )
    parser.add_argument(
        "--hpc",
        choices=tic_per_node.keys(),
        default=identify_hpc(),
        help="The HPC system to create the job for (needs a "
        "launcher_prepare_commands_template.txt in its slurm directory).",
    )
    parser.add_argument(
        "--nodes-per-job",
        type=int,
        default=1,
        help="How many nodes the job uses.",
    )
    parser.add_argument(
        "--job-name",
        default=datetime.now().strftime("%Y%m%d%H%M%S"),
        help="Identifies the job and its files (default: the current time).",
    )
    add_slurm_config(
        parser,
        request_priority=False,
        launcher_commands_fname=launcher_fname.replace(
            "launcher_commands_", "launcher_prepare_commands_"
        ),
    )
    return parser.parse_args()


def read_tic_ids(fname):
    """Return the TIC IDs listed in a file, in order, without repeats."""

    with open(fname, "r", encoding="utf-8") as tic_file:
        return list(
            dict.fromkeys(int(word) for word in tic_file.read().split())
        )


def fill_template(template_fname, substitutions):
    """Return a template's text with ``@@NAME@@`` replaced by the values."""

    with open(template_fname, "r", encoding="utf-8") as template:
        text = template.read()
    for name, value in substitutions.items():
        text = text.replace(f"@@{name.upper()}@@", str(value))
    assert "@@" not in text, f"Unfilled substitutions in:\n{text}"
    return text


def write_file(fname, text):
    """Write text to a file, creating its directory if needed."""

    makedirs(path.dirname(fname), exist_ok=True)
    with open(fname, "w", encoding="utf-8") as output:
        output.write(text)


def main(config):
    """Create the launcher commands file and the SLURM file."""

    tic_ids = read_tic_ids(config.tic_list)
    job_name = f"prepare-{config.job_name}"
    commands_fname = config.launcher_commands_fname.format(
        hpc=config.hpc, job_name=job_name
    )
    write_file(
        commands_fname,
        "".join(
            fill_template(
                config.launcher_commands_fname.format(
                    hpc=config.hpc, job_name="template"
                ),
                {"ticid": tic_id, "num_parallel": config.num_parallel},
            )
            for tic_id in tic_ids
        ),
    )
    job_fname = slurm_fname.format(
        hpc=config.hpc, mode="launcher", job_name=job_name
    )
    write_file(
        job_fname,
        fill_template(
            slurm_fname.format(
                hpc=config.hpc, mode="launcher", job_name="template"
            ),
            {
                "nodes_per_job": config.nodes_per_job,
                "partition": config.partition,
                "job_group": "prepare",
                "job_name": config.job_name,
                "launcher_cmd": commands_fname,
            },
        ),
    )
    print(
        f"{len(tic_ids)} TIC IDs in {commands_fname}.\n"
        f"Submit with: sbatch {job_fname}"
    )


if __name__ == "__main__":
    main(parse_command_line())
