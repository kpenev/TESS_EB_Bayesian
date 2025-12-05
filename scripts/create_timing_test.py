"""Create slurm job to demonstrate parallelism scaling."""

import os

from command_line_util import create_parser, add_slurm_config
from paths import data_dir
from new_sampling import FileFromTemplate


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument(
        "cores",
        type=int,
        help="The number of cores (possibly distributed over multiple nodes) to"
        " test timing on.",
    )
    parser.add_argument(
        "tic_list",
        help="Filename containing the list of TIC IDs to sample "
        "(one per line), optionally with a second entry on each line specifying"
        " the number of steps to take for each TIC.",
    )
    parser.add_argument(
        "--slurm-fname",
        default=os.path.join(
            os.path.dirname(data_dir),
            "slurm",
            "juno",
            "timing_test_{job_name}.slurm",
        ),
        help="The template for the slurm commands file. Should include "
        "at least ``{cores}`` substitutions.",
    )
    add_slurm_config(
        parser,
        partition="dev",
        time_limit="02:00:00",
        launcher_commands_fname=os.path.join(
            os.path.dirname(data_dir),
            "slurm",
            "juno",
            "timing_test_commands_{job_name}.txt",
        ),
    )
    return parser.parse_args()


def create_job(config):
    """Create the job per the command line configuration."""

    substitutions = {
        "nodes": (config.cores + 63) // 64,
        "cpus_per_node": min(config.cores, 64),
        "time_limit": config.time_limit,
        "partition": config.partition,
        "job_name": f"{config.cores}cores",
    }
    substitutions["tasks_per_node"] = max(
        substitutions["cpus_per_node"] // 16, 1
    )
    print(
        "Created " + FileFromTemplate(config.slurm_fname, substitutions)([{}])
    )
    command_substitutions = []
    with open(config.tic_list, "r", encoding="utf-8") as tic_list:
        for line in tic_list:
            tic, nsteps = map(int, (line.split() + [0])[:2])
            command_substitutions.append(
                {
                    "ticid": tic,
                    "extra_cmdline": (
                        f"--max-mcmc-steps {nsteps}" if nsteps else ""
                    ),
                }
            )

    print(
        "Created "
        + FileFromTemplate(config.launcher_commands_fname, substitutions)(
            command_substitutions
        )
    )
