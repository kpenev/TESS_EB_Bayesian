#!/usr/bin/env python3

"""
Create a SLURM job that renders the plots for the targets of one sampling job.

The job group and job are checked against the jobs database, which also gives
the table of TIC IDs. The plotting job takes the sampling job's name and SLURM's
singleton dependency, so it never runs at the same time as sampling of the same
job (in either order) and so never reads samples files while they are being
written. Plots go to the same directories as before,
regardless of job, so download_juno_plots.sh still collects them. Unlike
new_sampling.py, the jobs database is not changed.
"""

from sqlalchemy import select, func

from bui.db_interface import Session
from bui.select_ticids.data_model import JobGroup, get_ticid_select_tables
from command_line_util import create_parser, identify_hpc, tic_per_node
from new_prepare_job import fill_template, write_file
from paths import slurm_fname

# Each plot is rendered in its own process: one process per target, up to the
# cores of a node.
max_num_parallel = 64


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument("job_group", type=int, help="The job group ID.")
    parser.add_argument(
        "job_id", type=int, help="The ID of the job within the group."
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["best", "convergence"],
        help="The plots to render (plot types of bui/select_ticids/plots.py), "
        "one after the other.",
    )
    parser.add_argument(
        "--hpc",
        choices=tic_per_node.keys(),
        default=identify_hpc(),
        help="The HPC system to create the job for (needs a "
        "render_template.slurm in its slurm directory). Must match the job "
        "group.",
    )
    parser.add_argument(
        "--partition",
        default="normal",
        help="The SLURM partition to submit the job to.",
    )
    parser.add_argument(
        "--time-limit",
        default="48:00:00",
        help="The time limit for the job.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=None,
        help="How many plots to render at the same time (default: the number "
        f"of targets in the job, at most {max_num_parallel}).",
    )
    return parser.parse_args()


def get_job_targets(config):
    """Return the TIC table name and target count of the sampling job."""

    with Session.begin() as db_session:  # pylint: disable=no-member
        job_group = db_session.get(JobGroup, config.job_group)
        if job_group is None:
            raise ValueError(f"No job group {config.job_group} in the DB.")
        if job_group.hpc != config.hpc:
            raise ValueError(
                f"Job group {config.job_group} runs on {job_group.hpc}, not "
                f"{config.hpc}."
            )
        if not 0 <= config.job_id < job_group.num_jobs:
            raise ValueError(
                f"Job group {config.job_group} has jobs 0 to "
                f"{job_group.num_jobs - 1}, not {config.job_id}."
            )
        # This is actually a class
        # pylint: disable=invalid-name
        SelectTable, _ = get_ticid_select_tables(
            job_group.select_tic_table, must_exist=True
        )
        # pylint: enable=invalid-name
        num_targets = db_session.execute(
            select(func.count()).where(  # pylint: disable=not-callable
                SelectTable.job_group  # pylint: disable=no-member
                == config.job_group,
                SelectTable.job_id == config.job_id,
            )
        ).scalar_one()
        if num_targets == 0:
            raise ValueError(
                f"No targets assigned to job {config.job_group}-"
                f"{config.job_id}."
            )
        return job_group.select_tic_table, num_targets


def main(config):
    """Create the SLURM file for rendering the plots."""

    table_name, num_targets = get_job_targets(config)
    job_name = f"{config.job_group}-{config.job_id}"
    job_fname = slurm_fname.format(
        hpc=config.hpc, mode="render", job_name=job_name
    )
    write_file(
        job_fname,
        fill_template(
            slurm_fname.format(
                hpc=config.hpc, mode="render", job_name="template"
            ),
            {
                "job_name": job_name,
                "job_group": config.job_group,
                "job_id": config.job_id,
                "table_name": table_name,
                "plot_types": " ".join(config.plot_types),
                "num_parallel": config.num_parallel
                or min(num_targets, max_num_parallel),
                "partition": config.partition,
                "time_limit": config.time_limit,
            },
        ),
    )
    print(
        f"{num_targets} targets of job {job_name} ({table_name}).\n"
        f"Submit with: sbatch {job_fname}"
    )


if __name__ == "__main__":
    main(parse_command_line())
