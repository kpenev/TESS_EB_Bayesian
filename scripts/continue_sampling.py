#!/usr/bin/env python3

"""Use templates to make fully funcional slurm files."""

from sqlalchemy import select

from command_line_util import (
    identify_hpc,
    create_parser,
    add_slurm_config,
    tic_per_node,
)
from new_sampling import get_file_makers

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import JobGroup, get_ticid_select_table


def parse_command_line():
    """Return the command line configuration."""

    this_hpc = identify_hpc()

    with Session.begin() as db_session:  # pylint: disable=no-member
        select_groups = select(JobGroup.id)
        if this_hpc is not None:
            select_groups = select_groups.filter_by(hpc=this_hpc)
        job_groups = db_session.scalars(select_groups).all()

    parser = create_parser()
    parser.add_argument(
        "job_group",
        choices=[str(g) for g in job_groups],
        help="Which job group should be continued.",
    )
    add_slurm_config(parser)
    parser.add_argument(
        "--continue-statuses",
        type=int,
        default=None,
        nargs="+",
        help="The status(es) assigned to the tics for which sampling should "
        "continue. If not specified, all positive status values will be "
        "continued.",
    )
    parser.add_argument(
        "--changed-likelihood-statuses",
        type=int,
        default=[],
        nargs="+",
        help="The status(es) assigned to the tics for which likelihood has "
        "changed and need the ``--changed-likelihood`` argument.",
    )
    parser.add_argument(
        "--submit-jobs",
        default=False,
        action="store_true",
        help="If passed, and the script is running on a cluster, the generated "
        "jobs are automatically submitted.",
    )
    return parser.parse_args()


def update_job(
    group_id,
    job_id,
    config,
    make_file,
    SelectTICTable,  # pylint: disable=invalid-name
    db_session,
):
    """Replace TIC IDs from the given job which are no longer to be sampled."""

    job_entries = db_session.scalars(
        select(SelectTICTable).filter_by(job_group=group_id, job_id=job_id)
    ).all()
    expected_num_jobs = config.nodes_per_job * tic_per_node[config.hpc]
    assert len(job_entries) == expected_num_jobs, (
        f"Got {len(job_entries)} instead of {expected_num_jobs} for job group "
        f"{group_id} on {config.hpc}"
    )
    cmd_substitutions = []

    select_replacement = select(SelectTICTable).filter_by(
        job_group=None, job_id=None
    )
    if config.continue_statuses is None:
        select_replacement = select_replacement.where(
            SelectTICTable.status > 0  # pylint: disable=no-member
        )
    else:
        select_replacement = select_replacement.where(
            SelectTICTable.status.in_(  # pylint: disable=no-member
                (config.continue_statuses or [])
                + config.changed_likelihood_statuses
            )
        )
    select_replacement = select_replacement.limit(1)

    for entry in job_entries:
        substitution = {"job_id": job_id, "ticid": entry.id}
        if entry.status in config.changed_likelihood_statuses:
            substitution["extra_cmdline"] = "--changed-likelihood"
        elif (config.continue_statuses is None and entry.status > 0) or (
            config.continue_statuses is not None
            and entry.status in config.continue_statuses
        ):
            substitution["extra_cmdline"] = ""
        else:
            entry.job_id = None
            entry.job_group = None
            replacement = db_session.scalar(select_replacement)
            replacement.job_group = group_id
            replacement.job_id = job_id
            substitution["ticid"] = replacement.id
            substitution["extra_cmdline"] = (
                "--changed-likelihood"
                if replacement.status in config.changed_likelihood_statuses
                else ""
            )

        cmd_substitutions.append(substitution)
    launcher_cmd = make_file["launcher_cmd"](cmd_substitutions)
    return make_file["slurm"](
        [{"job_id": job_id, "launcher_cmd": launcher_cmd}]
    )


def update_job_group(config):
    """Update the job files for the given group as needed."""

    with Session.begin() as db_session:  # pylint: disable=no-member
        job_group = db_session.scalar(
            select(JobGroup).filter_by(id=config.job_group)
        )
        SelectTICTable = get_ticid_select_table(
            job_group.select_tic_table, must_exist=True
        )
        config.hpc = job_group.hpc
        config.nodes_per_job = job_group.nodes_per_job
        make_file = get_file_makers(job_group.id, config)

        for job_id in range(job_group.num_jobs):
            update_job(
                job_group.id,
                job_id,
                config,
                make_file,
                SelectTICTable,
                db_session,
            )


if __name__ == "__main__":
    update_job_group(parse_command_line())
