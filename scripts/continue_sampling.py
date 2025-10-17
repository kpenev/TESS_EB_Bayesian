#!/usr/bin/env python3

"""Use templates to make fully funcional slurm files."""

from sqlalchemy import select

from command_line_util import (
    identify_hpc,
    create_parser,
    add_slurm_config,
    tic_per_node,
)
from new_sampling import get_file_makers, get_priority_tics

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import JobGroup, get_ticid_select_tables


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
    parser.add_argument(
        "--ignore-git-hash",
        action="store_true",
        help="If specified the current git hash is not checked against what is "
        "in the file.",
    )
    parser.add_argument(
        "--update-git-hash",
        action="store_true",
        help="If specified the current git hash relpaces what is in the file.",
    )
    return parser.parse_args()


def get_pending_tics(
    config, SelectTICTable, db_session  # pylint: disable=invalid-name
):
    """
    Return ordered TICs to fill gaps with split by status.

    Returns:
        dict:
            ``"continue"``: TICs for which sampling should continue if it has
                            already started.

            ``"restart"``: TICs for which sampling should restart (i.e.
                           likelihood changed).
    """

    select_pending = select(SelectTICTable).filter_by(
        job_group=None, job_id=None
    )
    if (
        config.continue_statuses is None
        and config.changed_likelihood_statuses is None
    ):
        pending = db_session.scalars(
            select_pending.where(
                SelectTICTable.status > 0  # pylint: disable=no-member
            )
        ).all()
    else:
        pending = (
            db_session.scalars(
                select_pending.where(
                    SelectTICTable.status.in_(  # pylint: disable=no-member
                        (config.continue_statuses or [])
                        + config.changed_likelihood_statuses
                    )
                )
            ).all(),
        )

    if config.priority_tic_file:
        priority_tics = get_priority_tics(config.priority_tic_file)
        return [
            entry for entry in pending if entry in priority_tics
        ] + [entry for entry in pending if entry not in priority_tics]
    return pending


def update_job( #pylint: disable=too-many-arguments
    *,
    group_id,
    job_id,
    config,
    make_file,
    pending,
    SelectTICTable,  # pylint: disable=invalid-name
    db_session,
):
    """Replace TIC IDs from the given job which are no longer to be sampled."""

    job_entries = db_session.scalars(
        select(SelectTICTable).filter_by(job_group=group_id, job_id=job_id)
    ).all()
    expected_num_tics = config.nodes_per_job * tic_per_node[config.hpc]
    assert len(job_entries) == expected_num_tics, (
        f"Got {len(job_entries)} instead of {expected_num_tics} for job group "
        f"{group_id} on {config.hpc}"
    )
    cmd_substitutions = []

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
            replacement = pending.pop(0)
            replacement.job_group = group_id
            replacement.job_id = job_id
            substitution["ticid"] = replacement.id
            substitution["extra_cmdline"] = ""
            if replacement.status in config.changed_likelihood_statuses:
                substitution["extra_cmdline"] += " --changed-likelihood"
            if config.ignore_git_hash:
                substitution["extra_cmdline"] += " --ignore-git-hash"
            elif config.update_git_hash:
                substitution["extra_cmdline"] += " --update-git-hash"

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
        SelectTICTable = get_ticid_select_tables(
            job_group.select_tic_table, must_exist=True
        )[0]
        config.hpc = job_group.hpc
        config.nodes_per_job = job_group.nodes_per_job
        make_file = get_file_makers(job_group.id, config)
        pending_tics = get_pending_tics(config, SelectTICTable, db_session)

        for job_id in range(job_group.num_jobs):
            update_job(
                group_id=job_group.id,
                job_id=job_id,
                config=config,
                make_file=make_file,
                pending=pending_tics,
                SelectTICTable=SelectTICTable,
                db_session=db_session,
            )


if __name__ == "__main__":
    update_job_group(parse_command_line())
