#!/usr/bin/env python3

"""Use templates to make fully funcional slurm files."""

from sqlalchemy import select, func, update

from paths import slurm_fname, launcher_fname

from command_line_util import (
    identify_hpc,
    tic_per_node,
    create_parser,
    add_slurm_config,
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

    with Session.begin() as db_session:
        job_groups = db_session.scalars(
            select(JobGroup.id).filter_by(hpc=this_hpc)
        ).all()

    parser = create_parser()
    parser.add_argument(
        "job_group",
        choices=job_groups,
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


def update_job(group_id, job_id, config, make_file, SelectTICTable, db_session):
    """Replace TIC IDs from the given job which are no longer to be sampled."""

    job_entries = db_session.scalars(
        select(SelectTICTable).filter_by(job_group=group_id, job_id=job_id)
    ).all()
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
            entry.status in config.continue_statuses
        ):
            substitution["extra_cmdline"] = ""
        else:
            entry.job_id = None
            entry.job_grop = None
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

    SelectTICTable = get_ticid_select_table(config.tic_table, must_exist=True)
    with Session.begin() as db_session:  # pylint: disable=no-member
        job_group = db_session.scalar(
            select(JobGroup).filter_by(id=config.job_group)
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


def get_ticids_to_add(
    SelectTICIDs,  # pylint: disable=invalid-name
    acceptable_statuses=None,
    update_hpc=None,
):
    """
    Return the list of TIC IDs to add for sampling.
    """

    with Session.begin() as db_session:  # pylint: disable=no-member
        query = select(
            SelectTICIDs.id, SelectTICIDs.status  # pylint: disable=no-member
        )
        if update_hpc is None:
            if acceptable_statuses is None:
                query = query.where(
                    SelectTICIDs.status > 0  # pylint: disable=no-member
                )
            else:
                query = query.where(
                    SelectTICIDs.status.in_(  # pylint: disable=no-member
                        acceptable_statuses
                    )
                )
            print(f"query: {query}")
            return [(None, list(db_session.execute(query).all()))]
        replace = db_session.execute(
            select(
                func.count(  # pylint: disable=not-callable
                    SelectTICIDs.id  # pylint: disable=no-member
                ),
                SelectTICIDs.job_id,  # pylint: disable=no-member, not-callable
            )
            .where(SelectTICIDs.hpc == update_hpc)  # pylint: disable=no-member
            .where(
                SelectTICIDs.not_in(  # pylint: disable=no-member
                    acceptable_statuses
                )
            )
            .group_by(SelectTICIDs.job_id)  # pylint: disable=no-member
            .all()
        )
        result = []
        for num_replace, job_id in replace:
            result.append(
                job_id,
                list(
                    db_session.execute(
                        query.filter_by(job_id=None).limit(num_replace).all()
                    )
                )
                + list(
                    db_session.execute(
                        query.filter_by(job_id=job_id)
                        .filter_by(hpc=update_hpc)
                        .where(
                            SelectTICIDs.status.in_(  # pylint: disable=no-member
                                acceptable_statuses
                            )
                        )
                    )
                ),
            )
        return result


def make_slurm(config):
    """Create the slurm scripts per the given configuration."""

    SelectTICIDs = get_ticid_select_table(  # pylint: disable=invalid-name
        config.tic_table,
    )

    ticids_by_job = get_ticids_to_add(
        SelectTICIDs,
        (
            None
            if config.continue_statuses is None
            else (
                config.continue_statuses
                + (
                    []
                    if config.changed_likelihood_status is None
                    else [config.changed_likelihood_status]
                )
            )
        ),
        config.hpc if config.update_only else None,
    )
    print(f"TIC IDs by job: {ticids_by_job}")
    for job_index, ticid_list in ticids_by_job:
        if job_index is None and config.tic_range:
            ticid_list = ticid_list[
                config.tic_range[0] : config.tic_range[0] + config.tic_range[1]
            ]
        if config.launcher_njobs is None:
            ntics_per_job = tic_per_node[config.hpc]
        else:
            ntics_per_job = (
                len(ticid_list) + config.launcher_njobs - 1
            ) // config.launcher_njobs

        config.slurm_mode = (
            "basic" if config.launcher_njobs is None else "launcher"
        )
        substitutions = {
            "hpc": config.hpc,
            "partition": config.partition,
            "mode": config.slurm_mode,
            "nodes_per_job": (
                1
                if config.launcher_njobs is None
                else (
                    (ntics_per_job + tic_per_node[config.hpc] - 1)
                    // tic_per_node[config.hpc]
                )
            ),
            "ntics_per_job": ntics_per_job,
            "num_parallel": config.num_parallel,
            "time_limit": config.time_limit,
        }

        substitutions["processes_per_job"] = (
            (ntics_per_job + substitutions["nodes_per_job"] - 1)
            // substitutions["nodes_per_job"]
        ) * substitutions["nodes_per_job"]
        make_slurm_file = FileFromTemplate(slurm_fname, substitutions)
        if config.launcher_njobs is not None:
            make_launchercmd_file = FileFromTemplate(
                config.launcher_commands_fname, substitutions
            )

        first_tic = 0
        if job_index is None:
            jobid = 0
        while first_tic < len(ticid_list):

            if config.launcher_njobs is None:
                tic_list = " ".join(
                    [
                        str(tic)
                        for tic, _ in ticid_list[
                            first_tic : first_tic + ntics_per_job
                        ]
                    ]
                )
                make_slurm_file(
                    [
                        {
                            "tic_list": tic_list,
                            "jobid": tic_list.replace(" ", "_"),
                        }
                    ]
                )
            else:
                cmdfname = make_launchercmd_file(  # pylint: disable=possibly-used-before-assignment
                    [
                        {
                            "jobid": jobid,
                            "ticid": tic,
                            "extra_cmdline": (
                                "--changed-likelihood"
                                if status == config.changed_likelihood_status
                                else ""
                            ),
                        }
                        for tic, status in ticid_list[
                            first_tic : first_tic + ntics_per_job
                        ]
                    ]
                )
                make_slurm_file(
                    [{"jobid": str(jobid), "launcher_cmd": cmdfname}]
                )
                if job_index is None:
                    jobid += 1
                else:
                    assert first_tic + ntics_per_job == len(ticid_list)

                with Session.begin() as db_session:  # pylint: disable=no-member
                    db_session.execute(
                        update(SelectTICIDs),
                        [
                            {"id": tic_id, "hpc": config.hpc, "job_id": jobid}
                            for tic_id, _ in ticid_list[
                                first_tic : first_tic + ntics_per_job
                            ]
                        ],
                    )

            first_tic += ntics_per_job


if __name__ == "__main__":
    make_slurm(parse_command_line())
