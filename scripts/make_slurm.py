#!/usr/bin/env python3

"""Use templates to fully funcional make slurm files."""

from socket import gethostname
from os import makedirs
from os.path import dirname, exists

from configargparse import ArgumentParser, DefaultsFormatter
from sqlalchemy import select, func, update

from paths import slurm_fname, launcher_fname

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import get_ticid_select_table

_tic_per_node = {"juno": 4, "ls6": 8, "ganymede": 1}


def parse_command_line():
    """Return the command line configuration."""

    host = gethostname()
    this_hpc = None
    for candidate in _tic_per_node:
        if candidate in host.split("."):
            this_hpc = candidate

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["make_slurm.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "tic_table",
        help="Create slurm scripts for sampling the selected TIC identifiers in"
        " the given table. If the number is not divisible by how many a given "
        "HPC node can handle, the last job file will have fewer.",
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Pass this argument to update the job files after marking some "
        "systems as not to be sampled any more (e.g. finished or bad model). "
        "This will preserve the total number of jobs but bring in fresh TIC IDs"
        " to fill gaps in the job files. Jobs for which no systems are updated "
        "will be left untouched.",
    )
    parser.add_argument(
        "--tic-range",
        type=int,
        nargs=2,
        metavar=("START", "COUNT"),
        default=None,
        help="If specified, only COUNT tics, starting from START, will be "
        "used.",
    )
    parser.add_argument(
        "--hpc",
        choices=_tic_per_node.keys(),
        default=this_hpc,
        help="The HPC system to create the slurm scripts for.",
    )
    parser.add_argument(
        "--partition",
        default="normal",
        help="The SLURM partition to set up the script for.",
    )
    parser.add_argument(
        "--time-limit", default="48:00:00", help="The time limit for the jobs."
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="The number of processes to use for each TIC ID.",
    )
    parser.add_argument(
        "--launcher-njobs",
        type=int,
        default=None,
        metavar="NJOBS",
        help="If launcher is going to be used, this option specifies the number"
        " of jobs the list should be split into. If not specified, a "
        "non-launcher slurm file is created.",
    )
    parser.add_argument(
        "--launcher-commands-fname",
        "--launcher-fname",
        "--launcher-commands",
        default=launcher_fname,
        help="The template for the launcher commands file. Should include "
        "``{hpc}`` and ``{jobid}`` substitutions.",
    )
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
        "--changed-likelihood-status",
        type=int,
        default=None,
        help="The status assigned to the tics for which likelihood has changed "
        "and need the ``--changed-likelihood`` argument.",
    )
    return parser.parse_args()


def get_ticids_to_add(SelectTICIDs, acceptable_statuses=None, update_hpc=None):
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
            print(f'query: {query}')
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


# Meant to function as callable
# pylint: disable=too-few-public-methods
class FileFromTemplate:
    """Create a file from a template given substitutions."""

    def __init__(self, fname, fixed_substitutions):
        """Read the specified template get ready to make files."""

        self._fname = fname
        self._fixed_substitutions = fixed_substitutions
        with open(
            fname.format(**fixed_substitutions, jobid="template"),
            "r",
            encoding="utf-8",
        ) as template_f:
            self._template_text = template_f.read()

    def __call__(self, var_substitution_list):
        """Create a file adding the given substitutions to the fixed ones."""

        substitutions = self._fixed_substitutions.copy()
        substitutions.update(var_substitution_list[0])
        fname = self._fname.format_map(substitutions)
        dest_dir = dirname(fname)
        if not exists(dest_dir):
            makedirs(dest_dir)

        with open(fname, "w", encoding="utf-8") as outf:
            for var_substitutions in var_substitution_list:
                substitutions = self._fixed_substitutions.copy()
                substitutions.update(var_substitutions)
                file_contents = self._template_text
                for sub, value in substitutions.items():
                    file_contents = file_contents.replace(
                        f"@@{sub.upper()}@@", str(value)
                    )
                assert "@@" not in file_contents

                assert fname == self._fname.format_map(substitutions)
                outf.write(file_contents)
        return fname


# pylint: enable=too-few-public-methods


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
    print(f'TIC IDs by job: {ticids_by_job}')
    for job_index, ticid_list in ticids_by_job:
        if job_index is None and config.tic_range:
            ticid_list = ticid_list[
                config.tic_range[0] : config.tic_range[0] + config.tic_range[1]
            ]
        if config.launcher_njobs is None:
            ntics_per_job = _tic_per_node[config.hpc]
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
                    (ntics_per_job + _tic_per_node[config.hpc] - 1)
                    // _tic_per_node[config.hpc]
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
