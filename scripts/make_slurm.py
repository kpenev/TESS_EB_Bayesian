#!/usr/bin/env python3

"""Use templates to fully funcional make slurm files."""

from socket import gethostname
from os import makedirs
from os.path import dirname, exists

from configargparse import ArgumentParser, DefaultsFormatter
from sqlalchemy import select

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
        "--slurm-mode",
        default=None,
        help="Determines which slurm file to use to generate the template. By "
        "default assumes ``basic`` if launcher is not used and ``launcher`` "
        "otherwise.",
    )
    parser.add_argument(
        "--tic-range",
        type=int,
        nargs=2,
        metavar=("START", "COUNT"),
        default=None,
        help="If specified, only COUNT tics, starting from START, will be " "used.",
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
        "--continue-flags",
        type=int,
        default=None,
        nargs="+",
        help="The flag assigned to the tics for which sampling should continue."
        " If not specified, all positive flags will be continued.",
    )
    parser.add_argument(
        "--changed-likelihood-flag",
        type=int,
        default=None,
        help="The flag assigned to the tics for which likelihood has changed "
        "and need the ``--changed-likelihood`` argument.",
    )
    return parser.parse_args()


def get_ticid_list(tablename, acceptable_flags=None):
    """Return the list of TIC IDs to sample."""

    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(tablename)
    # pylint: enable=invalid-name

    query = select(SelectTICIDs.id, SelectTICIDs.flag)
    if acceptable_flags is None:
        query = query.where(SelectTICIDs.flag > 0)
    else:
        query = query.where(SelectTICIDs.flag.in_(acceptable_flags))
    with Session.begin() as db_session:
        # pylint: enable=no-member
        return list(db_session.execute(query).all())


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

    ticid_list = get_ticid_list(
        config.tic_table,
        (
            None
            if config.continue_flags is None
            else (
                config.continue_flags
                + (
                    []
                    if config.changed_likelihood_flag is None
                    else [config.changed_likelihood_flag]
                )
            )
        ),
    )
    if config.tic_range:
        ticid_list = ticid_list[
            config.tic_range[0] : config.tic_range[0] + config.tic_range[1]
        ]
    if config.launcher_njobs is None:
        ntics_per_job = _tic_per_node[config.hpc]
    else:
        ntics_per_job = (
            len(ticid_list) + config.launcher_njobs - 1
        ) // config.launcher_njobs

    if config.slurm_mode is None:
        config.slurm_mode = "basic" if config.launcher_njobs is None else "launcher"
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
    while first_tic < len(ticid_list):

        if config.launcher_njobs is None:
            tic_list = " ".join(
                [
                    str(tic)
                    for tic, _ in ticid_list[first_tic : first_tic + ntics_per_job]
                ]
            )
            make_slurm_file(
                [{"tic_list": tic_list, "jobid": tic_list.replace(" ", "_")}]
            )
        else:
            jobid = f"{first_tic:03d}_{ntics_per_job:03d}"
            cmdfname = make_launchercmd_file(
                [
                    {
                        "jobid": jobid,
                        "ticid": tic,
                        "extra_cmdline": (
                            "--changed-likelihood"
                            if flag == config.changed_likelihood_flag
                            else ""
                        ),
                    }
                    for tic, flag in ticid_list[first_tic : first_tic + ntics_per_job]
                ]
            )
            make_slurm_file([{"jobid": jobid, "launcher_cmd": cmdfname}])

        first_tic += ntics_per_job


if __name__ == "__main__":
    make_slurm(parse_command_line())
