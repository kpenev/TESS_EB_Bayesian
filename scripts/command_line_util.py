"""Utilities used by scripts during command line parsing ."""

from socket import gethostname
import inspect
import os.path

from configargparse import ArgumentParser, DefaultsFormatter

from paths import launcher_fname

tic_per_node = {"juno": 4, "ls6": 8, "ganymede": 1}
assumed_num_parallel = 16


def identify_hpc():
    """Try to identify which HPC the script is running on."""

    host = gethostname()
    for candidate in tic_per_node:
        if candidate in host.split(".") + host.split("-"):
            return candidate
    return None


def create_parser():
    """Boiler plate command line parser creation."""

    caller_info = inspect.stack()[1]
    return ArgumentParser(
        description=caller_info.frame.f_globals["__doc__"],
        default_config_files=[
            f"{os.path.splitext(os.path.basename(caller_info.filename))[0]}.cfg"
        ],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )


def add_slurm_config(parser):
    """Add slurm configuration options to the given parser."""

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
        default=assumed_num_parallel,
        help="The number of processes to use for each TIC ID.",
    )
    parser.add_argument(
        "--launcher-commands-fname",
        "--launcher-fname",
        "--launcher-commands",
        default=launcher_fname,
        help="The template for the launcher commands file. Should include "
        "at least ``{hpc}`` and ``{job_name}`` substitutions.",
    )
    parser.add_argument(
        "--priority-tic-file",
        default=None,
        help="Specify a file with list of TIC IDs that should get priority for "
        "sampling.",
    )
