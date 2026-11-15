#!/usr/bin/env python3

"""
Send the likelihood cache rows of targets to be sampled to the sampling host.

Takes a consistent copy of this machine's likelihood cache, copies it to the
sampling host (by default Juno) with rsync, and runs merge_cache.py there:
the BLS, SED and exclusion rows of the given targets the host does not have
yet are added to its cache, and every target the host already has (given or
not) is compared with this machine's instead of being changed, since it may
be being sampled (see merge_cache.py). The targets are those given on the
command line or, by default, those with status *sample* in the BUI review
table.

Needs passwordless ssh to the host, and on it a checkout of this repository
with merge_cache.py and a Python environment able to run it.

Exit status: merge_cache.py's on the host (0 if all compared targets match, 3
if some differ), or 1 if a step failed before.
"""

from argparse import ArgumentParser
import logging
from os import chdir, path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

from merge_cache import save_copy

_logger = logging.getLogger(__name__)

# The BUI statuses of targets to sample: sample (from the prepared walkers)
# and changed_likelihood (from the prepared walkers, after data exclusions or
# SED settings changed in the review).
sample_statuses = (4, 5)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "tic_ids",
        nargs="*",
        type=int,
        help="The targets to send (default: those with status sample in "
        "--review-table).",
    )
    parser.add_argument(
        "--review-table",
        default="sample_prsa",
        help="The BUI table of the reviewed TICs (default: %(default)s).",
    )
    parser.add_argument(
        "--host",
        default="kxp174430@juno",
        help="The sampling host, as given to ssh (default: %(default)s).",
    )
    parser.add_argument(
        "--remote-repository",
        default="projects/git/TESS_EB_Bayesian",
        help="The repository on the host, absolute or relative to the home "
        "directory (default: %(default)s).",
    )
    parser.add_argument(
        "--remote-python",
        default="python3",
        help="The Python command on the host (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Copy the cache to the host, but only check and compare there.",
    )
    return parser.parse_args()


def get_sample_tic_ids(review_table):
    """Return the TICs with a status to sample in the BUI review table."""

    # The BUI database is opened relative to the working directory.
    chdir(path.dirname(path.abspath(__file__)))
    # pylint: disable=import-outside-toplevel
    from sqlalchemy import select
    from bui.db_interface import Session
    from bui.select_ticids.data_model import get_ticid_select_tables

    # pylint: enable=import-outside-toplevel

    select_table, _ = get_ticid_select_tables(review_table, must_exist=True)
    with Session() as db_session:  # pylint: disable=not-callable
        return sorted(
            db_session.scalars(
                select(select_table.id).where(
                    select_table.status.in_(sample_statuses)
                )
            )
        )


def main(config):
    """Send the cache and merge it on the host."""

    tic_ids = config.tic_ids or get_sample_tic_ids(config.review_table)
    if not tic_ids:
        _logger.error("No targets to send.")
        return 1
    _logger.info("Sending %d targets: %s", len(tic_ids), tic_ids)
    remote_incoming = path.join(
        config.remote_repository, "data", "incoming_mcmc_cache.sqlite"
    )
    with TemporaryDirectory() as temp_dir:
        snapshot = save_copy(path.join(temp_dir, "mcmc_cache.sqlite"))
        subprocess.run(
            ["rsync", "-az", snapshot, f"{config.host}:{remote_incoming}"],
            check=True,
        )
    merge_command = shlex.join(
        [
            config.remote_python,
            "merge_cache.py",
            path.join("..", "data", "incoming_mcmc_cache.sqlite"),
            *map(str, tic_ids),
        ]
        + (["--dry-run"] if config.dry_run else [])
    )
    remote_command = (
        f"cd {shlex.quote(path.join(config.remote_repository, 'scripts'))} "
        f"&& {merge_command}"
    )
    _logger.info("Running on %s: %s", config.host, remote_command)
    return subprocess.run(
        ["ssh", config.host, remote_command], check=False
    ).returncode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(parse_command_line()))
