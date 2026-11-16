#!/usr/bin/env python3

"""
Merge the BUI database here with the sampling host's (Juno), both ways.

Each side owns part of ``scripts/TESS_EBs.db``:

* the host owns the job groups (``job_groups``) and each target's job
  assignment (``job_group``, ``job_id`` of the review table): sampling jobs
  are created and filled there (new_sampling.py, continue_sampling.py);
* both change the status of targets: the review here (BUI decisions), and
  the host when mcmc_sampling.py --changed-likelihood has recomputed the
  walkers (changed_likelihood -> sample).

Everything else (review flags, rendered plots, re-preparation requests) is
left alone on both sides.

Before any status is merged, the data of the targets to sample (light curves,
prepared samples files, cache rows) is pushed to the host (push_cache.py), so
a target the host may start sampling always has its data there. If cache rows
of targets the host already has differ, the user is asked whether to go on
(default: no).

A status is merged against the baseline, the merged statuses saved at the end
of the previous sync: a status changed on one side since then is taken from
that side; one changed differently on both sides is a conflict, and the user
is asked which to keep (default: this machine's). Without a baseline (first
sync, ``--no-baseline``) every status that differs is a conflict.

The host's database is read from a consistent snapshot. The changes are then
applied there first, then here, each after saving a timestamped copy of the
database next to it (``TESS_EBs.db.before-sync-<time>``), each machine in one
transaction. Every status changed must still have the value the merge saw:
if one changed since the snapshot, the sync stops without changing anything
on that machine (if it is this one, the host's changes are already applied)
and saves no baseline; running it again merges the new status.
"""

from argparse import ArgumentParser
from datetime import datetime
import json
import logging
import os
from os import path
import shlex
import socket
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from bui.db_interface import Session, db_fname
from bui.select_ticids.data_model import JobGroup, get_ticid_select_tables
from push_cache import add_host_arguments, push

_logger = logging.getLogger(__name__)

scripts_dir = path.dirname(path.abspath(__file__))
# Status values of the review tables (as in bui/select_ticids/urls.py).
status_names = {
    0: "unreviewed",
    1: "bad",
    2: "prepare",
    3: "fix",
    4: "sample",
    5: "changed_likelihood",
    6: "finished",
    7: "prepare_failed",
}
# Files in the host's home directory used to exchange data with it.
remote_snapshot_fname = ".tess_ebs_sync_snapshot.db"
remote_changes_fname = ".tess_ebs_sync_changes.json"


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-table",
        default="sample_prsa",
        help="The BUI table of the reviewed TICs (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline",
        default=path.join(scripts_dir, "TESS_EBs.sync_baseline.json"),
        help="The merged statuses of the previous sync (default: "
        "%(default)s).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Merge without a baseline (first sync): every status that "
        "differs is asked about.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change (conflicts resolved as this machine's), "
        "without asking or changing anything.",
    )
    parser.add_argument(
        "--apply-status-changes",
        metavar="CHANGES_JSON",
        help="Internal: run on the host to apply the given status changes.",
    )
    add_host_arguments(parser)
    return parser.parse_args()


def get_review_table(review_table):
    """Return the ORM class of the review table."""

    return get_ticid_select_tables(review_table, must_exist=True)[0]


def save_copy(database, copy_fname):
    """Save a consistent copy of an SQLite database file, even in use."""

    with sqlite3.connect(database) as source:
        with sqlite3.connect(copy_fname) as copy:
            source.backup(copy)
    _logger.info("Saved a copy of %s as %s.", database, copy_fname)


def read_state(database, review_table):
    """
    Return the statuses, job assignments and job groups of a database.

    Returns:
        dict:
            ``"statuses"``: {TIC: status}, ``"jobs"``: {TIC: (job_group,
            job_id)}, ``"job_groups"``: [{column: value}] of every job group.
    """

    select_table = get_review_table(review_table)
    engine = create_engine(
        f"sqlite:///file:{database}?mode=ro&uri=true", poolclass=NullPool
    )
    with sessionmaker(engine)() as session:
        rows = session.execute(
            select(
                select_table.id,
                select_table.status,
                select_table.job_group,
                select_table.job_id,
            )
        ).all()
        job_groups = [
            {
                column.name: getattr(group, column.name)
                for column in JobGroup.__table__.columns
            }
            for group in session.scalars(select(JobGroup))
        ]
    return {
        "statuses": {row.id: row.status for row in rows},
        "jobs": {row.id: (row.job_group, row.job_id) for row in rows},
        "job_groups": job_groups,
    }


def get_distinct_prefixes(names):
    """Return the shortest starts telling two machine names apart."""

    length = next(
        (
            length
            for length in range(1, min(map(len, names)) + 1)
            if names[0][:length] != names[1][:length]
        ),
        None,
    )
    if length is None:
        # One name starts the other: only the full names tell them apart.
        return names
    return tuple(name[:length] for name in names)


def ask(tic_id, statuses, names):
    """
    Ask which machine's status to keep for a conflict.

    The answer is the start of a machine's name, at least long enough to
    tell the two apart; an empty answer keeps this machine's. Anything else
    is asked again.

    Args:
        tic_id(int):    The target.

        statuses(tuple):    The baseline (None if none), this machine's and
            the host's status.

        names(tuple):    This machine's and the host's names.
    """

    base, *candidates = statuses
    prefixes = get_distinct_prefixes(names)
    question = (
        f"TIC {tic_id}: "
        + ", ".join(
            f"{name} {status_names.get(status, status)}"
            for name, status in zip(names, candidates)
        )
        + (
            ""
            if base is None
            else f" (last sync {status_names.get(base, base)})"
        )
        + f". Keep whose ({prefixes[0]}/{prefixes[1]}, default {names[0]})? "
    )
    if not sys.stdin.isatty():
        print(question + f"(not interactive: {names[0]})")
        return candidates[0]
    while True:
        answer = input(question).strip()
        if not answer:
            return candidates[0]
        for name, prefix, status in zip(names, prefixes, candidates):
            if answer.startswith(prefix) and name.startswith(answer):
                return status
        print(f"Answer with the start of {names[0]} or {names[1]}.")


def merge_statuses(base, here, host, names):
    """
    Return the merged statuses: {TIC: status} for TICs on both sides.

    Args:
        base(dict or None):    The baseline statuses (None: no baseline).

        here(dict):    This machine's statuses.

        host(dict):    The host's statuses.

        names(tuple or None):    This machine's and the host's names, to ask
            about conflicts; if None, keep this machine's status (and say
            so).
    """

    for tic_id in sorted(here.keys() ^ host.keys()):
        _logger.warning(
            "TIC %d is only %s; left alone.",
            tic_id,
            "here" if tic_id in here else "on the host",
        )
    merged = {}
    for tic_id in sorted(here.keys() & host.keys()):
        if here[tic_id] == host[tic_id]:
            merged[tic_id] = here[tic_id]
            continue
        base_status = None if base is None else base.get(tic_id)
        if base_status == here[tic_id]:
            merged[tic_id] = host[tic_id]
        elif base_status == host[tic_id]:
            merged[tic_id] = here[tic_id]
        elif names is not None:
            merged[tic_id] = ask(
                tic_id, (base_status, here[tic_id], host[tic_id]), names
            )
        else:
            print(
                f"Conflict, TIC {tic_id}: here "
                f"{status_names.get(here[tic_id])}, host "
                f"{status_names.get(host[tic_id])}: would keep here's."
            )
            merged[tic_id] = here[tic_id]
    return merged


def apply_status_changes(changes, review_table):
    """
    Apply status changes to this machine's database.

    All changes are applied in one transaction, after saving a timestamped
    copy of the database. Each change ({"id", "expected", "status"}) requires
    the status to still be ``expected``, the value the merge saw.

    Raises:
        RuntimeError:    If a status changed since the merge read it. Nothing
            is changed on this machine; rerunning the sync merges the new
            status.
    """

    if not changes:
        return
    save_copy(
        db_fname,
        f"{db_fname}.before-sync-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )
    select_table = get_review_table(review_table)
    with Session.begin() as session:  # pylint: disable=no-member
        for change in changes:
            if not session.execute(
                update(select_table)
                .where(
                    select_table.id == change["id"],
                    select_table.status == change["expected"],
                )
                .values(status=change["status"])
            ).rowcount:
                current = session.scalar(
                    select(select_table.status).where(
                        select_table.id == change["id"]
                    )
                )
                raise RuntimeError(
                    f"TIC {change['id']}: status changed on "
                    f"{socket.gethostname()} during the sync (now "
                    f"{status_names.get(current, current)}, the merge saw "
                    f"{status_names.get(change['expected'])}). Nothing was "
                    "changed on this machine; run the sync again."
                )


def apply_jobs_here(host_state, here_state, review_table):
    """Copy the host's job groups and job assignments into this database."""

    select_table = get_review_table(review_table)
    assignments = [
        {"id": tic_id, "job_group": job[0], "job_id": job[1]}
        for tic_id, job in host_state["jobs"].items()
        if tic_id in here_state["jobs"] and here_state["jobs"][tic_id] != job
    ]
    with Session.begin() as session:  # pylint: disable=no-member
        for group in host_state["job_groups"]:
            session.merge(JobGroup(**group))
        if assignments:
            session.execute(update(select_table), assignments)
    return len(assignments)


def run_on_host(config, command):
    """Run a command in the host's scripts/ directory; return its output."""

    remote_command = (
        f"cd {shlex.quote(path.join(config.remote_repository, 'scripts'))} "
        f"&& {command}"
    )
    _logger.info("Running on %s: %s", config.host, remote_command)
    # Errors on the host (stderr) go to the terminal.
    return subprocess.run(
        ["ssh", config.host, remote_command],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def apply_on_host(changes, config):
    """
    Apply status changes on the host (see `apply_status_changes()`).

    Raises:
        subprocess.CalledProcessError:    If the host did not apply them; its
            error is shown, and nothing was changed there.
    """

    with TemporaryDirectory() as temp_dir:
        changes_fname = path.join(temp_dir, "changes.json")
        with open(changes_fname, "w", encoding="ascii") as changes_file:
            json.dump(changes, changes_file)
        subprocess.run(
            [
                "rsync",
                "-a",
                changes_fname,
                f"{config.host}:{remote_changes_fname}",
            ],
            check=True,
        )
    run_on_host(
        config,
        shlex.join(
            [
                *shlex.split(config.remote_python),
                "sync_jobs_db.py",
                "--review-table",
                config.review_table,
                "--apply-status-changes",
            ]
        )
        # Outside the quoting, for the host's shell to expand.
        + f' "$HOME/{remote_changes_fname}"',
    )


def read_baseline(config):
    """Return the baseline statuses, None without one; refuse if missing."""

    if config.no_baseline:
        return None
    if not path.exists(config.baseline):
        raise RuntimeError(
            f"No sync baseline ({config.baseline}). For the first sync, run "
            "with --no-baseline."
        )
    with open(config.baseline, encoding="ascii") as baseline_file:
        return {
            int(tic_id): status
            for tic_id, status in json.load(baseline_file)["statuses"].items()
        }


def save_baseline(merged, config):
    """Save the merged statuses as the new baseline."""

    temp_fname = config.baseline + ".tmp"
    with open(temp_fname, "w", encoding="ascii") as baseline_file:
        json.dump(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "host": config.host,
                "statuses": {
                    str(tic_id): status for tic_id, status in merged.items()
                },
            },
            baseline_file,
        )
    os.replace(temp_fname, config.baseline)


def read_both_states(config):
    """
    Read both databases from consistent snapshots.

    Returns:
        (dict, dict, str):
            The host's and this machine's states (see `read_state()`), and
            the host's name.
    """

    with TemporaryDirectory() as temp_dir:
        host_snapshot = path.join(temp_dir, "host.db")
        here_snapshot = path.join(temp_dir, "here.db")
        # The host's name is printed last, after anything its shell prints.
        host_name = run_on_host(
            config,
            f"sqlite3 {path.basename(db_fname)} "
            f'".backup $HOME/{remote_snapshot_fname}" && hostname -s',
        ).split()[-1]
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{config.host}:{remote_snapshot_fname}",
                host_snapshot,
            ],
            check=True,
        )
        save_copy(db_fname, here_snapshot)
        return (
            read_state(host_snapshot, config.review_table),
            read_state(here_snapshot, config.review_table),
            host_name,
        )


def push_data(config):
    """
    Push the data of the targets to sample, before any status reaches the
    host (see push_cache.py); return whether to go on with the sync.

    If cache rows differ on the host, ask whether to go on (default: no).
    """

    push_status = push(config)
    if push_status == 0:
        return True
    if push_status != 3:
        print(f"Pushing the targets' data failed (exit {push_status}).")
        return False
    question = (
        "Cache rows differ on the host (listed above). Continue the status "
        "sync? (yes/no, default no) "
    )
    if config.dry_run:
        print(question + "(dry run: yes)")
        return True
    if not sys.stdin.isatty():
        print(question + "(not interactive: no)")
        return False
    while True:
        answer = input(question).strip()
        if answer in ("", "no"):
            return False
        if answer == "yes":
            return True
        print("Answer yes or no.")


def sync(config):
    """Push the targets' data, then merge both machines' databases."""

    base = read_baseline(config)
    if not push_data(config):
        return 1
    host_state, here_state, host_name = read_both_states(config)
    merged = merge_statuses(
        base,
        here_state["statuses"],
        host_state["statuses"],
        (
            None
            if config.dry_run
            else (socket.gethostname().split(".")[0], host_name)
        ),
    )
    to_host, to_here = (
        [
            {
                "id": tic_id,
                "expected": state["statuses"][tic_id],
                "status": status,
            }
            for tic_id, status in merged.items()
            if status != state["statuses"][tic_id]
        ]
        for state in (host_state, here_state)
    )
    print(
        f"Statuses: {len(to_host)} to change on the host, {len(to_here)} "
        "here."
    )
    for label, changes in (("host", to_host), ("here", to_here)):
        for change in changes:
            print(
                f"   {label}: TIC {change['id']} "
                f"{status_names.get(change['expected'])} -> "
                f"{status_names.get(change['status'])}"
            )
    if config.dry_run:
        return 0
    if to_host:
        apply_on_host(to_host, config)
    apply_status_changes(to_here, config.review_table)
    print(
        "Job assignments updated here: "
        f"{apply_jobs_here(host_state, here_state, config.review_table)}."
    )
    save_baseline(merged, config)
    return 0


def main(config):
    """Sync, or apply status changes when run on the host."""

    if config.apply_status_changes:
        with open(
            config.apply_status_changes, encoding="ascii"
        ) as changes_file:
            apply_status_changes(json.load(changes_file), config.review_table)
        return 0
    return sync(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(parse_command_line()))
