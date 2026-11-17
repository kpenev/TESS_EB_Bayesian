#!/usr/bin/env python3

"""
Apply new re-preparation requests, then list the targets to prepare.

Run by prepare_batch.sh on the preparation machine before each batch. The
machine running the review puts snapshots of its databases in
``EXCHANGE_DIR/to_prep/`` (sync_jobs_db.py): ``TESS_EBs.db`` (statuses and
re-preparation requests) and ``mcmc_cache.sqlite`` (data exclusions and bad
SED settings). This script:

1. Applies each re-preparation request of the snapshot not applied yet (the
   ids applied are recorded in ``OUTPUT_DIR/applied_prepare_requests.txt``):

   * the target's data exclusions and bad SED settings in this machine's
     cache are set to the snapshot's, keeping this machine's SED values; if
     the exclusions changed, its BLS row is deleted, so that the BLS search
     is redone on the new selection of data;
   * the outputs from the request's restart step on are moved to
     ``OUTPUT_DIR/<TIC>/superseded-<time>/``, with its samples file:
     prepare_sampling.sh skips a step whose output exists.

2. Prints, one per line, the targets to prepare: status prepare in the
   snapshot and no samples file yet (``OUTPUT_DIR/samples_seeded/``), those
   in the ``--priority`` list first.

Without a database snapshot, the ``--priority`` list is printed as given
(prepare exactly those targets); without either, it fails.

The snapshots are read with SQLAlchemy reflection, not the BUI modules, which
would open this machine's own ``scripts/TESS_EBs.db``.
"""

from argparse import ArgumentParser
from datetime import datetime
from glob import glob
import logging
import os
from os import path
import shutil
import sys

from sqlalchemy import MetaData, Table, create_engine, delete, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cache_interface import (
    CacheSession,
    CachedBLS,
    CachedSED,
    ExcludeDataTable,
)

_logger = logging.getLogger(__name__)

status_prepare = 2
# The bad SED settings set in the review (as in install_prepared.py).
sed_settings = ("bad_sed_threshold", "bad_sed_penalty", "bad_sed_sharpness")
# The outputs (in OUTPUT_DIR/<TIC>/) kept when the preparation restarts from
# each step: those of the steps before it.
kept_outputs = {
    "sed_grid_search": (),
    "sed_seeded_fits": (
        "sed_grid_search_{tic_id}.ecsv",
        "sed_grid_search.log",
        "sed_grid_corner_{tic_id}.*",
        "plot_sed_grid_search.log",
    ),
}


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="The OUTPUT_DIR of prepare_batch.sh: each target's outputs are "
        "in OUTPUT_DIR/<TIC>/, its samples file in OUTPUT_DIR/samples_seeded/.",
    )
    parser.add_argument(
        "--exchange-dir",
        default=os.environ.get("EXCHANGE_DIR"),
        help="The folder shared with the machine running the review "
        "(default: the EXCHANGE_DIR environment variable, %(default)s).",
    )
    parser.add_argument(
        "--priority",
        help="A file listing TICs to prepare first, one per line (blank "
        "lines and lines starting with # are skipped).",
    )
    parser.add_argument(
        "--review-table",
        default="sample_prsa",
        help="The BUI table of the reviewed TICs (default: %(default)s).",
    )
    return parser.parse_args()


def read_tic_list(fname):
    """Return the TICs listed in a file, in order."""

    with open(fname, encoding="ascii") as tic_file:
        return [
            int(line.split()[0])
            for line in tic_file
            if line.strip() and not line.lstrip().startswith("#")
        ]


def get_samples_fname(output_dir, tic_id):
    """Return the samples file prepare_batch.sh writes for a target."""

    return path.join(output_dir, "samples_seeded", f"tess{tic_id}_samples.h5")


def open_snapshot(fname):
    """Return a read-only engine for a database snapshot."""

    return create_engine(
        f"sqlite:///file:{fname}?mode=ro&uri=true", poolclass=NullPool
    )


def get_new_requests(engine, review_table, applied):
    """Return the snapshot's re-preparation requests not applied yet."""

    requests = Table("prepare_requests", MetaData(), autoload_with=engine)
    with engine.connect() as connection:
        return [
            request
            for request in connection.execute(
                select(requests)
                .where(requests.c.select_tic_table == review_table)
                .order_by(requests.c.id)
            ).mappings()
            if request["id"] not in applied
        ]


def update_cache(tic_id, cache_snapshot):
    """
    Set a target's exclusions and bad SED settings to the snapshot's.

    Returns:
        bool:    Whether the exclusions changed (then the BLS row, computed
            from the old selection of data, is deleted).
    """

    with sessionmaker(open_snapshot(cache_snapshot))() as incoming:
        exclusions = set(
            incoming.scalars(
                select(ExcludeDataTable.exclude).where(
                    ExcludeDataTable.tic_id == tic_id
                )
            )
        )
        incoming_sed = incoming.get(CachedSED, tic_id)
        settings = (
            {}
            if incoming_sed is None
            else {name: getattr(incoming_sed, name) for name in sed_settings}
        )
    with CacheSession.begin() as cache:  # pylint: disable=no-member
        changed = exclusions != set(
            cache.scalars(
                select(ExcludeDataTable.exclude).where(
                    ExcludeDataTable.tic_id == tic_id
                )
            )
        )
        if changed:
            cache.execute(
                delete(ExcludeDataTable).where(
                    ExcludeDataTable.tic_id == tic_id
                )
            )
            cache.add_all(
                ExcludeDataTable(tic_id=tic_id, exclude=exclude)
                for exclude in exclusions
            )
            cache.execute(delete(CachedBLS).where(CachedBLS.tic_id == tic_id))
        sed = cache.get(CachedSED, tic_id)
        if sed is None:
            cache.add(CachedSED(tic_id=tic_id, **settings))
        else:
            for name, value in settings.items():
                setattr(sed, name, value)
    return changed


def move_outputs(tic_id, restart_step, output_dir):
    """Move a target's outputs from the restart step on out of the way."""

    target_dir = path.join(output_dir, str(tic_id))
    kept = {
        fname
        for pattern in kept_outputs[restart_step]
        for fname in glob(path.join(target_dir, pattern.format(tic_id=tic_id)))
    }
    moving = [
        fname
        for fname in glob(path.join(target_dir, "*"))
        if fname not in kept
        and not path.basename(fname).startswith("superseded-")
    ] + glob(get_samples_fname(output_dir, tic_id) + "*")
    if not moving:
        return
    superseded_dir = path.join(
        target_dir, f"superseded-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(superseded_dir)
    for fname in moving:
        shutil.move(fname, superseded_dir)
    _logger.info(
        "TIC %d: moved %d outputs to %s.",
        tic_id,
        len(moving),
        superseded_dir,
    )


def apply_requests(snapshots, review_table, output_dir):
    """Apply the re-preparation requests not applied yet."""

    record_fname = path.join(output_dir, "applied_prepare_requests.txt")
    applied = (
        set(read_tic_list(record_fname)) if path.exists(record_fname) else set()
    )
    for request in get_new_requests(
        open_snapshot(snapshots["database"]), review_table, applied
    ):
        if request["restart_step"] not in kept_outputs:
            raise ValueError(
                f"Unknown restart step {request['restart_step']!r} of request "
                f"{request['id']} (TIC {request['tic_id']})."
            )
        exclusions_changed = update_cache(request["tic_id"], snapshots["cache"])
        move_outputs(request["tic_id"], request["restart_step"], output_dir)
        with open(record_fname, "a", encoding="ascii") as record:
            record.write(f"{request['id']}\n")
        _logger.info(
            "Applied request %d: TIC %d from %s (%s)%s.",
            request["id"],
            request["tic_id"],
            request["restart_step"],
            request["changes"],
            (
                "; exclusions changed, BLS row deleted"
                if exclusions_changed
                else ""
            ),
        )


def get_pending(database_snapshot, review_table, output_dir):
    """Return the targets with status prepare not prepared yet."""

    engine = open_snapshot(database_snapshot)
    review = Table(review_table, MetaData(), autoload_with=engine)
    with engine.connect() as connection:
        tic_ids = connection.scalars(
            select(review.c.id)
            .where(review.c.status == status_prepare)
            .order_by(review.c.id)
        ).all()
    return [
        tic_id
        for tic_id in tic_ids
        if not path.exists(get_samples_fname(output_dir, tic_id))
    ]


def main(config):
    """Apply the requests and print the targets to prepare."""

    priority = read_tic_list(config.priority) if config.priority else []
    snapshots = {
        "database": path.join(
            config.exchange_dir or "", "to_prep", "TESS_EBs.db"
        ),
        "cache": path.join(
            config.exchange_dir or "", "to_prep", "mcmc_cache.sqlite"
        ),
    }
    if not config.exchange_dir or not path.exists(snapshots["database"]):
        missing = (
            snapshots["database"]
            if config.exchange_dir
            else "EXCHANGE_DIR not set"
        )
        if not priority:
            _logger.error(
                "No database snapshot (%s) and no --priority list: nothing "
                "says what to prepare.",
                missing,
            )
            return 1
        _logger.warning(
            "No database snapshot (%s): preparing the --priority list as "
            "given.",
            missing,
        )
        print("\n".join(map(str, priority)))
        return 0
    apply_requests(snapshots, config.review_table, config.output_dir)
    pending = get_pending(
        snapshots["database"], config.review_table, config.output_dir
    )
    pending_set = set(pending)
    for tic_id in priority:
        if tic_id not in pending_set:
            _logger.warning(
                "TIC %d is listed in --priority but is not pending (status "
                "not prepare, or already prepared); skipped.",
                tic_id,
            )
    priority_set = set(priority)
    ordered = [tic_id for tic_id in priority if tic_id in pending_set] + [
        tic_id for tic_id in pending if tic_id not in priority_set
    ]
    _logger.info("%d targets to prepare.", len(ordered))
    print("\n".join(map(str, ordered)))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(parse_command_line()))
