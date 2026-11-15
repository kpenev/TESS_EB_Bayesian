#!/usr/bin/env python3

"""
Merge another machine's likelihood cache into this one's, for given targets.

The likelihood cache (``data/mcmc_cache.sqlite``) holds per target the BLS
search result (``bls``; the walkers' coordinates are defined relative to it),
the SED (``sed``; computing it queries Gaia and other archives) and the
excluded data (``exclude_data``). A target must be sampled with the same rows
it was prepared with. The machine running the review holds them for every
prepared target (install_prepared.py merges them in from the preparation
machine), and sends a copy of its whole cache to the machine that samples.
There, this script:

* checks that the tables have the same columns in both caches,
* saves a timestamped copy of this cache next to it,
* adds the incoming ``bls``, ``sed`` and ``exclude_data`` rows of the given
  targets this cache does not have yet (refusing if the incoming cache has no
  ``bls`` or ``sed`` row for one of them: this machine would compute its
  own). Given targets this cache already has (a ``bls`` or ``sed`` row) are
  left alone and compared like the others, since one may be being sampled:
  changing its rows would change the likelihood under the running chain.
  ``--replace-existing`` replaces them too, for when none is running,
* compares the rows of every other target, and reports those that differ or
  that only this cache has: a target sampled with rows other than those it
  was prepared with would be sampled from the wrong starting point. A target
  this cache has BLS or SED rows for but no exclusion rows is compared as
  excluding nothing. Targets only in the incoming cache (not sent here yet)
  are counted but are not a mismatch.

Exit status 0 if all other targets match, 3 if some differ or are only here
(the given targets are replaced either way, unless ``--dry-run``).
"""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import logging
from os import chdir, path
import sqlite3
import sys

from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import paths
from cache_interface import (
    CacheSession,
    CachedBLS,
    CachedSED,
    ExcludeDataTable,
    db_engine,
)

_logger = logging.getLogger(__name__)

_models = (CachedBLS, CachedSED, ExcludeDataTable)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "incoming", help="The other machine's cache (a copy of it)."
    )
    parser.add_argument(
        "tic_ids",
        nargs="*",
        type=int,
        help="The targets whose rows to replace with the incoming ones.",
    )
    parser.add_argument(
        "--tics-file",
        help="A file listing more targets to replace, one TIC per line.",
    )
    parser.add_argument(
        "--review-table",
        default="sample_prsa",
        help="The BUI table assigning targets to sampling jobs; a warning is "
        "issued for each target added that is already assigned to one "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Also replace the rows of given targets this cache already has. "
        "Only when none of them is being sampled: changing a target's rows "
        "changes the likelihood under a running chain.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check and compare only; change nothing and save no copy.",
    )
    parser.add_argument(
        "--max-listed",
        type=int,
        default=20,
        help="List at most this many differing targets per table "
        "(default: %(default)s).",
    )
    return parser.parse_args()


def get_tic_ids(config):
    """Return the set of targets to replace, from arguments and file."""

    tic_ids = set(config.tic_ids)
    if config.tics_file is not None:
        with open(config.tics_file, encoding="utf-8") as tics_file:
            tic_ids.update(
                int(line.split()[0])
                for line in tics_file
                if line.strip() and not line.lstrip().startswith("#")
            )
    return tic_ids


def get_column_names(model):
    """Return the names of a model's columns, in table order."""

    return [column.name for column in model.__table__.columns]


def check_columns(incoming_engine):
    """Raise if any table's columns differ between the two caches."""

    for model in _models:
        table = model.__tablename__
        here, incoming = (
            {column["name"] for column in inspect(engine).get_columns(table)}
            for engine in (db_engine, incoming_engine)
        )
        if here != incoming:
            raise RuntimeError(
                f"Table {table} differs between the caches: only here "
                f"{sorted(here - incoming)}, only incoming "
                f"{sorted(incoming - here)}."
            )


def get_rows(session, model):
    """Return a table's rows as tuples, grouped by TIC."""

    names = get_column_names(model)
    result = {}
    for entry in session.scalars(select(model)):
        result.setdefault(entry.tic_id, set()).add(
            tuple(getattr(entry, name) for name in names)
        )
    return result


def check_incoming_rows(incoming, tic_ids):
    """Raise if the incoming cache lacks a bls or sed row of a target."""

    for model in (CachedBLS, CachedSED):
        missing = sorted(tic_ids - set(incoming[model]))
        if missing:
            raise RuntimeError(
                f"The incoming cache has no {model.__tablename__} row for "
                f"TIC(s) {missing}."
            )


def save_copy(copy_fname=None):
    """
    Save a consistent copy of this cache, even while it is in use.

    Args:
        copy_fname(str or None):    Where to save it. By default next to the
            cache, with a timestamp appended to its name.

    Returns:
        str:    The name of the copy.
    """

    if copy_fname is None:
        copy_fname = (
            f"{paths.cache_db}.bak-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    with db_engine.connect() as connection:
        with sqlite3.connect(copy_fname) as copy:
            connection.connection.driver_connection.backup(copy)
    _logger.info("Saved a copy of %s as %s.", paths.cache_db, copy_fname)
    return copy_fname


def replace_rows(incoming, tic_ids):
    """Replace the given targets' rows with the incoming ones."""

    with CacheSession.begin() as session:  # pylint: disable=no-member
        for model in _models:
            names = get_column_names(model)
            removed = session.execute(
                delete(model).where(model.tic_id.in_(tic_ids))
            ).rowcount
            added = [
                model(**dict(zip(names, row)))
                for tic_id in tic_ids
                for row in incoming[model].get(tic_id, ())
            ]
            session.add_all(added)
            _logger.info(
                "%s: replaced %d row(s) with %d incoming.",
                model.__tablename__,
                removed,
                len(added),
            )


def get_compared_rows(here, incoming, model):
    """
    Return the rows of a table to compare, from this and the incoming cache.

    For a target this cache has a BLS or SED row for, having no exclusion
    rows means excluding nothing, so it is compared as an empty set.
    """

    here_rows, incoming_rows = here[model], incoming[model]
    if model is not ExcludeDataTable:
        return here_rows, incoming_rows
    known = here[CachedBLS].keys() | here[CachedSED].keys()
    return (
        {tic_id: rows.get(tic_id, set()) for tic_id in known | rows.keys()}
        for rows in (here_rows, incoming_rows)
    )


def report_differing_columns(here_rows, incoming_rows, model, differ):
    """Print how many of the differing targets differ in each column."""

    column_counts = Counter(
        name
        for tic_id in differ
        for name, value_here, value_incoming in zip(
            get_column_names(model),
            next(iter(here_rows[tic_id])),
            next(iter(incoming_rows[tic_id])),
        )
        if value_here != value_incoming
    )
    print(
        "    columns differing (number of targets): "
        + ", ".join(
            f"{name} ({count})" for name, count in column_counts.most_common()
        )
    )


def compare_rows(here, incoming, skip, max_listed):
    """Report the targets other than ``skip`` whose rows differ."""

    any_differ = False
    for model in _models:
        here_rows, incoming_rows = get_compared_rows(here, incoming, model)
        common = (here_rows.keys() & incoming_rows.keys()) - skip
        differ = sorted(
            tic_id
            for tic_id in common
            if here_rows[tic_id] != incoming_rows[tic_id]
        )
        only_here = sorted(here_rows.keys() - incoming_rows.keys() - skip)
        only_incoming = sorted(incoming_rows.keys() - here_rows.keys() - skip)
        print(
            f"{model.__tablename__}: {len(common)} other targets in both "
            f"caches, {len(differ)} differ; {len(only_here)} only here, "
            f"{len(only_incoming)} only incoming."
        )
        if differ and model is not ExcludeDataTable:
            report_differing_columns(here_rows, incoming_rows, model, differ)
        for label, listed in (
            ("differ", differ),
            ("only here", only_here),
            ("only incoming", only_incoming),
        ):
            if listed:
                print(
                    f"    {label}: {listed[:max_listed]}"
                    + (" ..." if len(listed) > max_listed else "")
                )
        # Targets only in the incoming cache have not been sent here yet.
        any_differ = any_differ or bool(differ or only_here)
    return any_differ


def warn_assigned(tic_ids, review_table):
    """
    Warn about targets already assigned to a sampling job.

    Jobs are created only for targets that passed the review after their
    preparation, so a newly prepared target being added should not have a
    job yet: one that does may be sampled with other rows than it was
    prepared with.
    """

    # The BUI database is opened relative to the working directory.
    chdir(path.dirname(path.abspath(__file__)))
    # pylint: disable=import-outside-toplevel
    from bui.db_interface import Session
    from bui.select_ticids.data_model import get_ticid_select_tables

    # pylint: enable=import-outside-toplevel

    select_table, _ = get_ticid_select_tables(review_table, must_exist=True)
    with Session() as db_session:  # pylint: disable=not-callable
        assigned = db_session.execute(
            select(
                select_table.id, select_table.job_group, select_table.job_id
            ).where(
                select_table.id.in_(tic_ids),
                select_table.job_group.is_not(None)
                | select_table.job_id.is_not(None),
            )
        ).all()
    for tic_id, job_group, job_id in assigned:
        _logger.warning(
            "TIC %d is already assigned to job group %s, job %s.",
            tic_id,
            job_group,
            job_id,
        )


def main(config):
    """Merge the incoming cache and report differences."""

    tic_ids = get_tic_ids(config)
    incoming_engine = create_engine(
        f"sqlite:///file:{config.incoming}?mode=ro&uri=true",
        poolclass=NullPool,
    )
    check_columns(incoming_engine)
    with sessionmaker(incoming_engine)() as session:
        incoming = {model: get_rows(session, model) for model in _models}
    with CacheSession() as session:
        here = {model: get_rows(session, model) for model in _models}
    check_incoming_rows(incoming, tic_ids)
    existing = tic_ids & (here[CachedBLS].keys() | here[CachedSED].keys())
    if existing and not config.replace_existing:
        print(
            f"Not replacing {len(existing)} TICs this cache already has "
            f"(compared below instead): {sorted(existing)}"
        )
        tic_ids = tic_ids - existing
    if tic_ids:
        warn_assigned(tic_ids, config.review_table)
        if config.dry_run:
            print(f"Dry run: would replace the rows of {len(tic_ids)} TICs.")
        else:
            save_copy()
            replace_rows(incoming, tic_ids)
            print(f"Replaced the rows of {len(tic_ids)} TICs.")
            with CacheSession() as session:
                here = {model: get_rows(session, model) for model in _models}
    return 3 if compare_rows(here, incoming, tic_ids, config.max_listed) else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(parse_command_line()))
