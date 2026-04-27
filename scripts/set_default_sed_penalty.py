"""Set default bad-SED threshold/penalty in cache for unstarted TIC IDs."""

from argparse import ArgumentParser
from os import path

from sqlalchemy import select

from bui.db_interface import Session  # pylint: disable=import-error
from bui.select_ticids.data_model import (  # pylint: disable=import-error
    JobGroup,
    get_ticid_select_tables,
)
from cache_interface import CacheSession, CachedSED
import paths


def set_default_sed_penalty(tic_id, dry_run=False):
    """Set default bad-SED penalty for TIC ID if sampling has not started.

    Only updates existing cache entries with NULL values — if no entry exists,
    does nothing (sampling will apply defaults correctly in memory).

    Returns a string describing what was done (for logging).
    """

    samples_fname = paths.samples.format(tic_id=tic_id)
    if path.exists(samples_fname):
        return f"TIC {tic_id}: sampling started, skipped"

    # pylint: disable=no-member
    with CacheSession.begin() as session:
        # pylint: enable=no-member
        cached_sed = session.execute(
            select(CachedSED).filter_by(tic_id=tic_id)
        ).scalar_one_or_none()

        if cached_sed is None:
            return f"TIC {tic_id}: no cached SED, skipped"

        changed = []
        if cached_sed.bad_sed_threshold is None:
            if not dry_run:
                cached_sed.bad_sed_threshold = (
                    CachedSED.default_bad_sed_threshold
                )
            changed.append(f"threshold={CachedSED.default_bad_sed_threshold}")
        if cached_sed.bad_sed_penalty is None:
            if not dry_run:
                cached_sed.bad_sed_penalty = CachedSED.default_bad_sed_penalty
            changed.append(f"penalty={CachedSED.default_bad_sed_penalty}")

        if changed:
            verb = "would set" if dry_run else "set"
            return f"TIC {tic_id}: {verb} {', '.join(changed)}"
        return f"TIC {tic_id}: already has threshold and penalty, skipped"


def parse_command_line():
    """Return parsed command line arguments."""

    with Session.begin() as db_session:  # pylint: disable=no-member
        job_groups = db_session.scalars(select(JobGroup.id)).all()

    parser = ArgumentParser(
        description="Set default bad-SED threshold/penalty for unstarted TICs."
    )
    parser.add_argument(
        "job_group",
        choices=[str(g) for g in job_groups],
        help="Job group whose TIC IDs to process.",
    )
    parser.add_argument(
        "--status",
        type=int,
        default=2,
        help="Only process TICs with this status (default: 2).",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="If given, restrict to TICs assigned to this job ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without modifying the database.",
    )
    return parser.parse_args()


def main(config):
    """Avoid polluting global namespace."""

    job_group_id = int(config.job_group)

    with Session.begin() as db_session:  # pylint: disable=no-member
        job_group = db_session.scalar(
            select(JobGroup).filter_by(id=job_group_id)
        )
        SelectTICTable, _ = get_ticid_select_tables(
            job_group.select_tic_table, must_exist=True
        )
        query = select(SelectTICTable).filter_by(
            job_group=job_group_id, status=config.status
        )
        if config.job_id is not None:
            query = query.filter_by(job_id=config.job_id)
        tic_ids = [entry.id for entry in db_session.scalars(query).all()]

    for tic_id in tic_ids:
        print(set_default_sed_penalty(tic_id, dry_run=config.dry_run))

if __name__ == "__main__":
    main(parse_command_line())
