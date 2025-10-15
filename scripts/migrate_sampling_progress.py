"""Migrate the information from old-style DB to new."""

from datetime import datetime

from sqlalchemy.orm import sessionmaker, mapped_column, Mapped
from sqlalchemy import create_engine, select, TIMESTAMP, func
from sqlalchemy.pool import NullPool

from bui.select_ticids.data_model import SelectTICIDBase, get_ticid_select_table


def start_db_session(db_fname):
    """Return session maker for given DB file."""

    db_engine = create_engine(
        f"sqlite:///{db_fname}?timeout=100&uri=true",
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        poolclass=NullPool,
    )

    return sessionmaker(  # pylint:disable=no-member
        db_engine, expire_on_commit=False
    ).begin()


# pylint: disable=too-few-public-methods
class OldTICTable(SelectTICIDBase):
    """The table for tracking TIC ID selection."""

    __tablename__ = "sampling"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="TIC ID of the binary",
    )

    flag: Mapped[int] = mapped_column(
        doc="Status assigned to the TIC ID (selection dependent).",
    )

    rendered: Mapped[int] = mapped_column(doc="1 - rendered, 0 - not")

    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        default=func.now(),  # pylint: disable=not-callable
        onupdate=func.now(),  # pylint: disable=not-callable
        doc="When record was last changed",
    )


# pylint: enable=too-few-public-methods

NewTICTable = get_ticid_select_table("sample_prsa", must_exist=True)


def record_progress(old_flag, old_flag_str, new_status, db):
    """Update the new DB entry per old."""

    for old in (
        db["old"].scalars(select(OldTICTable).filter_by(flag=old_flag)).all()
    ):
        new = db["new"].scalar(select(NewTICTable).filter_by(id=old.id))
        if new.job_id is not None:
            print(f"Sampling {old_flag_str} {new} again!")
        else:
            new.status = new_status


def migrate():
    """Perform the migration."""

    with start_db_session(
        "/mnt/md2/TESS_EBs/first1000/ls6/samples.9_on/TESS_EBs.db"
    ) as old_db, start_db_session(
        "/home/kpenev/projects/git/TESS_EB_Bayesian/scripts/TESS_EBs.db"
    ) as new_db:
        db = {"old": old_db, "new": new_db}
        record_progress(1, "old finished", 4, db)
        record_progress(2, "old LS6 sampling", 5, db)
        record_progress(3, "old Juno sampling", 6, db)


if __name__ == "__main__":
    migrate()
