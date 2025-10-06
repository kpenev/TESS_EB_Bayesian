"""On-the-fly creation of tables to track TIC selection."""

import re
from glob import glob
from os import path
from datetime import datetime

from sqlalchemy import (
    Table,
    Column,
    Integer,
    TIMESTAMP,
    text,
    inspect,
    ForeignKey,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from bui.db_interface import db_engine, Session


class SelectTICIDBase(  # pylint: disable=too-few-public-methods
    DeclarativeBase
):
    """Base class for tables that track TIC ID selection."""


class JobGroup(SelectTICIDBase):  # pylint: disable=too-few-public-methods
    """The group of jobs used to sample a collection of TICs."""

    __tablename__ = "job_groups"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Unique identifier for the job collection",
    )
    select_tic_table: Mapped[str] = mapped_column(
        nullable=False,
        doc="The table of TICs this job group is sampling from.",
    )
    hpc: Mapped[str] = mapped_column(
        nullable=False,
        doc="The HPC system this job collection is/was running on.",
    )
    nodes_per_job: Mapped[int] = mapped_column(
        doc="The number of nodes each job of this collection uses."
    )
    num_jobs: Mapped[int] = mapped_column(
        nullable=False, doc="The number of jobs in the job collection."
    )
    description: Mapped[str | None] = mapped_column(
        doc="User supplied description of the job group."
    )
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        default=func.now(),  # pylint: disable=not-callable
        onupdate=func.now(),  # pylint: disable=not-callable
        doc="When record was last changed",
    )


if not inspect(db_engine).has_table(JobGroup.__tablename__):
    JobGroup.__table__.create(db_engine)


def get_ticids(plot_dir):
    """Return the TIC IDs with corresponding plots in the given directory."""

    plot_name_rex = re.compile(r"tess(?P<tic>[0-9]*)\.(?P<ext>[a-zA-Z]+)$")
    for plot_fname in glob(path.join(plot_dir, "*")):
        parsed = plot_name_rex.match(path.basename(plot_fname))
        if parsed and parsed["ext"] in ("png", "jpg", "jpeg", "pdf"):
            yield int(parsed["tic"]), plot_fname


def get_ticid_select_table(
    tablename, plot_dirs=(), require_all=False, must_exist=False
):
    """Create a table for tracking TIC ID selection with given name."""

    # pylint: disable=too-few-public-methods
    class Result(SelectTICIDBase):
        """The table for tracking TIC ID selection."""

        __table__ = Table(
            tablename,
            SelectTICIDBase.metadata,
            Column(
                "id", Integer, primary_key=True, doc="The TIC ID to consider."
            ),
            Column(
                "status",
                Integer,
                doc="Status assigned to the TIC ID (selection dependent).",
            ),
            Column("rendered", Integer, doc="1 - rendered, 0 - not"),
            Column(
                "job_group",
                Integer,
                ForeignKey(
                    "job_groups.id", onupdate="CASCADE", ondelete="RESTRICT"
                ),
            ),
            Column(
                "job_id",
                Integer,
                doc="The ID of the job this TIC is assigned to.",
            ),
            Column(
                "timestamp",
                TIMESTAMP,
                SelectTICIDBase.metadata,
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
                doc="When record was last changed",
            ),
            keep_existing=True,
        )

    # pylint: enable=too-few-public-methods

    if not inspect(db_engine).has_table(tablename):
        assert (
            not must_exist
        ), f"Table {tablename} must already exist, not creating!"
        assert plot_dirs is not None
        Result.__table__.create(db_engine)
        plot_tic_ids = None
        for plot_dir in plot_dirs:
            if plot_tic_ids is None:
                plot_tic_ids = set(tic_id for tic_id, _ in get_ticids(plot_dir))
            else:
                getattr(
                    plot_tic_ids,
                    "intersection_update" if require_all else "update",
                )(set(tic_id for tic_id, _ in get_ticids(plot_dir)))
        if plot_tic_ids is not None:
            # False positive
            # pylint: disable=no-member
            with Session.begin() as db_session:
                # pylint: enable=no-member
                for tic_id in plot_tic_ids:
                    db_session.add(Result(id=tic_id, status=0, rendered=1))

    return Result
