"""On-the-fly creation of tables to track TIC selection."""

import re
from glob import glob
from os import path
from datetime import datetime

from sqlalchemy import (
    TIMESTAMP,
    inspect,
    ForeignKey,
    func,
    delete,
    insert,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from bui.db_interface import db_engine, Session
from .path_util import parse_render_dir


class SelectTICIDBase(  # pylint: disable=too-few-public-methods
    DeclarativeBase
):
    """Base class for tables that track TIC ID selection."""

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Unique identifier for the job collection",
    )
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        default=func.now(),  # pylint: disable=not-callable
        onupdate=func.now(),  # pylint: disable=not-callable
        doc="When record was last changed",
    )


class JobGroup(SelectTICIDBase):  # pylint: disable=too-few-public-methods
    """The group of jobs used to sample a collection of TICs."""

    __tablename__ = "job_groups"

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


if not inspect(db_engine).has_table(JobGroup.__tablename__):
    JobGroup.__table__.create(db_engine)


def get_ticids(plot_dir):
    """Return the TIC IDs with corresponding plots in the given directory."""

    plot_name_rex = re.compile(r"tess(?P<tic>[0-9]*)\.(?P<ext>[a-zA-Z]+)$")
    for plot_fname in glob(path.join(plot_dir, "*")):
        parsed = plot_name_rex.match(path.basename(plot_fname))
        if parsed and parsed["ext"] in ("png", "jpg", "jpeg", "pdf"):
            yield int(parsed["tic"])


def set_rendered(RenderedTable, plot_dirs):  # pylint: disable=invalid-name
    """Update the rendered to match the plots available."""

    with Session.begin() as db_session:  # pylint: disable=no-member
        db_session.execute(delete(RenderedTable))

        for plot_dir in plot_dirs:
            print("Setting rendered for plots in directory:", plot_dir)
            db_session.execute(
                insert(RenderedTable),
                [
                    {"id": tic_id, "plot": parse_render_dir(plot_dir)[1]}
                    for tic_id in get_ticids(plot_dir)
                ],
            )


def get_ticid_select_tables(
    tablename, plot_dirs=(), must_exist=False, refresh_rendered=False
):
    """Create a table for tracking TIC ID selection with given name."""

    # pylint: disable=too-few-public-methods
    class SelectTable(SelectTICIDBase):
        """The table for tracking TIC ID selection."""

        __tablename__ = tablename
        __table_args__ = {"extend_existing": True}

        status: Mapped[int] = mapped_column(
            doc="Status assigned to the TIC ID (selection dependent).",
        )
        rendered: Mapped[int] = mapped_column(doc="1 - rendered, 0 - not")

        job_group: Mapped[int] = mapped_column(
            ForeignKey(
                "job_groups.id", onupdate="CASCADE", ondelete="RESTRICT"
            ),
        )

        job_id: Mapped[int] = mapped_column(
            doc="The ID of the job this TIC is assigned to.",
        )

        def __str__(self):
            # pylint: disable=no-member
            return (
                f"{self.id}: status={self.status}, job grp={self.job_group}, "
                f"job={self.job_id} ({self.timestamp})"
            )
            # pylint: enable=no-member

    class RenderedTable(SelectTICIDBase):
        """The table tracking which plots are available."""

        __tablename__ = tablename + "_rendered"
        __table_args__ = {"extend_existing": True}

        plot: Mapped[str] = mapped_column(
            primary_key=True,
            doc="The type of plot available for this TIC",
        )

    # pylint: enable=too-few-public-methods

    if not inspect(db_engine).has_table(tablename):
        assert (
            not must_exist
        ), f"Table {tablename} must already exist, not creating!"
        assert plot_dirs is not None
        SelectTable.__table__.create(db_engine)

    if not inspect(db_engine).has_table(RenderedTable.__tablename__):
        RenderedTable.__table__.create(db_engine)
        refresh_rendered = True
    if refresh_rendered:
        set_rendered(RenderedTable, plot_dirs)

    return SelectTable, RenderedTable
