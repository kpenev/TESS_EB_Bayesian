"""On-the-fly creation of tables to track TIC selection."""

import re
from glob import glob
from os import path

from sqlalchemy import Table, Column, Integer, String, TIMESTAMP, text, inspect
from sqlalchemy.orm import DeclarativeBase

from bui.db_interface import db_engine, Session


# pylint: disable=too-few-public-methods
class SelectTICIDBase(DeclarativeBase):
    """Base class for tables that track TIC ID selection."""


# pylint: enable=too-few-public-methods


def get_ticids(plot_dir):
    """Return the TIC IDs with corresponding plots in the given directory."""

    plot_name_rex = re.compile(r"tess(?P<tic>[0-9]*)\.(?P<ext>[a-zA-Z]+)$")
    for plot_fname in glob(path.join(plot_dir, "*")):
        parsed = plot_name_rex.match(path.basename(plot_fname))
        if parsed and parsed["ext"] in ("png", "jpg", "jpeg", "pdf"):
            yield int(parsed["tic"]), plot_fname


def get_ticid_select_table(tablename, plot_dirs=(), require_all=False):
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
                "hpc", String, doc="The HPC system this TIC is assigned to."
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
