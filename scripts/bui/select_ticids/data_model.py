from sqlalchemy import Column, Integer, TIMESTAMP, text

from sqlalchemy.orm import DeclarativeBase


class SelectTICIDBase(DeclarativeBase):
    """Model for keeping track of user selection of TIC IDs."""

    id = Column(Integer, primary_key=True, doc="The TIC ID to consider.")

    flag = Column(Integer, doc="-1 - rejected, 0 - pending, 1 - selected")
    rendered = Column(Integer, doc="1 - rendered, 0 - not")

    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        doc="When record was last changed",
    )


def get_ticid_select_table(tablename):
    """Create a table for tracking TIC ID selection with given name."""

    for mapper in SelectTICIDBase.registry.mappers:
        candidate = mapper.class_
        if (
            not candidate.__name__.startswith("_")
            and getattr(candidate, "__tablename__", "") == tablename
        ):
            return candidate

    class SelectTICIDs(SelectTICIDBase):
        __tablename__ = tablename

    return SelectTICIDs
