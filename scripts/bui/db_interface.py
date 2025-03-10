"""Connect to the database and provide a session scope for queries."""

from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

db_engine = create_engine(
    "sqlite:///TESS_EBs.db?timeout=100&uri=true",
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=NullPool,
)

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
Session = sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name

if __name__ == "__main__":
    from bui.select_ticids.data_model import (
        SelectTICIDBase,
        get_ticid_select_table,
    )

    SelectTICIDs = create_ticid_select_table("test_batch")
    SelectTICIDBase.metadata.create_all(db_engine)
    with Session.begin() as db_session:
        for i in range(100):
            if i % 7 == 0:
                flag = -1
            elif i % 7 < 3:
                flag = 1
            else:
                flag = 0
            db_session.add(SelectTICIDs(id=i, flag=flag))
