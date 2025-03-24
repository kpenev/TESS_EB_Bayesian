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
    from os import path
    from bui.select_ticids.data_model import (
        SelectTICIDBase,
        get_ticid_select_table,
    )
    from paths import prsa_ebs
    from astropy.io import fits

    SelectTICIDs = get_ticid_select_table("prsa_ebs")
    SelectTICIDBase.metadata.create_all(db_engine)
    with Session.begin() as db_session:
        with fits.open(prsa_ebs, 'readonly') as prsa:
            data = prsa[1].data
            for ticid in data['TIC'][data['m_TIC'] == 1]:
                db_session.add(SelectTICIDs(id=int(ticid), flag=0, rendered=0))
