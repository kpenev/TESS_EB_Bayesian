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
    from glob import glob
    from bui.select_ticids.data_model import (
        SelectTICIDBase,
        get_ticid_select_table,
    )
    from paths import prsa_ebs
    from astropy.io import fits
    import re

    samples_rex = re.compile("tess(?P<tic>[0-9]*)_samples.h5")
    SelectTICIDs = get_ticid_select_table("attempted_sampling")
    SelectTICIDBase.metadata.create_all(db_engine)
    with Session.begin() as db_session:
        for samples_fname in glob("/scratch/juno/jas180011/TESS_EBs/restarted_samples/*.h5"):
            parsed = samples_rex.match(path.basename(samples_fname))
            assert parsed
            db_session.add(
                SelectTICIDs(id=int(parsed["tic"]), flag=0, rendered=0)
            )
