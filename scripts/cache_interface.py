"""Define the cache database."""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import Float, Integer, inspect, select, delete

# from sqlalchemy.ext.hybrid import hybrid_property

from paths import cache_db as db_fname

db_engine = create_engine(
    f"sqlite:///{db_fname}?timeout=100&uri=true",
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=NullPool,
)

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
CacheSession = sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name


# pylint: disable=too-few-public-methods
class DataModelBase(DeclarativeBase):
    """Base class that handles 64 bit unsigned integer Gaia IDs."""

    tic_id = mapped_column(
        Integer,
        primary_key=True,
        doc="The TIC identifier of the EB.",
    )


#    @hybrid_property
#    def gaia_id(self):
#        """Convert Gaia IDs back to unsigned integer for user."""
#
#        return int(self._gaia_id)
#
#    @gaia_id.expression
#    def gaia_id(cls):
#        return cls._gaia_id
#
#    @gaia_id.setter
#    def gaia_id(self, gaia_id):
#        """Convert Gaia IDs to signed integer for storing in DB."""
#
#        self._gaia_id = repr(gaia_id)


class CachedSED(DataModelBase):
    """Cache absolute magnitudes in PannSTARRS and WISE passbands."""

    __tablename__ = "sed"

    gp1 = mapped_column(
        Float, doc="The PanSTARRS1 g filter magnitude of the EB"
    )
    rp1 = mapped_column(
        Float, doc="The PanSTARRS1 r filter magnitude of the EB"
    )
    ip1 = mapped_column(
        Float, doc="The PanSTARRS1 i filter magnitude of the EB"
    )
    zp1 = mapped_column(
        Float, doc="The PanSTARRS1 z filter magnitude of the EB"
    )
    yp1 = mapped_column(
        Float, doc="The PanSTARRS1 y filter magnitude of the EB"
    )
    j2m = mapped_column(Float, doc="The 2MASS J filter magnitude of the EB")
    h2m = mapped_column(Float, doc="The 2MASS H filter magnitude of the EB")
    k2m = mapped_column(Float, doc="The 2MASS Kz filter magnitude of the EB")
    w1 = mapped_column(Float, doc="The WISE W1 filter magnitude of the EB")
    w2 = mapped_column(Float, doc="The WISE W2 filter magnitude of the EB")

    gp1_err = mapped_column(
        Float, doc="The PanSTARRS1 g filter magnitude uncertainty of the EB"
    )
    rp1_err = mapped_column(
        Float, doc="The PanSTARRS1 r filter magnitude uncertainty of the EB"
    )
    ip1_err = mapped_column(
        Float, doc="The PanSTARRS1 i filter magnitude uncertainty of the EB"
    )
    zp1_err = mapped_column(
        Float, doc="The PanSTARRS1 z filter magnitude uncertainty of the EB"
    )
    yp1_err = mapped_column(
        Float, doc="The PanSTARRS1 y filter magnitude uncertainty of the EB"
    )
    j2m_err = mapped_column(
        Float, doc="The 2MASS J filter magnitude uncertainty of the EB"
    )
    h2m_err = mapped_column(
        Float, doc="The 2MASS H filter magnitude uncertainty of the EB"
    )
    k2m_err = mapped_column(
        Float, doc="The 2MASS Kz filter magnitude uncertainty of the EB"
    )
    w1_err = mapped_column(
        Float, doc="The WISE W1 filter magnitude uncertainty of the EB"
    )
    w2_err = mapped_column(
        Float, doc="The WISE W2 filter magnitude uncertainty of the EB"
    )

    def __str__(self):
        return (
            f"{self.tic_id}: "
            f"{self.gp1}, {self.rp1}, {self.ip1}, {self.zp1}, {self.yp1}"
        )


class CachedBLS(DataModelBase):
    """Cache best fit BLS properties."""

    __tablename__ = "bls"

    period = mapped_column(Float, doc="The best fit BLS orbital period.")
    period_uncertainty = mapped_column(
        Float, doc="The best fit BLS orbital period."
    )
    transit_time = mapped_column(
        Float, doc="The best fit BLS time of first transit."
    )
    duration = mapped_column(Float, doc="The best fit BLS transit duration.")
    depth = mapped_column(Float, doc="The best fit BLS transit depth.")
    depth_odd = mapped_column(Float, doc="The best fit BLS odd transit depth.")
    depth_even = mapped_column(
        Float, doc="The best fit BLS even transit depth."
    )
    depth_half = mapped_column(
        Float, doc="The best fit BLS half period transit depth."
    )
    depth_phased = mapped_column(
        Float, doc="The best fit BLS phase shifted transit depth."
    )

    depth_uncertainty = mapped_column(
        Float, doc="The best fit BLS transit depth."
    )
    depth_odd_uncertainty = mapped_column(
        Float, doc="The best fit BLS odd transit depth."
    )
    depth_even_uncertainty = mapped_column(
        Float, doc="The best fit BLS even transit depth."
    )
    depth_half_uncertainty = mapped_column(
        Float, doc="The best fit BLS half period transit depth."
    )
    depth_phased_uncertainty = mapped_column(
        Float, doc="The best fit BLS phase shifted transit depth."
    )

    harmonic_amplitude = mapped_column(
        Float, doc="The amplitude of the best fit sinusoidal model."
    )

    harmonic_delta_log_likelihood = mapped_column(
        Float,
        doc="The difference in log likelihood between a sinusoidal model"
        " and the transit model. If harmonic_delta_log_likelihood is greater "
        "than zero, the sinusoidal model is preferred.",
    )
    count_odd = mapped_column(
        Integer, doc="The total number of points in all odd transits."
    )
    count_even = mapped_column(
        Integer, doc="The total number of points in all even transits."
    )

    masked_period = mapped_column(Float, doc="The best fit BLS orbital period.")
    masked_period_uncertainty = mapped_column(
        Float, doc="The best fit BLS orbital period."
    )
    masked_transit_time = mapped_column(
        Float, doc="The best fit BLS time of first transit."
    )
    masked_duration = mapped_column(
        Float, doc="The best fit BLS transit duration."
    )
    masked_depth = mapped_column(Float, doc="The best fit BLS transit depth.")
    masked_depth_odd = mapped_column(
        Float, doc="The best fit BLS odd transit depth."
    )
    masked_depth_even = mapped_column(
        Float, doc="The best fit BLS even transit depth."
    )
    masked_depth_half = mapped_column(
        Float, doc="The best fit BLS half period transit depth."
    )
    masked_depth_phased = mapped_column(
        Float, doc="The best fit BLS phase shifted transit depth."
    )

    masked_depth_uncertainty = mapped_column(
        Float, doc="The best fit BLS transit depth."
    )
    masked_depth_odd_uncertainty = mapped_column(
        Float, doc="The best fit BLS odd transit depth."
    )
    masked_depth_even_uncertainty = mapped_column(
        Float, doc="The best fit BLS even transit depth."
    )
    masked_depth_half_uncertainty = mapped_column(
        Float, doc="The best fit BLS half period transit depth."
    )
    masked_depth_phased_uncertainty = mapped_column(
        Float, doc="The best fit BLS phase shifted transit depth."
    )

    masked_harmonic_amplitude = mapped_column(
        Float, doc="The amplitude of the best fit sinusoidal model."
    )

    masked_harmonic_delta_log_likelihood = mapped_column(
        Float,
        doc="The difference in log likelihood between a sinusoidal model"
        " and the transit model. If harmonic_delta_log_likelihood is greater "
        "than zero, the sinusoidal model is preferred.",
    )

    masked_count_odd = mapped_column(
        Integer,
        doc="The total number of points in all odd transits after masking.",
    )
    masked_count_even = mapped_column(
        Integer,
        doc="The total number of points in all even transits after masking.",
    )


for table in DataModelBase.metadata.sorted_tables:
    if not inspect(db_engine).has_table(table.name):
        table.create(db_engine)


if __name__ == "__main__":

    with CacheSession.begin() as cache:
        cache.add(
            CachedSED(
                tic_id=11119600,
                gp1=1.0,
                rp1=2.0,
                ip1=3.0,
                zp1=4.0,
                yp1=5.0,
            )
        )

    with CacheSession.begin() as cache:
        cache.execute(delete(CachedSED).filter_by(tic_id=56))

    with CacheSession.begin() as cache:
        print(
            cache.execute(select(CachedSED).filter_by(tic_id=11119600)).scalar()
        )
# pylint: enable=too-few-public-methods
