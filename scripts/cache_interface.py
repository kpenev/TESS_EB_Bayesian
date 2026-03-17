"""Define the cache database."""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import Float, Integer, inspect

# from sqlalchemy.ext.hybrid import hybrid_property

import paths

db_engine = create_engine(
    f"sqlite:///{paths.cache_db}?timeout=100&uri=true",
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=NullPool,
)

CacheSession = sessionmaker(
    db_engine, expire_on_commit=False
)  # pylint: disable=invalid-name


class DataModelBase(DeclarativeBase):  # pylint: disable=too-few-public-methods
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


class CachedSED(DataModelBase):  # pylint: disable=too-few-public-methods
    """Cache absolute magnitudes in PannSTARRS and WISE passbands."""

    __tablename__ = "sed"

    default_bad_sed_threshold = 10.0
    default_bad_sed_penalty = 1.0

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
    bad_sed_threshold = mapped_column(
        Float,
        doc="How many sigma away should model be from measured SED before the "
        "bad SED penalty kicks in.",
    )
    bad_sed_penalty = mapped_column(
        Float,
        doc="The factor by which to enhance the SED error bar when the "
        "threshold is exceeded."
    )

    def __str__(self):
        return (
            f"{self.tic_id}: "
            f"{self.gp1}, {self.rp1}, {self.ip1}, {self.zp1}, {self.yp1}"
        )


class CachedBLS(DataModelBase):  # pylint: disable=too-few-public-methods
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


class ExcludeDataTable(DataModelBase):  # pylint: disable=too-few-public-methods
    """For each TIC ID specify sectors and/or (OOE=0, BLSOOE=-1) to exclude."""

    __tablename__ = "exclude_data"

    exclude = mapped_column(
        Integer,
        primary_key=True,
        doc="Sector or part of data to exclude. Positive integers specify "
        "sector numbers, 0 removes OOE variability only when running BLS, "
        "-1 excludes OOE variability from LC model, -2 excludes all QLP "
        "sectors, and -3 excludes all SPOC sectrs."
        "",
    )


for table in DataModelBase.metadata.sorted_tables:
    if not inspect(db_engine).has_table(table.name):
        table.create(db_engine)
