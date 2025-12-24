"""Utilities for querrying the PanSTARRS1 catalog."""

import logging

import requests
import numpy
import astropy.io
from astropy.table import Table
from astroquery.gaia import Gaia

_logger = logging.getLogger(__name__)


def ps1search(
    table="mean",
    release="dr1",
    *,
    fmt="json",
    columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
    **kw,
):
    """
    Do a general search of the PS1 catalog (possibly without ra/dec/radius)

    Args:
        table (string):    mean, stack, or detection

        release (string):    dr1 or dr2

        fmt:    csv, votable, json

        columns:    list of column names to include (None means use defaults)

        baseurl:    base URL for the request

        **kw:    other parameters (e.g., 'nDetections.min':2).  Note this is
            required!
    """

    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table, release)
    if fmt not in ("csv", "votable", "json"):
        raise ValueError("Bad value for fmt")
    url = f"{baseurl}/{release}/{table}.{fmt}"
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table, release)["name"]:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError(
                f"Some columns not found in table: {', '.join(badcols)}"
            )
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data["columns"] = f"[{','.join(columns)}]"
        _logger.debug("Querying with columns: %s", repr(data["columns"]))

    # either get or post works
    #    r = requests.post(url, data=data)
    r = requests.get(url, params=data, timeout=300)

    _logger.debug("Query URL: %s", r.url)
    r.raise_for_status()
    if fmt == "json":
        return r.json()
    return r.text


def checklegal(table, release):
    """
    Checks if this combination of table and release is acceptable

    Raises a VelueError exception if there is problem
    """

    releaselist = ("dr1", "dr2")
    if release not in ("dr1", "dr2"):
        raise ValueError(
            f"Bad value for release (must be one of {', '.join(releaselist)})"
        )
    if release == "dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError(
            f"Bad value for table (for {release} must be one of "
            f"{', '.join(tablelist)})"
        )


def ps1metadata(
    table="mean",
    release="dr1",
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
):
    """
    Return metadata for the specified catalog and table

    Args:
        table (string):    mean, stack, or detection

        release (string):    dr1 or dr2

        baseurl:    base URL for the request

    Returns:
        an astropy table with columns name, type, description
    """

    checklegal(table, release)
    url = f"{baseurl}/{release}/{table}/metadata"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(
        rows=[
            (x["name"], x.get("type", float), x.get("description", ""))
            for x in v
        ],
        names=("name", "type", "description"),
    )
    return tab


def get_panstarrs_mags(gaia_id, phot_mode="MeanPSFMag", filters="grizy"):
    """
    Return the Pan-STARRS magnitudes of the given Gaia ID.

    See
    https://outerspace.stsci.edu/display/PANSTARRS/PS1+FAQ+-+Frequently+asked+questions
    for choice of photometry mode."""

    ps1_id = Gaia.launch_job(
        "SELECT original_ext_source_id FROM "
        "gaiadr3.panstarrs1_best_neighbour WHERE source_id = " + str(gaia_id)
    ).get_results()["original_ext_source_id"]
    if ps1_id.size == 0:
        return (
            [numpy.nan] * len(filters),
            [numpy.nan] * len(filters),
        )
    _logger.debug(
        "Gaia ID %s corresponds to PS1 ID %s.", repr(gaia_id), repr(ps1_id)
    )
    ps1_id = int(ps1_id)

    ps1_result = numpy.array(
        ps1search(
            release="dr1",
            verbose=False,
            objid=ps1_id,
            columns=[f"{fil}{phot_mode}" for fil in filters]
            + [f"{fil}{phot_mode}Err" for fil in filters],
        )["data"][0]
    )
    ps1_result[ps1_result < -900.0] = numpy.nan
    return (
        numpy.array(ps1_result[: len(filters)]),
        (numpy.array(ps1_result[len(filters) :]) ** 2 + 0.015**2) ** 0.5,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(get_panstarrs_mags(1347743939869168640))
