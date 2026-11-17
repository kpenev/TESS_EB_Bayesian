#!/usr/bin/env python3

"""
Check that the data files the pipeline needs are installed and load.

Each required file is loaded the way the pipeline loads it, so a file that
exists but is wrong (truncated, a broken link, another version) fails here
rather than deep inside a run. Every file's SHA-256 is printed, so the output
of two machines can be compared (e.g. with diff) to find files that differ.

Run from anywhere; light curves are looked for in ``scripts/mastDownload``,
where the pipeline (run from ``scripts/``) keeps them. Exits with status 1 if
anything required is missing or fails to load.
"""

from hashlib import sha256
import logging
from os import access, listdir, path, W_OK
import sqlite3
import sys

script_dir = path.dirname(path.abspath(__file__))
sys.path.insert(0, script_dir)

# pylint: disable=wrong-import-position
import paths
from data_fnames import data_fnames

# pylint: enable=wrong-import-position

_failures = []


def file_summary(fname):
    """Return the resolved path, size and SHA-256 of a file, as a string."""

    checksum = sha256()
    with open(fname, "rb") as opened:
        for block in iter(lambda: opened.read(2**24), b""):
            checksum.update(block)
    return (
        f"{path.realpath(fname)}  {path.getsize(fname) / 2**20:.1f} MB  "
        f"sha256 {checksum.hexdigest()}"
    )


def check(description, fname, load=None):
    """Report a required file: present, its checksum, and that it loads."""

    if fname is None or not path.isfile(fname):
        _failures.append(description)
        print(f"MISSING  {description}: {fname}")
        return
    try:
        if load is not None:
            load(fname)
    except Exception as error:  # pylint: disable=broad-exception-caught
        _failures.append(description)
        print(f"FAILED   {description}: {type(error).__name__}: {error}")
        print(f"         {file_summary(fname)}")
        return
    print(f"OK       {description}: {file_summary(fname)}")


def load_isochrones(fname):
    """Load an isochrone grid as `BinaryParams` does."""

    # pylint: disable=import-outside-toplevel
    from general_purpose_python_modules.cmd_utils import CMDInterpolator

    CMDInterpolator(fname, binary_dir=paths.isochrone_binary_dir)


def load_gravity_darkening(_):
    """Build the gravity darkening interpolator the binary model uses."""

    # pylint: disable=import-outside-toplevel
    from gravity_darkening import GravDarkInterpolator

    GravDarkInterpolator()


def load_gaia_extinction(fname):
    """Build the Gaia extinction coefficients as the SED does."""

    # pylint: disable=import-outside-toplevel
    from extinction_correction import GaiaExtinctionCoefficients

    GaiaExtinctionCoefficients(model_fname=fname)


def bayestar_fname():
    """Return where dustmaps looks for the Bayestar19 map."""

    # pylint: disable=import-outside-toplevel
    from dustmaps.config import config

    print(f"         dustmaps data_dir: {config['data_dir']}")
    if config["data_dir"] is None:
        return None
    return path.join(config["data_dir"], "bayestar", "bayestar2019.h5")


def load_bayestar(_):
    """Query the map once, as the SED does (loading takes a minute or two)."""

    # pylint: disable=import-outside-toplevel
    from astropy.coordinates import SkyCoord
    from astropy import units
    from dustmaps.bayestar import BayestarQuery

    query = BayestarQuery(version="bayestar2019")
    reddening = query(
        SkyCoord(
            l=90.0 * units.deg,
            b=10.0 * units.deg,
            distance=1.0 * units.kpc,
            frame="galactic",
        ),
        mode="median",
    )
    # Negated so that NaN fails too.
    if not reddening > 0:  # pylint: disable=unnecessary-negation
        raise ValueError(
            f"Bayestar19 gave E = {reddening!r} at l=90, b=10, d=1 kpc"
        )


def check_phoebe():
    """Check that PHOEBE has the passband the binary model uses."""

    # pylint: disable=import-outside-toplevel
    import phoebe

    passband = "Bolometric:900-40000"
    if passband in phoebe.list_installed_passbands():
        print(f"OK       PHOEBE {phoebe.__version__} passband {passband}")
    else:
        _failures.append("PHOEBE passband")
        print(f"MISSING  PHOEBE {phoebe.__version__} passband {passband}")


def load_cache(fname):
    """
    Open the cache read-only, list its tables, and require data exclusions.

    BLS and SED rows are computed per target when missing, so they may be
    empty. The data exclusions are set by hand in the BUI and nothing
    recreates them: without them, excluded data are silently fit.
    """

    with sqlite3.connect(f"file:{fname}?mode=ro", uri=True) as cache:
        counts = {
            table: cache.execute(f"select count(*) from {table}").fetchone()[0]
            for table in ("bls", "sed", "exclude_data")
        }
    print(f"         cache rows: {counts}")
    if counts["exclude_data"] == 0:
        raise ValueError(
            "no data exclusions; copy the cache from the BUI machine"
        )


def check_lightcurves():
    """Check the light curve directory the pipeline downloads into."""

    download_dir = path.join(script_dir, "mastDownload")
    markers = path.join(download_dir, "complete")
    if not path.isdir(download_dir):
        _failures.append("light curve directory")
        print(f"MISSING  light curve directory: {download_dir}")
        return
    writable = access(download_dir, W_OK)
    num_markers = len(listdir(markers)) if path.isdir(markers) else 0
    print(
        f"{'OK' if writable else 'READONLY':8s} light curve directory: "
        f"{path.realpath(download_dir)}, {num_markers} completeness markers"
    )
    if not writable:
        _failures.append("light curve directory not writable")


def main():
    """Check everything and report."""

    logging.disable(logging.CRITICAL)

    print("Required by the pipeline:")
    for photsys in ("gaiaEDR3", "2mass_spitzer_wise", "tess_2mass"):
        check(
            f"isochrones {photsys}",
            paths.cmd_data_fname.format(photsys=photsys),
            load_isochrones,
        )
    check("gravity darkening", paths.grav_dark["TESS"], load_gravity_darkening)
    check(
        "Gaia extinction model",
        paths.gaia_extinction_model,
        load_gaia_extinction,
    )
    check("Bayestar19 map", bayestar_fname(), load_bayestar)
    check_phoebe()
    check("likelihood cache", paths.cache_db, load_cache)
    check_lightcurves()

    print("\nUsed only by side scripts (not required):")
    optional = dict(data_fnames)
    for mode in ("globalmin", "simplefit"):
        optional[f"jktebob {mode}"] = paths.jktebob["template"].format(
            mode=mode
        )
    for name, fname in optional.items():
        if path.isfile(fname):
            print(f"present  {name}: {file_summary(fname)}")
        else:
            print(f"absent   {name}: {fname}")

    if _failures:
        print(f"\n{len(_failures)} problem(s): {', '.join(_failures)}")
        sys.exit(1)
    print("\nAll required data present and loading.")


if __name__ == "__main__":
    main()
