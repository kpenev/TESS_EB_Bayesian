#!/usr/bin/env python3

"""
Send a prepared target to the exchange folder, for review and sampling.

Copies to ``EXCHANGE_DIR/from_prep/<TIC>/`` what the machine running the
review and the sampling needs from a run of prepare_sampling.sh:

* the review images: the SED search corner plot, and the plot of each
  family listed in ``init_models.txt`` (best family first),
* the samples file holding the starting walkers,
* the target's BLS and SED rows of the likelihood cache (``cache_rows.sqlite``):
  the walkers' coordinates are defined relative to them,
* the failure record, if the preparation failed.

Whatever of these exists is sent, so a failed preparation sends its record and
any plots it made. An earlier export of the same target is replaced. The
manifest (``manifest.json``: each file's SHA-256, the commit, the outcome) is
written last; the folder is synchronized file by file, so the receiving side
(install_prepared.py) installs a target only once every file in its manifest
has arrived intact.
"""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
import logging
from os import listdir, path, makedirs, replace
import shutil
import sqlite3
import subprocess

from paths import cache_db

_logger = logging.getLogger(__name__)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC prepared.")
    parser.add_argument(
        "output_dir", help="The OUTPUT_DIR given to prepare_sampling.sh."
    )
    parser.add_argument(
        "samples_fname", help="The SAMPLES_FNAME given to prepare_sampling.sh."
    )
    parser.add_argument(
        "exchange_dir", help="The root of the folder shared with the reviewer."
    )
    return parser.parse_args()


def file_checksum(fname):
    """Return the SHA-256 of a file as a hex string."""

    checksum = sha256()
    with open(fname, "rb") as opened:
        for block in iter(lambda: opened.read(2**24), b""):
            checksum.update(block)
    return checksum.hexdigest()


def family_images(output_dir):
    """Return the family plots listed in init_models.txt, best first."""

    list_fname = path.join(output_dir, "init_models.txt")
    if not path.exists(list_fname):
        return []
    with open(list_fname, encoding="ascii") as list_file:
        return [
            line.split()[0]
            for line in list_file
            if line.strip() and not line.startswith("#")
        ]


def export_cache_rows(tic_id, destination):
    """Copy the target's rows of the cache tables to a new SQLite file."""

    with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as source:
        with sqlite3.connect(destination) as exported:
            for table in ("bls", "sed"):
                (create_sql,) = source.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
                exported.execute(create_sql)
                rows = source.execute(
                    f"SELECT * FROM {table} WHERE tic_id = ?", (tic_id,)
                ).fetchall()
                if rows:
                    exported.executemany(
                        f"INSERT INTO {table} VALUES "
                        f"({', '.join('?' * len(rows[0]))})",
                        rows,
                    )
                _logger.info("Exported %d %s row(s).", len(rows), table)


def main(config):
    """Export the target."""

    target_dir = path.join(config.exchange_dir, "from_prep", str(config.tic_id))
    if path.exists(target_dir):
        shutil.rmtree(target_dir)
    makedirs(target_dir)

    sources = [
        path.join(config.output_dir, f"sed_grid_corner_{config.tic_id}.png"),
        path.join(config.output_dir, "init_models.txt"),
        path.join(config.output_dir, "prepare_failed.txt"),
        config.samples_fname,
    ] + [
        path.join(config.output_dir, image)
        for image in family_images(config.output_dir)
    ]
    for source in sources:
        if path.exists(source):
            shutil.copy2(source, target_dir)
    export_cache_rows(config.tic_id, path.join(target_dir, "cache_rows.sqlite"))

    failed = path.exists(path.join(target_dir, "prepare_failed.txt"))
    manifest = {
        "tic_id": config.tic_id,
        "outcome": "failed" if failed else "prepared",
        "samples_fname": path.basename(config.samples_fname),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path.dirname(path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "exported": datetime.now().astimezone().isoformat(),
        "files": {
            fname: file_checksum(path.join(target_dir, fname))
            for fname in sorted(listdir(target_dir))
        },
    }
    # Syncthing does not send files named like its own temporary files, so
    # the manifest appears on the other side only once complete.
    temporary = path.join(target_dir, ".syncthing.manifest.json.tmp")
    with open(temporary, "w", encoding="ascii") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    replace(temporary, path.join(target_dir, "manifest.json"))
    _logger.info(
        "Exported TIC %d (%s, %d files) to %s",
        config.tic_id,
        manifest["outcome"],
        len(manifest["files"]),
        target_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
