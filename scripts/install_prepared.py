#!/usr/bin/env python3

"""
Install the targets the preparation machine exported, for review.

For each ``EXCHANGE_DIR/from_prep/<TIC>/`` written by export_prepared.py
whose files have all arrived intact (checked against its manifest):

* its review images replace the target's earlier ones in the BUI's render
  directories (``prepare_sed/tess<TIC>.png``, and
  ``prepare_families/tess<TIC>_<k>.png`` in the family order of
  ``init_models.txt``),
* its cache rows are merged into the likelihood cache: the BLS row replaces
  this machine's, since the walkers' coordinates are defined relative to it;
  the SED values replace this machine's, keeping the bad SED threshold,
  penalty and sharpness set in the BUI,
* its samples file is stored in SAMPLES_DIR, for sampling once approved,
* a failed preparation sets the target's status to prepare_failed (7); a
  successful one sets a target that had failed back to prepare (2),
* its exchange directory is removed (the preparation machine keeps the
  originals in its OUTPUT_DIR).

Targets whose files are still arriving are left for the next run. Run from
anywhere; the BUI database is ``scripts/TESS_EBs.db``.
"""

from argparse import ArgumentParser
from glob import glob
from hashlib import sha256
import json
import logging
from os import chdir, makedirs, path, remove
import shutil
import sqlite3

import paths

_logger = logging.getLogger(__name__)

status_prepare = 2
status_prepare_failed = 7
sed_settings = ("bad_sed_threshold", "bad_sed_penalty", "bad_sed_sharpness")


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "exchange_dir", help="The root of the folder shared with the preparer."
    )
    parser.add_argument(
        "--review-table",
        default="sample_prsa",
        help="The BUI table of the reviewed TICs (default: %(default)s).",
    )
    parser.add_argument(
        "--samples-dir",
        default=path.join(paths.results_dir, "samples_seeded"),
        help="Where to store the samples files (default: %(default)s).",
    )
    return parser.parse_args()


def file_checksum(fname):
    """Return the SHA-256 of a file as a hex string."""

    checksum = sha256()
    with open(fname, "rb") as opened:
        for block in iter(lambda: opened.read(2**24), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_complete_manifest(target_dir):
    """Return the target's manifest once all its files arrived intact."""

    manifest_fname = path.join(target_dir, "manifest.json")
    if not path.exists(manifest_fname):
        return None
    with open(manifest_fname, encoding="ascii") as manifest_file:
        manifest = json.load(manifest_file)
    for fname, checksum in manifest["files"].items():
        fname = path.join(target_dir, fname)
        if not path.exists(fname) or file_checksum(fname) != checksum:
            _logger.info("%s: still arriving, left for later.", fname)
            return None
    return manifest


def install_images(target_dir, tic_id, render_dirs):
    """Replace the target's review images in the BUI render directories."""

    for render_dir in render_dirs.values():
        makedirs(render_dir, exist_ok=True)
        for old in glob(path.join(render_dir, f"tess{tic_id}.png")) + glob(
            path.join(render_dir, f"tess{tic_id}_*.png")
        ):
            remove(old)

    sed_image = path.join(target_dir, f"sed_grid_corner_{tic_id}.png")
    if path.exists(sed_image):
        shutil.copy(
            sed_image,
            path.join(render_dirs["prepare_sed"], f"tess{tic_id}.png"),
        )
    list_fname = path.join(target_dir, "init_models.txt")
    if not path.exists(list_fname):
        return
    with open(list_fname, encoding="ascii") as list_file:
        images = [
            line.split()[0]
            for line in list_file
            if line.strip() and not line.startswith("#")
        ]
    for family, image in enumerate(images):
        shutil.copy(
            path.join(target_dir, image),
            path.join(
                render_dirs["prepare_families"], f"tess{tic_id}_{family}.png"
            ),
        )


def merge_cache_rows(rows_fname):
    """Merge a target's exported BLS and SED rows into the cache."""

    with sqlite3.connect(paths.cache_db, timeout=100) as cache:
        cache.execute("ATTACH DATABASE ? AS exported", (rows_fname,))
        cache.execute(
            "DELETE FROM main.bls "
            "WHERE tic_id IN (SELECT tic_id FROM exported.bls)"
        )
        bls_columns = ", ".join(
            row[1] for row in cache.execute("PRAGMA main.table_info(bls)")
        )
        cache.execute(
            f"INSERT INTO main.bls ({bls_columns}) "
            f"SELECT {bls_columns} FROM exported.bls"
        )

        columns = [
            row[1] for row in cache.execute("PRAGMA main.table_info(sed)")
        ]
        values = [
            column
            for column in columns
            if column != "tic_id" and column not in sed_settings
        ]
        cache.execute(
            f"UPDATE main.sed SET ({', '.join(values)}) = "
            f"(SELECT {', '.join(values)} FROM exported.sed "
            "WHERE exported.sed.tic_id = main.sed.tic_id) "
            "WHERE tic_id IN (SELECT tic_id FROM exported.sed)"
        )
        cache.execute(
            f"INSERT INTO main.sed ({', '.join(columns)}) "
            f"SELECT {', '.join(columns)} FROM exported.sed "
            "WHERE tic_id NOT IN (SELECT tic_id FROM main.sed)"
        )
        cache.commit()
        cache.execute("DETACH DATABASE exported")


def update_status(select_table, tic_id, outcome):
    """Set prepare_failed after a failure, or clear it after a success."""

    # pylint: disable=import-outside-toplevel
    from bui.db_interface import Session

    with Session.begin() as db_session:  # pylint: disable=no-member
        entry = db_session.get(select_table, tic_id)
        if entry is None:
            _logger.warning("TIC %d is not in the review table.", tic_id)
            return
        if outcome == "failed":
            if entry.status in (status_prepare, status_prepare_failed):
                entry.status = status_prepare_failed
            else:
                _logger.warning(
                    "TIC %d failed preparation but has status %d; left as is.",
                    tic_id,
                    entry.status,
                )
        elif entry.status == status_prepare_failed:
            entry.status = status_prepare


def main(config):
    """Install every complete target in the exchange folder."""

    # The BUI database is opened relative to the working directory.
    chdir(path.dirname(path.abspath(__file__)))
    # pylint: disable=import-outside-toplevel
    from bui.select_ticids.data_model import get_ticid_select_tables
    from bui.select_ticids.path_util import get_render_dir

    # pylint: enable=import-outside-toplevel

    render_dirs = {
        plot_type: get_render_dir(config.review_table, plot_type)
        for plot_type in ("prepare_families", "prepare_sed")
    }
    select_table, _ = get_ticid_select_tables(
        config.review_table, must_exist=True
    )
    makedirs(config.samples_dir, exist_ok=True)

    installed = 0
    for target_dir in sorted(
        glob(path.join(config.exchange_dir, "from_prep", "*"))
    ):
        manifest = read_complete_manifest(target_dir)
        if manifest is None:
            continue
        tic_id = manifest["tic_id"]
        install_images(target_dir, tic_id, render_dirs)
        merge_cache_rows(path.join(target_dir, "cache_rows.sqlite"))
        samples = path.join(target_dir, manifest["samples_fname"])
        if path.exists(samples):
            shutil.copy2(samples, config.samples_dir)
        update_status(select_table, tic_id, manifest["outcome"])
        shutil.rmtree(target_dir)
        installed += 1
        _logger.info(
            "Installed TIC %d (%s, commit %s).",
            tic_id,
            manifest["outcome"],
            manifest["commit"],
        )

    if installed:
        # Register the images the way the BUI's rendered-plots refresh does.
        get_ticid_select_tables(
            config.review_table,
            plot_dirs=tuple(render_dirs.values()),
            must_exist=True,
            refresh_rendered=True,
        )
    _logger.info("Installed %d target(s).", installed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
