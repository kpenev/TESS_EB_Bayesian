#!/usr/bin/env python3
"""Collect maximum-likelihood blob values for finished-sampling TICs.

For every TIC whose status in its ``SelectTICTable`` equals the configured
"finished" value, the maximum-likelihood set of blob (``SampleParams``) values
is extracted from the ``mcmc`` chain and stored as one row in a FITS table.

Re-running is incremental: TICs already present in the output file are kept
untouched and only newly finished TICs are read and appended, so the chain
reads are never repeated.
"""

import os

import numpy
from astropy.table import QTable, vstack
from sqlalchemy import select

import paths
from command_line_util import create_parser
from sample_params import SampleParams, param_units, param_descriptions
from chain_analysis import get_max_likelihood_params

# False positive
# pylint: disable=import-error
from bui.db_interface import Session
from bui.select_ticids.data_model import JobGroup, get_ticid_select_tables

# pylint: enable=import-error


def parse_command_line():
    """Return the command line configuration."""

    parser = create_parser()
    parser.add_argument(
        "--finished-status",
        type=int,
        default=5,
        help="The status value in the SelectTICTable that marks a TIC as "
        "finished sampling.",
    )
    parser.add_argument(
        "--select-tic-table",
        nargs="+",
        default=None,
        help="Name(s) of the SelectTICTable(s) to scan for finished TICs. If "
        "not given, every distinct table referenced by a job group is scanned.",
    )
    parser.add_argument(
        "--output-fname",
        default=os.path.join(
            paths.results_dir, "finished_max_likelihood.fits"
        ),
        help="The FITS file to create/update with one row per finished TIC.",
    )
    parser.add_argument(
        "--samples-fname-pattern",
        default=paths.samples,
        help="The filename pattern where samples were saved.",
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Ignore any existing output file and recompute every finished TIC "
        "from scratch instead of only adding newly finished ones.",
    )
    return parser.parse_args()


def get_finished_tics(config):
    """Return the sorted list of TIC IDs marked finished across the tables."""

    finished = set()
    with Session.begin() as db_session:  # pylint: disable=no-member
        if config.select_tic_table:
            tablenames = config.select_tic_table
        else:
            tablenames = db_session.scalars(
                select(JobGroup.select_tic_table).distinct()
            ).all()
        for tablename in tablenames:
            select_table = get_ticid_select_tables(
                tablename, must_exist=True
            )[0]
            finished.update(
                db_session.scalars(
                    select(select_table.id).where(
                        select_table.status  # pylint: disable=no-member
                        == config.finished_status
                    )
                ).all()
            )
    return sorted(finished)


def compute_record(config, tic_id):
    """Return a ``(tic_id, SampleParams, log_prob)`` record, or None on failure.

    Any error while reading or processing a single TIC is caught and logged so
    that one bad system cannot abort the whole run.
    """

    samples_fname = config.samples_fname_pattern.format(tic_id=tic_id)
    if not os.path.exists(samples_fname):
        print(
            f"WARNING: no samples file for TIC {tic_id} "
            f"({samples_fname!r}); skipping."
        )
        return None
    try:
        best_params, best_log_prob = get_max_likelihood_params(samples_fname)
    except Exception as err:  # pylint: disable=broad-except
        print(f"WARNING: could not process TIC {tic_id}: {err}; skipping.")
        return None
    print(f"TIC {tic_id}: max log-prob = {best_log_prob:.6g}")
    return tic_id, best_params, best_log_prob


def build_new_table(records):
    """Build a QTable from ``(tic_id, SampleParams, log_prob)`` records."""

    table = QTable()
    table["tic_id"] = numpy.array(
        [rec[0] for rec in records], dtype=numpy.int64
    )
    table["tic_id"].description = "TESS Input Catalog identifier."
    for index, name in enumerate(SampleParams._fields):
        table[name] = (
            numpy.array([rec[1][index] for rec in records], dtype=float)
            * param_units[name]
        )
        table[name].description = param_descriptions.get(name, "")
    table["log_prob"] = numpy.array([rec[2] for rec in records], dtype=float)
    table["log_prob"].description = (
        "Maximum log-probability (log posterior) attained in the mcmc chain."
    )
    return table


def main(config):
    """Update the FITS table with newly finished TICs, saving after each one."""

    finished = get_finished_tics(config)
    print(
        f"{len(finished)} TIC(s) marked finished "
        f"(status == {config.finished_status})."
    )

    combined = None
    known = set()
    if os.path.exists(config.output_fname) and not config.recompute_all:
        combined = QTable.read(config.output_fname, format="fits")
        known = {int(tic) for tic in combined["tic_id"]}
        print(
            f"{len(known)} TIC(s) already present in {config.output_fname!r}; "
            "they will not be recomputed."
        )

    todo = [tic_id for tic_id in finished if tic_id not in known]
    if not todo:
        print("No new finished TICs to add; output is up to date.")
        return

    os.makedirs(
        os.path.dirname(os.path.abspath(config.output_fname)), exist_ok=True
    )
    print(f"Reading chains for {len(todo)} new TIC(s)...")
    num_added = 0
    for tic_id in todo:
        record = compute_record(config, tic_id)
        if record is None:
            continue
        row = build_new_table([record])
        combined = row if combined is None else vstack([combined, row])
        # Persist immediately so a later crash cannot lose finished results.
        combined.write(config.output_fname, format="fits", overwrite=True)
        num_added += 1

    total = 0 if combined is None else len(combined)
    print(
        f"Done: added {num_added} new TIC(s); {total} total in "
        f"{config.output_fname!r}."
    )


if __name__ == "__main__":
    main(parse_command_line())
