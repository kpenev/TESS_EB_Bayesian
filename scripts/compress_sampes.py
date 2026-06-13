#!/usr/bin/env python3

"""Remove unused sample entries in emcee backend files."""

import subprocess
import os
import shutil
from multiprocessing import Pool
from functools import partial

import h5py
from configargparse import ArgumentParser, DefaultsFormatter

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)

from hacked_emcee_hdf5_backend import HDFBackend


def truncate(fname, trunc_fname):
    """Truncate unused parts of datasets and repack the file."""

    assert fname.endswith(
        ".h5"
    ), "Input file must be an HDF5 file with .h5 extension."

    shutil.copy2(fname, trunc_fname)
    chains_found = []
    with h5py.File(trunc_fname, "r+") as f:
        for chain in f:
            chains_found.append(chain)
            iterations = f[chain].attrs["iteration"]
            print(
                f"Processing chain {chain} containing "
                f"{iterations} iterations..."
            )
            for dset in ["blobs", "chain", "log_prob"]:
                print(f"{dset} has shape: {f[chain][dset].shape}")
                f[chain][dset].resize((iterations, *f[chain][dset].shape[1:]))
                print("Truncating")
    return chains_found


def list_chains(fname):
    """List available MCMC chains in the give file."""

    chains_found = []
    with h5py.File(fname, "r") as f:
        for chain in f:
            chains_found.append(chain)
    return chains_found


def repack(fname):
    """Repark the file to reduce file size."""

    repacked_fname = fname[:-3] + "_repacked.h5"
    repack_cmd = [
        "h5repack",
        "-f",
        "GZIP=9",
        fname,
        repacked_fname,
    ]
    print(f"Repacking command: {repack_cmd}")
    print("Repacking ...")
    subprocess.run(repack_cmd, check=True)
    print(f"Overwriting: {repacked_fname!r} -> {fname!r}")
    os.replace(repacked_fname, fname)


def verify(orig_fname, repacked_fname, chains_expected):
    """Verify that the truncated file data is identical to original."""

    print(f"Verifying {orig_fname!r} vs {repacked_fname!r}.")
    for chain in chains_expected:
        for fname in [orig_fname, repacked_fname]:
            try:
                os.remove(fname[:-3] + ".unsaved_steps")  # unsaved steps
            except FileNotFoundError:
                pass

        backends = {
            "original": HDFBackend(orig_fname, name=chain),
            "truncated": HDFBackend(repacked_fname, name=chain),
        }
        assert (
            backends["original"].iteration == backends["truncated"].iteration
        ), (
            f"Iteration count mismatch for chain {chain}: "
            f"{backends['original'].iteration} vs "
            f"{backends['truncated'].iteration}"
        )
        assert (
            backends["original"].get_chain()
            == backends["truncated"].get_chain()
        ).all(), f"Chain data mismatch for chain {chain}"
        assert (
            backends["original"].get_log_prob()
            == backends["truncated"].get_log_prob()
        ).all(), f"Log-probability data mismatch for chain {chain}"
        assert (
            backends["original"].get_blobs()
            == backends["truncated"].get_blobs()
        ).all(), f"Blobs data mismatch for chain {chain}"
        print(f"Verified chain {chain}")


def process(fname, verify_only=False):
    """Process the file by truncating, repacking, and verifying."""

    trunc_fname = fname[:-3] + "_truncated.h5"
    if verify_only:
        chains_expected = list_chains(fname)
    else:
        chains_expected = truncate(fname, trunc_fname)
        repack(trunc_fname)
    verify(fname, trunc_fname, chains_expected)
    os.replace(trunc_fname, fname)
    print(f"Finished processing {fname}.")


def parse_command_line():
    """Parse command-line arguments."""

    parser = ArgumentParser(
        description="Truncate padding sample entries in emcee backend files.",
        formatter_class=DefaultsFormatter,
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Path to the HDF5 file to process (e.g., 'samples.h5').",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=16,
        help="The number of parallel processes to use.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Assume all compression steps were already done just run "
        "verification.",
    )
    return parser.parse_args()


def main(args):
    if args.num_parallel > 1:
        with Pool(
            args.num_parallel,
            initializer=setup_process_map,
            initargs=[{"task": "pack_samples"}],
            maxtasksperchild=1,
        ) as pool:
            pool.map(
                partial(process, verify_only=args.verify_only), args.filenames
            )
    else:
        for fname in args.filenames:
            process(fname, verify_only=args.verify_only)

if __name__ == '__main__':
    main(parse_command_line())
