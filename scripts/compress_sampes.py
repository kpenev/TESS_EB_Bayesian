#!/usr/bin/env python3

"""Remove unused sample entries in emcee backend files."""

import subprocess
import os
import shutil
from multiprocessing import Pool

import h5py
from configargparse import ArgumentParser, DefaultsFormatter

from general_purpose_python_modules.multiprocessing_util import (
    setup_process_map,
)

from hacked_emcee_hdf5_backend import HDFBackend


def truncate(fname):
    """Truncate unused parts of datasets and repack the file."""

    assert fname.endswith(
        ".h5"
    ), "Input file must be an HDF5 file with .h5 extension."

    trunc_fname = fname[:-3] + "_truncated.h5"
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
    return trunc_fname, chains_found


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


def process(fname):
    """Process the file by truncating, repacking, and verifying."""

    trunc_fname, chains_expected = truncate(fname)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line()
    with Pool(
        args.num_parallel,
        initializer=setup_process_map,
        initargs=[{"task": "pack_samples"}],
        maxtasksperchild=1,
    ) as pool:
        pool.map(process, args.filenames)
