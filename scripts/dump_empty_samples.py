#!/usr/bin/env python3

"""Remove unused sample entries in emcee backend files."""

import subprocess
import os
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
    chains_found = []
    with h5py.File(fname, "r+") as f:
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


def repack(fname):
    """Repark the file to reduce file size."""

    repack_cmd = [
        "h5repack",
        "-f",
        "GZIP=9",
        fname,
        fname[:-3] + "_truncated.h5",
    ]
    print(f"Repacking command: {repack_cmd}")
    print("Repacking ...")
    subprocess.run(repack_cmd, check=True)


def verify(fname, chains_expected):
    """Verify that the truncated file data is identical to original."""

    print("Verifying ...")
    for chain in chains_expected:
        try:
            os.remove(fname[:-3] + ".unsaved_steps")  # unsaved steps
        except FileNotFoundError:
            pass
        try:
            os.remove(fname[:-3] + "_truncated.unsaved_steps")  # unsaved steps
        except FileNotFoundError:
            pass

        backends = {
            "original": HDFBackend(fname, name=chain),
            "truncated": HDFBackend(fname[:-3] + "_truncated.h5", name=chain),
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


def process(fname):
    """Process the file by truncating, repacking, and verifying."""

    chains_expected = truncate(fname)
    repack(fname)
    verify(fname, chains_expected)
    os.replace(fname[:-3] + "_truncated.h5", fname)
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
