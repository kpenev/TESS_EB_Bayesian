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


def verify(orig_fname, repacked_fname, chains_expected, max_read_steps):
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
        # Steps past this are unused padding in the original and must never be
        # read (the backend applies the same limit to all its reads).
        iterations = backends["original"].iteration
        assert iterations == backends["truncated"].iteration, (
            f"Iteration count mismatch for chain {chain}: "
            f"{iterations} vs {backends['truncated'].iteration}"
        )
        with h5py.File(orig_fname, "r") as orig_f:
            with h5py.File(repacked_fname, "r") as trunc_f:
                for dset in ["blobs", "chain", "log_prob"]:
                    orig_dset = orig_f[chain][dset]
                    trunc_dset = trunc_f[chain][dset]
                    assert (
                        trunc_dset.shape[0] == iterations
                        and trunc_dset.shape[1:] == orig_dset.shape[1:]
                    ), (
                        f"Shape mismatch in {dset} of chain {chain}: "
                        f"{orig_dset.shape} vs {trunc_dset.shape} for "
                        f"{iterations} iterations"
                    )
                    for start in range(0, iterations, max_read_steps):
                        end = min(start + max_read_steps, iterations)
                        assert (
                            orig_dset[start:end] == trunc_dset[start:end]
                        ).all(), (
                            f"Mismatch in {dset} of chain {chain} among "
                            f"steps {start} - {end}"
                        )
                        print(
                            f"Verified {dset} steps {start} - {end} of chain "
                            f"{chain}"
                        )
        print(f"Verified chain {chain}")


def process(fname, verify_only=False, max_read_steps=100000):
    """Process the file by truncating, repacking, and verifying."""

    trunc_fname = fname[:-3] + "_truncated.h5"
    if verify_only:
        chains_expected = list_chains(fname)
    else:
        chains_expected = truncate(fname, trunc_fname)
        repack(trunc_fname)
    verify(fname, trunc_fname, chains_expected, max_read_steps)
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
        default=1,
        help="The number of parallel processes to use.",
    )
    parser.add_argument(
        "--max-read-steps",
        type=int,
        default=100000,
        help="Verification compares the chains in chunks of at most this many "
        "steps at a time, to limit the memory used for large files.",
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
                partial(
                    process,
                    verify_only=args.verify_only,
                    max_read_steps=args.max_read_steps,
                ),
                args.filenames,
            )
    else:
        for fname in args.filenames:
            process(
                fname,
                verify_only=args.verify_only,
                max_read_steps=args.max_read_steps,
            )

if __name__ == '__main__':
    main(parse_command_line())
