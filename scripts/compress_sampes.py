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


def truncate_fresh(fname):
    """Create a fresh truncated file by copying only valid iterations.

    Avoids in-place dataset resize, which fails on very large files.
    Always writes to fname[:-3] + "_truncated.h5".
    Returns (trunc_fname, chains_found).
    """

    assert fname.endswith(
        ".h5"
    ), "Input file must be an HDF5 file with .h5 extension."

    trunc_fname = fname[:-3] + "_truncated.h5"
    chains_found = []
    truncated_dsets = {"blobs", "chain", "log_prob"}

    with h5py.File(fname, "r") as src, h5py.File(trunc_fname, "w") as dst:
        for chain in src:
            chains_found.append(chain)
            iterations = src[chain].attrs["iteration"]
            print(
                f"Processing chain {chain} containing "
                f"{iterations} iterations..."
            )
            grp = dst.create_group(chain)
            for attr_key, attr_val in src[chain].attrs.items():
                grp.attrs[attr_key] = attr_val
            for dset_name, dset in src[chain].items():
                if dset_name in truncated_dsets:
                    print(f"{dset_name} has shape: {dset.shape}")
                    # Read one row to resolve the true numpy shape,
                    # which may differ from dset.shape when blobs use a
                    # subarray dtype (h5py stores it as 2-D but numpy
                    # expands the inner dimensions on read).
                    sample = dset[:1]
                    inner_shape = sample.shape[1:]
                    out_dset = grp.create_dataset(
                        dset_name,
                        shape=(iterations, *inner_shape),
                        maxshape=(None, *inner_shape),
                        dtype=sample.dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                    batch_size = 100000
                    for start in range(0, iterations, batch_size):
                        end = min(start + batch_size, iterations)
                        out_dset[start:end] = dset[start:end]
                        print(
                            f"  {dset_name}: copied rows "
                            f"{start}:{end} / {iterations}"
                        )
                    print(f"Copied {dset_name} truncated to {iterations} rows")
                else:
                    src[chain].copy(dset_name, grp)

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
    batch_size = 100000
    verified_dsets = ["chain", "log_prob", "blobs"]

    with h5py.File(orig_fname, "r") as orig, h5py.File(
        repacked_fname, "r"
    ) as trunc:
        for chain in chains_expected:
            orig_grp = orig[chain]
            trunc_grp = trunc[chain]

            orig_iter = orig_grp.attrs["iteration"]
            trunc_iter = trunc_grp.attrs["iteration"]
            assert orig_iter == trunc_iter, (
                f"Iteration count mismatch for chain {chain}: "
                f"{orig_iter} vs {trunc_iter}"
            )

            for dset_name in verified_dsets:
                orig_dset = orig_grp[dset_name]
                trunc_dset = trunc_grp[dset_name]
                for start in range(0, orig_iter, batch_size):
                    end = min(start + batch_size, orig_iter)
                    assert (
                        orig_dset[start:end] == trunc_dset[start:end]
                    ).all(), (
                        f"{dset_name} mismatch for chain {chain} "
                        f"at rows {start}:{end}"
                    )
                    print(
                        f"  {dset_name}: verified rows "
                        f"{start}:{end} / {orig_iter}"
                    )

            print(f"Verified chain {chain}")
        for fname in [orig_fname, repacked_fname]:
            try:
                os.remove(fname[:-3] + ".unsaved_steps")  # unsaved steps
            except FileNotFoundError:
                pass


def process(fname, verify_only=False):
    """Process the file by truncating, repacking, and verifying."""

    trunc_fname = fname[:-3] + "_truncated.h5"
    if verify_only:
        chains_expected = list_chains(fname)
    else:
        chains_expected = truncate(fname, trunc_fname)
        repack(trunc_fname)
    verify(fname, trunc_fname, chains_expected)
    print(f"Renaming {trunc_fname!r} -> {fname!r}")
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
        "--in-place",
        action="store_true",
        help="If passed, instead of working on a copy of the file truncating "
        "is done in-place.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip compression; only verify that fname[:-3] + '_truncated.h5' "
        "matches the original fname. Useful when compression already ran but "
        "verification was not completed.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Assume all compression steps were already done just run "
        "verification.",
    )
    return parser.parse_args()


def main(args):
    """Keep global namespace clean."""

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


if __name__ == "__main__":
    main(parse_command_line())
