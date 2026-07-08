#!/usr/bin/env python3
"""Verify HDFBackend.get_chain_rows matches full-chain point indexing.

Standalone check that the memory-light selective row reader returns exactly
the same thing as the current restart code, which does::

    samples = backend.get_chain()               # whole chain in RAM
    initial_state = samples[top_indices]        # numpy point indexing

For many index patterns -- random, heavily duplicated, edge, reverse-order,
and the actual restart top-walker selection -- it asserts::

    backend.get_chain_rows(steps, walkers) == backend.get_chain()[steps, walkers]

element for element.

Usage:
    python test_get_chain_rows.py <samples.h5> [chain_name]

``chain_name`` defaults to "mcmc". Exits non-zero on any mismatch.
"""

import sys

import numpy

from hacked_emcee_hdf5_backend import HDFBackend


def check(backend, full, steps, walkers, label):
    """Assert get_chain_rows equals full[steps, walkers] for one pattern."""

    steps = numpy.asarray(steps)
    walkers = numpy.asarray(walkers)
    expected = full[steps, walkers]  # numpy point indexing == samples[top]
    got = backend.get_chain_rows(steps, walkers)

    assert got.shape == expected.shape, (
        f"{label}: shape {got.shape} != expected {expected.shape}"
    )
    assert numpy.array_equal(got, expected), (
        f"{label}: values differ (max abs diff "
        f"{numpy.abs(got - expected).max()})"
    )
    print(f"  OK  {label:32s} ({steps.size} rows)")


def main():
    """Run all equivalence checks against the given samples file."""

    fname = sys.argv[1]
    chain_name = sys.argv[2] if len(sys.argv) > 2 else "mcmc"

    backend = HDFBackend(fname, name=chain_name, read_only=True)
    niter = backend.iteration
    nwalkers, ndim = backend.shape
    print(
        f"chain {chain_name!r}: iteration={niter}, "
        f"nwalkers={nwalkers}, ndim={ndim}"
    )
    assert niter > 0, "chain has no stored steps"

    full = backend.get_chain()  # reference: (niter, nwalkers, ndim) in RAM
    assert full.shape == (niter, nwalkers, ndim), full.shape

    rng = numpy.random.default_rng(0)

    # 1. random pairs of varying length (the common case)
    for trial in range(25):
        size = int(rng.integers(1, nwalkers + 1))
        steps = rng.integers(0, niter, size=size)
        walkers = rng.integers(0, nwalkers, size=size)
        check(backend, full, steps, walkers, f"random#{trial}")

    # 2. duplicate-heavy patterns (point reads must handle repeats)
    check(
        backend, full,
        numpy.full(nwalkers, niter - 1),
        rng.integers(0, nwalkers, size=nwalkers),
        "all-on-last-step",
    )
    check(
        backend, full,
        numpy.zeros(nwalkers, dtype=int),
        numpy.arange(nwalkers),
        "first-step-all-walkers (fallback)",
    )
    check(
        backend, full,
        rng.choice([0, niter // 2, niter - 1], size=nwalkers),
        rng.choice(nwalkers, size=nwalkers),
        "few-unique-steps",
    )
    row_s = int(rng.integers(0, niter))
    row_w = int(rng.integers(0, nwalkers))
    check(
        backend, full,
        numpy.full(7, row_s), numpy.full(7, row_w),
        "identical-row-x7",
    )

    # 3. edge singletons
    check(backend, full, [0], [0], "single-first-corner")
    check(backend, full, [niter - 1], [nwalkers - 1], "single-last-corner")

    # 4. reverse / unsorted order (get_chain_rows must not assume sorted)
    steps = numpy.arange(min(nwalkers, niter))[::-1]
    walkers = numpy.arange(steps.size) % nwalkers
    check(backend, full, steps, walkers, "reverse-order")

    # 5. the real restart selection path end-to-end
    log_prob = backend.get_log_prob()
    ordered = numpy.unique(log_prob, return_index=True)[1]
    flat = rng.choice(ordered[-nwalkers:], nwalkers)
    steps, walkers = numpy.unravel_index(flat, log_prob.shape)
    check(backend, full, steps, walkers, "restart-top-selection")
    assert log_prob[steps, walkers].shape == (nwalkers,)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
