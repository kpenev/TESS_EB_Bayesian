# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This directory is an unpacked copy of upstream **dynesty 3.1.0** (dynamic nested sampling, github.com/joshspeagle/dynesty), placed inside the TESS_EB_Bayesian repo. It has no `.git` or `PKG-INFO` of its own (the files show as untracked in the parent repo). `AGENTS.md` holds the upstream contributor guidelines; the parent repo's `CLAUDE.md` describes the EB project that may use this package.

Because the version comes from `setuptools_scm` and there is no git metadata here, `pip install .` may fail to detect a version. If it does, set `SETUPTOOLS_SCM_PRETEND_VERSION=3.1.0`, or skip installing and use `PYTHONPATH=py` as shown below.

The top-level `priors.py` is a standalone example module. It is not part of the package and nothing imports it.

## Commands

```bash
pip install .[dev]                       # package + test/docs/lint deps

# Fast test suite (what CI runs); tests/ has a conftest.py so `from utils import ...` resolves
OMP_NUM_THREADS=1 PYTHONPATH=py pytest -m "not slow"
PYTHONPATH=py pytest tests/test_gau.py::test_gaussian # single test
PYTHONPATH=py pytest -m "not slow" --dist loadgroup -n 10   # parallel (pytest-xdist)
PYTHONPATH=py pytest -m "not slow" --cov=dynesty      # coverage (target ~91%, don't lower it)

pylint --fail-under=9 --ignored-modules=scipy.special py/dynesty/*py   # CI lint gate
cd docs && make html SPHINXOPTS="-W --keep-going"                      # docs; warnings fail
```

- The full suite takes about 30 minutes; mark long tests `@pytest.mark.slow`. `pytest.ini` uses `--strict-markers`, so a new marker must be registered there first.
- Test seed: tests must get their RNG from `tests/utils.py:get_rstate()`. You can override the seed with the env var **`DYNESTY_TEST_RANDOMSEED`**. (TESTING.md and AGENTS.md call it `DYNESTY_RANDOM_SEED`, which is wrong; the code reads `DYNESTY_TEST_RANDOMSEED`.) Set `DYNESTY_TEST_PRINTING=1` and pass `-s` to see sampler progress.
- `tests/test_notebooks.py` runs `demos/*.ipynb`. It needs dynesty installed in the default `python3` Jupyter kernel.
- CI also fails if the package calls `np.random.*` directly, except through `default_rng`, `Generator`, or `SeedSequence` (plotting.py is exempt). Pass a `rstate` Generator around instead.

## Architecture

All package code is in `py/dynesty/`.

**Entry points (`dynesty.py`).** `NestedSampler` is a factory: its `__new__` validates options in `_common_sampler_init` (which wraps the likelihood and prior transform, picks the bound and internal sampler, and sets up the pool). It then draws the initial live points and returns a plain `sampler.Sampler` instance, not a `NestedSampler`. `DynamicNestedSampler` subclasses `dynamicsampler.DynamicSampler`. The long user-facing docstrings for both are built by `_assemble_sampler_docstring`, so edit option docs there.

**Static sampler loop (`sampler.py: Sampler`).** `run_nested` calls the `sample()` generator, which replaces the worst live point each iteration through `_new_point`, then `_get_point_value`, then `_fill_queue`. `_fill_queue` proposes `queue_size` starting points from the current bound (`propose_live`). It packs them into `SamplerArgument`s with `internal_sampler.prepare_sampler` and evaluates them in parallel with `mapper(internal_sampler.sample, args)`. The run starts by sampling the unit cube. `update_bound_if_needed` switches to the real bound (`bound_next`/`internal_sampler_next`) once efficiency drops, then refits it every `bound_update_interval` likelihood calls.

**Two pluggable layers:**
- **Bounds (`bounding.py`)**: subclasses of `Bound`: `UnitCube` ('none'), `Ellipsoid` ('single'), `MultiEllipsoid` ('multi', the default), `RadFriends` ('balls'), `SupFriends` ('cubes'). Each implements `update` (refit to live points, with optional bootstrap expansion), `sample`/`samples`, `contains`, and `get_random_axes`. `sampler._get_bound` maps the string names to classes.
- **Internal samplers (`internal_samplers.py`)**: subclasses of `InternalSampler`: `UniformBoundSampler` ('unif'), `UnitCubeSampler`, `RWalkSampler`, `SliceSampler`, `RSliceSampler`. `sample(args)` is a **staticmethod** that runs in worker processes with no access to the instance, so everything it needs must travel in the `SamplerArgument` (including `kwargs`). `tune()` runs in the parent and adjusts `self.scale`. `sample='auto'` picks by dimension: unif below 10, rwalk for 10–20, rslice above 20 (`dynesty._get_internal_sampler`).

**Dynamic sampling (`dynamicsampler.py`).** `DynamicSampler` runs a baseline static run (`self.sampler`, a `Sampler`). It then adds batches over likelihood ranges chosen by `weight_function`. Each batch gets a fresh `Sampler` built by `_configure_batch_sampler` from the main sampler's bound and live points. `combine_runs` merges the results, and `stopping_function` decides when to stop.

**Shared utilities (`utils.py`).** This module contains:
- `Results`, a dict-like run output; `results.py` only re-exports it.
- `LogLikelihood`, which wraps the user's likelihood and handles blobs and evaluation history.
- `RunRecord` and the progress printers.
- Run post-processing: `jitter_run`, `resample_run`, `merge_runs`, `reweight_run`, `unravel_run`.
- Checkpointing: `save_sampler`/`restore_sampler`, which pickle the sampler and are used by `checkpoint_file=` and `resume=True` in `run_nested`.

**Parallelism (`pool.py`).** `dynesty.pool.Pool` is a multiprocessing pool that ships the likelihood and prior transform to workers once (`FunctionCache`), so they aren't pickled on every call. Any pool with a `.map` also works. `use_pool={...}` controls which stages (prior transform, loglikelihood, propose_point, update_bound) use it.

**Periodic/reflective parameters** are handled in unit-cube space: `nonbounded` masks from `utils.get_nonbounded`, wrapping and reflection in `unitcheck`/`apply_reflect`. Both bounds and internal samplers take these into account.

## Conventions

- Keep the public API stable. When behavior or an interface changes, update `docs/source/` and `CHANGELOG.md` in the same change.
- Every behavior change needs a test in `tests/test_*.py`.
- Upstream PRs must declare LLM use: which model, what it was asked, and how the output was checked (CONTRIBUTING.md).
