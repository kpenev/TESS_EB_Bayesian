# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian characterization of eclipsing binaries (EBs) observed by NASA's TESS mission. Combines TESS photometry with multi-band SED data (PanSTARRS, 2MASS, WISE, Gaia) and models secondary effects (ellipsoidal variations, reflected light, Doppler beaming) to derive posterior distributions of physical parameters via MCMC sampling with `emcee`.

## Running Key Scripts

All scripts live in `scripts/` and use `configargparse` (config files + CLI args). Config files are auto-discovered by script name (e.g., `mcmc_sampling.cfg` for `mcmc_sampling.py`).

```bash
# Run MCMC sampling for a single target (from scripts/ directory)
python mcmc_sampling.py <TIC_ID> [--num-parallel 16] [--config-file myconfig.cfg]

# Generate visualization plots
python visualize.py <TIC_ID>

# Create SLURM batch jobs for HPC sampling
python new_sampling.py <TIC_TABLE> --hpc <juno|ls6|ganymede>

# Continue existing sampling jobs
python continue_sampling.py <JOB_GROUP>

# Collect max-likelihood params of all finished-sampling TICs into a FITS table
# (incremental: only newly finished TICs are read and appended on re-run)
python collect_max_likelihood.py [--finished-status 5]

# Compare models against PHOEBE reference
python test_against_phoebe.py
python test_against_phoebe_eclipsing.py
```

## Linting

```bash
pylint --rcfile=scripts/.pylintrc scripts/<file>.py
```

The `.pylintrc` enforces: 80-char line length, snake_case naming, max 5 args per function. Modules `numpy`, `scipy`, `astropy` are in `ignored-modules` (dynamic attributes). Constants use a relaxed regex (`[a-z_][a-z0-9_]{2,30}$`) rather than strict `UPPER_CASE`.

## Architecture

### Class Hierarchy for Binary Modeling

```
batman.TransitParams
  └── BinaryParams          (binary_parameters.py) - orbital mechanics, eclipse phases, CMD interpolation
        └── EBEERBinary     (ebeer.py) - out-of-eclipse effects (beaming, reflection, ellipsoidal)
              └── Binary    (binary.py) - combined light curve generation, BEER coefficient fitting
```

`BinaryParams` wraps `batman.TransitParams` and adds stellar evolution via `CMDInterpolator` (from `general_purpose_python_modules`), Kepler angle conversions, and gravity darkening. It uses PHOEBE for radius/temperature lookups.

### MCMC Sampling Pipeline

1. **`mcmc_sampling.py`** - Main entry point. Orchestrates the full pipeline:
   - Creates `LogLikelihood` instance for the target TIC ID
   - Uses `FindStartingPositions` to optimize initial walker positions across a grid of (age, metallicity, argument of periapsis) values
   - Runs `emcee.EnsembleSampler` with periodic restarts to escape local minima (every `--restart-steps` steps)
   - Stores chains in HDF5 via custom `HDFBackend` (hacked_emcee_hdf5_backend.py)
   - Supports resuming from existing chain files and git hash verification

2. **`log_likelihood.py`** (LogLikelihood, extends TESSTarget) - The core likelihood function combining:
   - Light curve likelihood from TESS photometry (with BLS-based eclipse masking)
   - SED likelihood from multi-band photometry
   - Priors on all 21 `SampleParams` parameters (some log-uniform: age, reflection/beaming coefficients, systematics)

3. **`find_starting_positions.py`** (FindStartingPositions) - Optimization-based walker initialization using scipy.optimize with multiprocessing

### The 21 MCMC Parameters (SampleParams)

Defined in `sample_params.py` as a namedtuple: `mtotal`, `mratio`, `age_gyr`, `meh` ([M/H]), `per`, `ecc`, `w`, `primary_impact_param`, `eclipse_time`, limb darkening (4), rotation periods (2), reflection coefficients (2), beaming coefficients (2), `lc_sys`, `sed_sys`.

### Data Flow

- **Input**: TESS light curves downloaded via `astroquery` (`download_lcs.py`), detrended (`detrending.py`), with manual exclusions (`exclude_data.py`)
- **Caching**: SED fits and BLS periodogram results cached in SQLite (`cache_interface.py` using SQLAlchemy ORM, database at `data/mcmc_cache.sqlite`)
- **Output**: MCMC chains stored as HDF5 (`results/tess{tic_id}_samples.h5`), release files as FITS (`results/release/tess{tic_id}.fits`)

### MCMC Chain Storage & Max-Likelihood Extraction

Each `results/tess{tic_id}_samples.h5` holds one or more emcee chains as HDF5 groups (`hacked_emcee_hdf5_backend.HDFBackend`). Per group:
- `chain` (nsteps, nwalkers, ndim) — sampled coordinates
- `log_prob` (nsteps, nwalkers) — log posterior of each sample
- `blobs` (nsteps, nwalkers) — the derived 21 `SampleParams` values per sample
- `attrs["iteration"]` — steps actually written; can be far smaller than the pre-allocated dataset length, so **always slice to `iteration`** (unused rows are zero-filled and would corrupt any `min`/`max`)

The production chain is the group named `mcmc`. Restarts may also leave `prelim_mcmc_N` groups — **only read `mcmc` for max-likelihood / "best" samples**: the log-probability definition can differ between runs, so a preliminary chain could yield a spurious maximum.

`chain_analysis.get_max_likelihood_params(samples_fname)` returns `(SampleParams, log_prob)` for the single highest-`log_prob` `mcmc` sample. It reads the (small) `log_prob`, picks the winner with `nanargmax`, then reads back only that one blob row via the backend stride trick — `get_blobs(discard=0, thin=best_step+1)[0][best_walker]`, which exploits `get_value`'s `discard + thin - 1` slice start so `[0]` lands exactly on `best_step`. Shared by `measure_etv.py` and `collect_max_likelihood.py`.

`chain_analysis.py` holds non-plotting chain helpers extracted from `visualize.py` (which imports them back); it pulls in neither matplotlib nor phoebe, so it is safe to import from lightweight scripts. Note `visualize.get_plot_data` caps reads at `--max-plot-steps` most-recent steps and falls back to `prelim_mcmc_N` groups, so it is *not* suitable for finding a true global maximum.

### TIC Status Tracking (finished sampling / review)

TIC selection and sampling progress live in dynamically-created, per-job-group tables (`bui/select_ticids/data_model.py`, `get_ticid_select_tables(name)`), one row per TIC keyed by `.id == TIC ID`, with an integer `status` column and a `skip_review` boolean. A specific `status` value marks a TIC as finished sampling (5 by current convention, but treat it as configurable). Query outside Django via `bui.db_interface.Session` + `get_ticid_select_tables(tablename, must_exist=True)[0]`; enumerate every review table with the distinct `JobGroup.select_tic_table` values (see `clear_skip_review` and `continue_sampling.py` for the pattern).

### Path Configuration

All file paths defined centrally in `paths.py`. Key directories:
- `data/` - Isochrone grids, gravity darkening tables, dust maps, EB catalogs
- `results/` - MCMC chain outputs, SLURM job database, rendered plots
- `slurm/{hpc}/` - SLURM batch script templates per HPC cluster

### Web Interface (bui/)

Django app (`bui/`) for managing TIC ID selection and job tracking. Uses SQLAlchemy (`bui/db_interface.py`) for the jobs database and Django ORM for the web models. Dynamic table creation in `bui/select_ticids/data_model.py`.

### External Dependencies

- **`general_purpose_python_modules`** - Project-external package providing `CMDInterpolator` (stellar evolution), `kepler_angles` (orbital mechanics), `emcee_util` (chain I/O), `multiprocessing_util`, and `visuals` (corner plots)
- **`batman`** - Transit/eclipse light curve modeling (base class for `BinaryParams`)
- **`phoebe`** - Binary star evolution modeling (used for radius/temperature calculations in `BinaryParams`)
- **`emcee`** - Ensemble MCMC sampler

### HPC Support

Three supported clusters configured in `command_line_util.py`: `juno` (4 TICs/node), `ls6` (8 TICs/node), `ganymede` (1 TIC/node). SLURM templates live in `slurm/{hpc}/`. The `new_sampling.py` and `continue_sampling.py` scripts generate and manage batch jobs.
