# Bayesian Analysis of TESS Eclipsing Binaries

Bayesian characterization of eclipsing binaries (EBs) observed by NASA's TESS
mission. TESS photometry and multi-band SED data (PanSTARRS, 2MASS, WISE, Gaia)
are fit together with a model that includes stellar evolution and
out-of-eclipse effects (ellipsoidal variations, reflection, Doppler beaming).
Posterior distributions of the physical parameters are sampled with `emcee`.

## Installation

Main dependencies: `numpy`, `scipy`, `astropy`, `astroquery`, `emcee`, `h5py`,
`batman-package`, `dustmaps`, `sqlalchemy`, `configargparse`, `matplotlib`,
`pandas`, `django` (web interface only), `phoebe` (tests/comparisons only), and
the project-external `general_purpose_python_modules` package (stellar
evolution interpolation, orbital mechanics, chain I/O, plotting utilities).

If `batman` fails to compile because OpenMP is unavailable, clone the batman
repository, edit `setup.py` so that `compiler_has_openmp()` returns `False`,
and run `pip3 install .` in that directory. Leave the batman directory before
checking that `import batman` works.

## Usage

All scripts live in `scripts/` and should be run from there. Every script
accepts `--help`. Options can be given on the command line or in a config file
named after the script (e.g. `mcmc_sampling.cfg`) or passed with
`--config-file`; `--generate-config-file` writes out the current settings.

```bash
# Sample the posterior for a single target
python mcmc_sampling.py <TIC_ID> [--num-parallel 16]

# Plot the sampling results and light curve
python visualize.py <TIC_ID>

# Create SLURM jobs for a table of selected TICs / continue a job group
python new_sampling.py <TIC_TABLE> --hpc <juno|ls6|vista|ganymede>
python continue_sampling.py <JOB_GROUP>

# Collect maximum-likelihood parameters of finished TICs into a FITS table
python collect_max_likelihood.py

# Prepare burned-in FITS files of the samples for public release
python prepare_sample_release.py <TIC_ID> [<TIC_ID> ...]

# Eclipse timing variation analysis
python etv_analysis.py <TIC_ID>
```

## How It Works

### Class hierarchy

The binary model and the likelihood are built as two class hierarchies:

```
batman.TransitParams
  └── BinaryParams      (binary_parameters.py)
        └── EBEERBinary (ebeer.py)
              └── Binary (binary.py)

TESSTarget              (tess_target.py)
  └── LogLikelihood     (log_likelihood.py)
```

- `BinaryParams`: the orbit (Kepler angle conversions, eclipse phases) and
  the stars. Stellar radii and temperatures are interpolated from isochrone
  grids (`CMDInterpolator` from `general_purpose_python_modules`) for the
  given masses, age, and metallicity, and gravity darkening comes from tables.
  `to_phoebe`/`from_phoebe` convert to and from PHOEBE, but only for testing
  against PHOEBE; PHOEBE is not used for sampling.
- `EBEERBinary`: adds the out-of-eclipse effects: Doppler beaming,
  reflection, and ellipsoidal variations.
- `Binary`: combines eclipses and out-of-eclipse effects into the model light
  curve and fits the BEER coefficients. It can be created directly from a set
  of sampled parameters (`Binary(from_mcmc=...)`).
- `TESSTarget`: downloads and prepares the TESS light curves of one TIC,
  organized by sector (SPOC preferred over QLP). It finds the eclipses with
  BLS, and the result is used to mask the data around them.
- `LogLikelihood`: the function being sampled. It detrends the light curve and
  combines the light curve likelihood (comparing a `Binary` model to the TESS
  data), the SED likelihood, and the priors.

### Sampling

`mcmc_sampling.py` finds starting walker positions by optimization
(`FindStartingPositions` in `find_starting_positions.py`) over a grid of age,
metallicity, and argument of periapsis. It then runs `emcee`, periodically
restarting the walkers to escape local maxima, and stores the chains in HDF5.

### Parameters

The 21 sampled parameters (`sample_params.py`) are: total mass, mass ratio,
age, [M/H], period, eccentricity, argument of periapsis, primary impact
parameter, eclipse time, four limb darkening coefficients, two rotation
periods, two reflection and two beaming coefficients, and light curve and SED
systematic-error terms.

### Data and results

File locations are defined in `scripts/paths.py`.

- `data/`: isochrone grids, gravity darkening tables, broadband filter data,
  EB catalogs, and a SQLite cache of SED fits and BLS periodograms
  (`mcmc_cache.sqlite`). Dust maps are read through the `dustmaps` package.
- `results/tess{TIC}_samples.h5`: MCMC chains (see below).
- `results/release/tess{TIC}.fits`: release files.

### MCMC chain storage and max-likelihood extraction

Each `results/tess{TIC}_samples.h5` file holds one or more `emcee` chains as
HDF5 groups (written by `hacked_emcee_hdf5_backend.HDFBackend`). Each group
contains:

- `chain` (nsteps, nwalkers, ndim): the sampled coordinates.
- `log_prob` (nsteps, nwalkers): the log posterior of each sample.
- `blobs` (nsteps, nwalkers): the 21 `SampleParams` values of each sample.
- `attrs["iteration"]`: the number of steps actually written. The datasets are
  pre-allocated and can be much longer, with the unused rows filled with
  zeros, so always slice to `iteration` before using them.

The production chain is the group named `mcmc`. Restarts may also leave
`prelim_mcmc_N` groups. Use only `mcmc` when looking for the best sample: the
log-probability definition can differ between runs, so a preliminary chain can
give a spurious maximum.

`chain_analysis.get_max_likelihood_params(samples_fname)` returns
`(SampleParams, log_prob)` for the highest-`log_prob` sample of the `mcmc`
chain, reading only that one row of `blobs`. It is used by `measure_etv.py`
and `collect_max_likelihood.py`. Note that the plotting helpers in
`visualize.py` only read the most recent `--max-plot-steps` steps and fall back
to `prelim_mcmc_N` groups, so they are not suitable for finding the global
maximum.

## Review Interface (BUI) and Workflow

Deciding what to sample, checking the results, and deciding what to do next are
done through a Django web app in `scripts/bui/`. It works together with the
SLURM job scripts.

### The jobs database

Job groups, the TICs in each review table, and each TIC's review status are
stored in the SQLite database `TESS_EBs.db`. It is opened **relative to the
current working directory** (`scripts/bui/db_interface.py`). Run the web
server, the plot renderer, and all job/sampling scripts from `scripts/` so they
all use `scripts/TESS_EBs.db`.

Each review table (currently `sample_prsa`, the Prša et al. TESS EB catalog)
has one row per TIC with:

- `status`: the review decision (see below).
- `skip_review`: hides the TIC from review lists without changing its status.
- `job_group`, `job_id`: which SLURM job group and job the TIC was assigned to.

A companion table `<table>_rendered` records which plots exist for each TIC.
The `job_groups` table records the HPC, review table, and number of jobs of
each job group.

Status values, as labeled in the web interface:

| Status | Label                | Meaning                                                        |
|--------|----------------------|----------------------------------------------------------------|
| 0      | pending              | Not yet reviewed                                               |
| 1      | bad                  | Rejected, do not sample                                        |
| 2      | continue             | Keep sampling (the default `--continue-status`)                |
| 3      | fix                  | Needs manual attention before sampling can proceed             |
| 4      | changed likelihood   | Data selection or SED settings changed; restart the chain     |
| 5      | finished             | Sampling is done (default for `collect_max_likelihood.py`)     |

### Running the web interface

```bash
cd scripts
python bui/manage.py migrate      # first time only (Django sessions/admin)
python bui/manage.py runserver
```

Then open `http://127.0.0.1:8000/select_ticids/sample_prsa/<mode>/`, where
`<mode>` is one of:

- `lightcurve`: the detrended TESS light curve, for deciding whether to
  sample a TIC and which data to use.
- `starting`, `best`, `convergence`: plots of the starting walker positions,
  the best-fit model, and chain convergence.
- `sampling`: the best-fit and convergence plots side by side, for reviewing
  sampling results.

The page shows one TIC at a time, with lists of TICs by status on the side.
The links at the top set the status of the displayed TIC and move to the next
one. **Skip** moves on without changing anything, and **Disable/Enable review**
toggles `skip_review`. Other controls:

- **Data selection**: click the SPOC/QLP sector numbers, or the provider and
  OOE/BLS flags, to exclude or include data (stored by `exclude_data.py`). In
  `lightcurve` mode the change is immediate and invalidates the cached BLS
  result; click "click to replot" to regenerate the plot.
- **Bad SED threshold/penalty**: per-TIC settings for down-weighting outlier
  SED points (stored in the SED cache).
- In the sampling modes, data selection and SED changes are staged until you
  click **Apply Changes**, which saves them and sets the status to
  *changed likelihood*.
- **Update rendered**: rescan the plot directory for newly rendered plots.

The table name `sample_prsa` is hard-coded in `bui/select_ticids/urls.py`.

### Rendering plots

The web interface only displays pre-rendered PNGs from
`results/render/<table>/<mode>/tess<TIC>.png`. Render them from `scripts/`:

```bash
PYTHONPATH=. python bui/select_ticids/plots.py <lightcurve|starting|best|convergence> \
    --table-name sample_prsa [--limit-to-statuses 2 4] [--limit-to-jobs 6:] \
    [--skip-rendered] [--auto-download] [--num-parallel 8]
```

`--auto-download` rsyncs missing samples files from the HPC scratch
directories. Note that the default `--table-name` (`prsa_ebs`) is not the
table currently being reviewed.

### Typical workflow

1. **Select targets**: render `lightcurve` plots and review them, setting each
   TIC's status and excluding bad sectors.
2. **Launch sampling**: `python new_sampling.py sample_prsa --hpc <hpc>
   --nodes-per-job N --num-jobs M [--restrict-status ...]` creates a job group,
   assigns TICs that have no job yet to jobs, and writes SLURM files from the
   templates in `slurm/<hpc>/`. Without `--restrict-status`, every TIC with a
   positive status is included. `--hpc` defaults to the cluster detected from
   the hostname.
   Supported clusters are `juno`, `ls6`, `vista`, and `ganymede`
   (`command_line_util.tic_per_node`).
3. **Review results**: render `best`, `convergence`, and `starting` plots and
   review them in `sampling` mode. Mark each TIC as *continue*, *fix*,
   *finished*, or *bad*, or change its data selection or SED settings and
   *Apply Changes*. TICs can also be flagged as not needing further review until
   autometed convergence checks are satisfied.
4. **Continue**: `python continue_sampling.py <JOB_GROUP> --continue-statuses 2
   --changed-likelihood-statuses 4` regenerates the job files of the group.
   TICs with those statuses keep their slots. The slots of all other TICs
   (e.g. *finished*, *bad*, *fix*) are given to TICs that are still waiting
   for a job. Without `--continue-statuses`, every positive status is
   continued. `mcmc_sampling.py` resumes *continue* TICs from their existing
   chains. For *changed likelihood* TICs it restarts from the last step with
   the new likelihood and then resets the status to *continue*.
   `--submit-jobs` submits the jobs when run on the cluster. Repeat steps 3–4
   until TICs are finished.
5. **Collect results**: `collect_max_likelihood.py` and
   `prepare_sample_release.py` for the *finished* TICs.
