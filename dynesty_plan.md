# Switching from emcee to dynesty: plan

Goal: replace emcee with dynesty dynamic nested sampling for the posterior
of each target, without losing the high-likelihood regions that
`FindStartingPositions`-style optimization finds.

## Approach

dynesty draws its initial live points from the prior, so optimizer results
cannot be used as live points. Instead they are passed as
`extra_bound_points`: each time the bound is updated, the extra points with
log-likelihood above the current threshold L* are made part of the bound.
They never become live points, so the nested sampling statistics are
unchanged; the only cost is a larger bound.

This only affects where points are drawn for `sample='unif'`. The MCMC-style
samplers (`rslice`, `rwalk`, `slice`) start from live points and use the
bound only for proposal directions.

## Decisions

- **One extra point per peak, `extra_bound_method='union'`.** The bound is
  fitted to the live points only; then an ellipsoid with the shape of the
  nearest bound ellipsoid and volume (bound volume) / nlive is added around
  each extra point above L* that the bound does not cover.
- **Separate prior transform and likelihood (dynesty formalism).** The
  likelihood contains no prior term. This lets w be declared periodic, which
  the emcee formalism could not do.
- **Dynamic nested sampling**, with batches focused on the posterior
  (`wt_kwargs={'pfrac': 1.0}`).
- Bias is judged by posteriors on real targets, not by toy-model evidence.

## Status of the dynesty changes (in `dynesty-3.1.0/`)

Done:

- [x] `extra_bound_points` (unit cube points; log-likelihoods computed at
      initialization with the same wrapped likelihood) and
      `extra_bound_method` (`'fit'` or `'union'`) on `NestedSampler` and
      `DynamicNestedSampler`, passed to the base-run and batch samplers.
- [x] `'union'` implemented via `bounding.add_point_ellipsoids`.
- [x] Stopping criterion: remaining evidence estimated with
      max(best live log-likelihood, best extra point log-likelihood)
      (`Sampler._max_remaining_logl`).
- [x] `tests/test_extra_bound.py`: 6 fast and 5 slow tests, all passing.
- [x] No regressions in 9 existing test files (`test_gau`, `test_dyn`,
      `test_misc`, `test_sampler_interface`, `test_bound_interface`,
      `test_ellipsoid`, `test_pool`, `test_saver`, `test_egg`).

## Findings from the 2-D toy (broad mode + narrow dominant mode)

- Without extra points, a narrow enough mode is never found.
- Without the stopping criterion change, the run stops before reaching the
  narrow mode even with extra points: the remaining-evidence estimate used
  only live points.
- `'fit'` with tightly clustered optimizer points undercovers the mode (the
  fitted ellipsoid covered 3-47% of the true region for thousands of
  iterations) and biases the result low. The fit assumes points spread
  uniformly over {L > L*}; optimizer points sit at the peak.
- `'fit'` with a single point per peak is very inefficient: one point
  cannot form its own cluster (multi-ellipsoid needs 2 * ndim points per
  cluster, 42 for our 21 parameters).
- `'union'` found the mode in every run with unbiased evidence, but is
  expensive after discovery: until 2 * ndim live points reach the mode, one
  ellipsoid spans both modes. Bootstrap expansion makes this much worse
  (bound volume up to ~200x the unit cube). With `bootstrap=0, enlarge=1.25`
  it was 3-5x cheaper and still unbiased in the toy, but this must be
  checked on real targets.
- An alternative kept in reserve: a multi-width cloud of points around each
  peak, shaped by (J^T J)^-1, with `'fit'`. Needed only if `'union'` is too
  expensive in 21-D.

## Plan

### 1. dynesty changes

- [x] 1.1 Make `'union'` the default `extra_bound_method`; update the
      docstring advice. Keep `'fit'` for now. The `'union'` requires
      `bound='multi'` check only applies when extra points are given, and
      is also made at construction so `DynamicNestedSampler` fails early.
- [x] 1.2 Run the slow tests; install `pytest-xdist` in the scratch venv and
      run `test_resume` and the full fast suite. All 260 fast tests pass.
- [x] 1.3 Tests for resuming with extra points and for evaluating them
      through `dynesty.pool.Pool`. The extra-bound tests now use the
      default `'union'` with one point per peak (`'fit'` cases are slow).
      Resume tests (slow): static stops with `maxiter` +
      `add_live=False`; dynamic steps through `sample_initial` and saves
      mid base run, because a dynamic `maxiter` limit finishes the run
      rather than interrupting it. Both must reproduce an uninterrupted
      run's logz exactly.
  - Known upstream bug, not fixed: `utils.restore_sampler` fails for a
    `DynamicNestedSampler` saved before `run_nested()` starts
    (`sampler.sampler` is still `None`). Not hit by mid-run checkpoints.
- [x] 1.4 Delete `py/dynesty.egg-info`; add a `CHANGELOG.md` entry.
- [ ] 1.5 Install into the pipeline environment
      (`SETUPTOOLS_SCM_PRETEND_VERSION=3.1.0 pip install -e .`) when step 3
      first needs it.

### 2. Optional: cost check on a high-dimensional toy

- [ ] Only if 3.5 is too expensive: measure how long the merged-ellipsoid
      phase lasts in 21-D. Not used to judge bias.

### 3. TESS pipeline integration

- [ ] 3.1 Peak points: a new module (not an extension of
      `FindStartingPositions`, whose walker tweaking, random walkers and
      emcee-backend saving are not needed). Pipeline: collect starting
      samples from pluggable sources -> least-squares refine them all ->
      one point per peak (3.2) -> unit cube points for dynesty.
  - Refinement: reuse `utils.fit_least_squares` (LM, LC + SED residuals
    plus the emcee latent coordinates x as prior residuals) through a
    parallel map like `utils.lstsq_optimize_samples`, extended to also
    return `result.jac` for J^T J.
  - Fits run in emcee latent coordinates x. Convert to the dynesty unit
    cube directly: u = Phi(x) for every parameter except w (same ranges in
    both formalisms, including `truncnorm.ppf` for [M/H]); for w, emcee
    has w = -360 + 720 Phi(x), and dynesty uses w = -180 + 360 u, so
    u = ((w + 180) mod 360) / 360. Note `FindStartingPositions` produces w
    in roughly (-90, 270]; the wrap handles that.
  - Sources in physical parameters go to x with the existing
    `inverse_prior` for the fit.
  - Source 1: the `FindStartingPositions` scenario grid, through its
    `_get_init_scenarios()` (make public) and `optimize()`.
  - Source 2: previous fits (before the secondary mass fix), in
    `/mnt/md2/TESS_EBs/first1000/`: start from the sample with the highest
    emcee `log_prob` (prior term not subtracted: this is what the
    least-squares fit, with its x residuals, maximizes). Release FITS
    files (`release/tess{tic}.fits`, ~256k samples in physical parameters
    + `log_prob`) are the cheap source; for TICs without one, scan the
    chain in `tess{tic}_samples.h5` (`mcmc/chain`, `mcmc/log_prob`, only
    the first `iteration` rows are filled). These were computed with the
    old model; the least-squares refit with the current model polishes
    them.
  - Source 3 (later): Justesen & Albrecht 2021, excerpt in
    `data/ja_high_precision.fits`. It will not cover all 21 parameters, so
    the rest must be filled in (e.g. with `FindStartingPositions`
    heuristics).
  - Status: first draft in `scripts/likelihood_peaks.py` (sources 1 and
    2, parallel refinement keeping J^T J, x -> u conversion, command
    line); `FindStartingPositions` gained `optimize_scenario()`, public
    `get_init_scenarios()` and `add_scenario_grid_arguments()` (shared
    with `mcmc_sampling.py`). pylint 10/10. The previous-fit reader was
    checked on TIC 10057647 (max log_prob 10438.09 from the samples file
    vs 10438.06 from the release file; read takes ~50 s for a 5.9 GB
    file). The full parallel run has not been tried yet.
- [ ] 3.1a Fix the least-squares refinement. On TIC 10057647, starting
      from the stored best point, `fit_least_squares` stops after 2
      evaluations (status 3, `xtol`) with optimality 1.6e6, i.e. without
      optimizing. `x_scale='jac'` alone does not help; with
      `x_scale='jac', xtol=1e-8` it runs 21 evaluations, gains 8.85 in
      log_prob, then stops on `xtol` again (optimality 5.9e4). Jacobian
      column norms span 3.6e6 (`per`), 3.0e6 (`eclipse_time`) down to ~15,
      so scaling is extreme, but since rescaling does not fix it the
      suspected cause is a noisy finite-difference Jacobian. Next: compare
      Jacobians for several `diff_step` values, then tune `diff_step`,
      `x_scale` and `xtol` on this TIC and one more. This also affects the
      emcee starting positions (`lmfit_and_tweak` uses the same fit).
  - By design, `sed_sys`, both `prot`, both reflection and both beaming
    coefficients do not enter the residuals (column norm exactly 1: prior
    row only), to keep the fit out of unphysical regions of their very
    broad ranges; the fit pulls them toward x = 0. `lc_sys` does enter
    (column norm 4.65e3); `fit_sys_err=False` would fix both `lc_sys` and
    `sed_sys` at zero if that is intended.
  - Possible improvement: a second fit stage that also frees the
    reflection and beaming coefficients, starting from the first fit.
  - The model change is visible: the current model gives the stored best
    point of TIC 10057647 log_prob 10568.92 vs 10438.09 in the old fit.
- [ ] 3.2 One point per peak: group fit results by Mahalanobis distance
      using J^T J, after wrapping w so results 360 degrees apart count as
      the same peak; split a group if the likelihood at the midpoint
      between two results drops well below both; keep the
      highest-likelihood point of each group.
  - Cache the peak points with the samples, in the same per-TIC output
    file as the dynesty results, so each sampling file is self-contained:
    unit cube coordinates and log-likelihoods, plus provenance (source,
    starting sample, least-squares result, git hash) so they can be
    reused or regenerated when the model changes.
- [ ] 3.3 Prior transform and likelihood for dynesty:
  - `prior_transform(u)`: `low + (high - low) * u`, `10**` for log-uniform
    parameters, `truncnorm.ppf(u)` for [M/H], w over one period.
  - `loglikelihood(params)`: light curve + SED only, no prior term.
  - Derived quantities returned as a blob if `visualize.py` needs them.
  - Extra points converted to unit cube coordinates with the inverse of
    `prior_transform`.
  - w is currently (-360, 360) (`log_likelihood.py:466`), two periods. For
    dynesty use one period, w = -180 + 360 u (`likelihood_peaks.W_MIN`,
    `W_PERIOD`), and check whether anything downstream (visualize.py,
    release scripts) assumes a particular w range.
  - Leave the emcee path unchanged.
- [ ] 3.4 `scripts/nested_sampling.py`, alongside `mcmc_sampling.py`:
  - `DynamicNestedSampler(sample='unif', bound='multi', periodic=[w index],
    extra_bound_points=..., extra_bound_method='union', pool=...)`.
  - `run_nested(wt_kwargs={'pfrac': 1.0}, checkpoint_file=..., resume=...)`.
  - configargparse options like the other scripts.
  - Output readable by `visualize.py` and the release scripts.
  - File formats: dynesty's only on-disk format is the checkpoint, a
    pickle of the whole sampler (`utils.save_sampler`, written via a
    temporary file and renamed). `sampler.results` is in-memory only.
    Pickles refer to classes by name and can break when our code or
    dynesty changes, so use checkpoints only for resuming, and write
    `sampler.results` (samples, samples_u, logl, logwt, logz, ncall,
    blobs) plus the extra bound points and their provenance to a per-TIC
    HDF5 file next to the samples. That file is the self-contained record.
  - Use `dynesty.pool.Pool(n, loglike, prior_transform)` and pass
    `pool.loglike` / `pool.prior_transform`. Otherwise the checkpoint
    pickles the whole `LogLikelihood` object (light curves included) at
    every save, and it has to be picklable. With dynesty's pool the
    sampler holds only small stand-in functions and the real likelihood
    lives in the workers.
- [ ] 3.5 Posterior validation on real TICs:
  - Compare marginal posteriors (quantiles, 2-D contours) with trusted
    emcee chains.
  - Same TIC with bootstrap on and off and several seeds; agreement decides
    the bootstrap setting.
  - Include a TIC where emcee needed `FindStartingPositions` to find the
    right mode, and one with an extra point in a secondary peak.
  - Track likelihood calls and wall time.
- [ ] 3.6 HPC: SLURM templates and `new_sampling.py` /
      `continue_sampling.py` support.

### Order

1, then 3.1-3.4, then 3.5, then 3.6. Step 2 only if needed.

## Reference

- Toy scripts and the scratch venv used for testing (temporary):
  `/tmp/claude-1000/-home-kpenev-projects-git-TESS-EB-Bayesian-dynesty-3-1-0/8e56357e-a822-43c1-895d-2cc88889dbd2/scratchpad/`
  (`scan_tiny_mode.py`, `diag_bound.py`, `diag_coverage.py`).
- Run the dynesty tests with an editable install, since dynesty cannot be
  imported from the source tree without its version metadata.
