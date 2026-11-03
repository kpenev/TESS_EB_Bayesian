import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import logsumexp
import dynesty
import dynesty.pool as dypool
from dynesty import bounding
from utils import get_rstate, get_printing
"""
Tests of extra_bound_points: points the bound must keep covering while
their log-likelihood is above the current threshold.
"""

printing = get_printing()

NDIM = 2
NLIVE = 200
# A broad low mode and a narrow high mode that holds 90% of the evidence.
# The narrow mode is small enough that a run without extra points almost
# never finds it.
BROAD_CENTER, BROAD_SIGMA, BROAD_LOGAMP = np.array([0.3, 0.3]), 0.1, 0.0
NARROW_CENTER, NARROW_SIGMA = np.array([0.8, 0.8]), 2e-4
NARROW_LOGAMP = np.log(9 * (BROAD_SIGMA / NARROW_SIGMA)**2)
LOGZ_BROAD = BROAD_LOGAMP + np.log(2 * np.pi * BROAD_SIGMA**2)
LOGZ_NARROW = NARROW_LOGAMP + np.log(2 * np.pi * NARROW_SIGMA**2)
LOGZ_TRUTH = np.logaddexp(LOGZ_BROAD, LOGZ_NARROW)
NARROW_MASS_TRUTH = np.exp(LOGZ_NARROW - LOGZ_TRUTH)


def loglike(x):
    broad = BROAD_LOGAMP - np.sum((x - BROAD_CENTER)**2) / 2 / BROAD_SIGMA**2
    narrow = (NARROW_LOGAMP -
              np.sum((x - NARROW_CENTER)**2) / 2 / NARROW_SIGMA**2)
    return np.logaddexp(broad, narrow)


def prior_transform(u):
    return u


def narrow_mode_mass(results):
    weights = np.exp(results.logwt - logsumexp(results.logwt))
    in_narrow = (np.linalg.norm(results.samples - NARROW_CENTER, axis=1) <
                 10 * NARROW_SIGMA)
    return weights[in_narrow].sum()


def make_peak_points(rstate, npoints):
    """Mimic optimizer results: points scattered within the narrow mode,
    one of them exactly at its peak."""
    jitter = rstate.normal(scale=NARROW_SIGMA / 2, size=(npoints, NDIM))
    jitter[0] = 0
    return NARROW_CENTER + jitter


def test_add_point_ellipsoids():
    rstate = get_rstate()
    points = BROAD_CENTER + 0.05 * rstate.standard_normal(size=(100, NDIM))
    bound = bounding.MultiEllipsoid(NDIM)
    bound.update(points, rstate=rstate)
    inside = BROAD_CENTER[None, :]
    outside = NARROW_CENTER[None, :]
    logvol_per_point = bound.logvol - np.log(len(points))

    # Points already covered leave the bound unchanged.
    assert bounding.add_point_ellipsoids(bound, inside,
                                         logvol_per_point) is bound

    new_bound = bounding.add_point_ellipsoids(
        bound, np.concatenate((inside, outside)), logvol_per_point)
    assert new_bound.nells == bound.nells + 1
    assert new_bound.contains(outside[0])
    assert all(new_bound.contains(point) for point in points)
    assert np.isclose(new_bound.ells[-1].logvol, logvol_per_point)
    # The input bound is not modified.
    assert bound.nells == new_bound.nells - 1


def run_toy(rstate, dynamic=False, **kwargs):
    """Run the two-mode toy problem and return the results."""
    sampler_class = (dynesty.DynamicNestedSampler
                     if dynamic else dynesty.NestedSampler)
    sampler = sampler_class(loglike,
                            prior_transform,
                            NDIM,
                            nlive=NLIVE,
                            bound='multi',
                            sample='unif',
                            rstate=rstate,
                            **kwargs)
    if dynamic:
        sampler.run_nested(print_progress=printing,
                           dlogz_init=0.1,
                           maxbatch=2)
    else:
        sampler.run_nested(print_progress=printing, dlogz=0.1)
    return sampler.results


def test_baseline_misses_narrow_mode():
    # Without extra points the bound shrinks onto the broad mode and the
    # narrow mode is never found, so the evidence is badly underestimated.
    results = run_toy(get_rstate())
    assert narrow_mode_mass(results) < 0.01
    assert results.logz[-1] < LOGZ_TRUTH - 1


@pytest.mark.parametrize('dynamic,method,npoints', [
    (False, 'union', 1),
    pytest.param(True, 'union', 1, marks=pytest.mark.slow),
    pytest.param(False, 'fit', 10, marks=pytest.mark.slow),
    pytest.param(False, 'fit', 1, marks=pytest.mark.slow),
])
def test_extra_bound_points_recover_mode(dynamic, method, npoints):
    rstate = get_rstate()
    results = run_toy(rstate,
                      dynamic=dynamic,
                      extra_bound_points=make_peak_points(rstate, npoints),
                      extra_bound_method=method)
    assert abs(results.logz[-1] - LOGZ_TRUTH) < 5 * results.logzerr[-1]
    assert abs(narrow_mode_mass(results) - NARROW_MASS_TRUTH) < 0.1


def test_extra_bound_points_below_threshold_dropped():
    # With only the broad mode's points the bound must not grow.
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    NDIM,
                                    nlive=NLIVE,
                                    bound='multi',
                                    sample='unif',
                                    rstate=rstate,
                                    extra_bound_points=[[0.05, 0.95]],
                                    extra_bound_method='union')
    sampler.run_nested(print_progress=printing, dlogz=0.1)
    # The far corner has logl ~ -40, below the final threshold, so the
    # final bound must not include it.
    assert not sampler.bound.contains(np.array([0.05, 0.95]))


def test_extra_bound_points_validation():
    kwargs = dict(nlive=NLIVE, extra_bound_points=NARROW_CENTER[None, :])
    with pytest.raises(ValueError):
        dynesty.NestedSampler(loglike,
                              prior_transform,
                              NDIM,
                              bound='balls',
                              sample='unif',
                              **kwargs)
    with pytest.raises(ValueError):
        dynesty.NestedSampler(loglike,
                              prior_transform,
                              NDIM,
                              bound='single',
                              sample='unif',
                              extra_bound_method='union',
                              **kwargs)
    with pytest.raises(ValueError):
        dynesty.NestedSampler(loglike,
                              prior_transform,
                              NDIM,
                              bound='multi',
                              sample='unif',
                              nlive=NLIVE,
                              extra_bound_points=[[0.5, 1.5]])
    with pytest.warns(UserWarning, match='extra_bound_points'):
        dynesty.NestedSampler(loglike,
                              prior_transform,
                              NDIM,
                              bound='multi',
                              sample='rslice',
                              **kwargs)


def make_toy_sampler(**kwargs):
    """Static sampler for the toy with the default extra_bound_method and
    one extra point at the narrow peak, the intended use."""
    return dynesty.NestedSampler(loglike,
                                 prior_transform,
                                 NDIM,
                                 nlive=NLIVE,
                                 bound='multi',
                                 sample='unif',
                                 rstate=get_rstate(),
                                 extra_bound_points=NARROW_CENTER[None, :],
                                 **kwargs)


@pytest.mark.slow
def test_extra_bound_points_resume(tmp_path):
    # A run stopped part way, saved and resumed must give exactly the same
    # result as an uninterrupted run, which requires the extra points (used
    # by both the bound and the stopping criterion) to survive the restore.
    fname = str(tmp_path / 'extra_bound_resume.pkl')
    reference = make_toy_sampler()
    reference.run_nested(print_progress=printing, dlogz=0.1)

    interrupted = make_toy_sampler()
    interrupted.run_nested(print_progress=printing,
                           dlogz=0.1,
                           maxiter=1000,
                           add_live=False,
                           checkpoint_file=fname)
    restored = dynesty.NestedSampler.restore(fname)
    npt.assert_array_equal(restored.extra_bound_u, interrupted.extra_bound_u)
    npt.assert_array_equal(restored.extra_bound_logl,
                           interrupted.extra_bound_logl)
    assert restored.extra_bound_method == 'union'
    restored.run_nested(print_progress=printing, dlogz=0.1, resume=True)
    assert restored.results.logz[-1] == reference.results.logz[-1]


@pytest.mark.slow
def test_extra_bound_points_dynamic_resume(tmp_path):
    # The dynamic sampler keeps the extra points for the samplers it creates
    # (base run and batches). A run saved part way through the base run and
    # resumed must give exactly the same result as an uninterrupted run.
    fname = str(tmp_path / 'extra_bound_dynamic.pkl')
    points = NARROW_CENTER[None, :]

    def make_sampler():
        return dynesty.DynamicNestedSampler(loglike,
                                            prior_transform,
                                            NDIM,
                                            nlive=NLIVE,
                                            bound='multi',
                                            sample='unif',
                                            rstate=get_rstate(),
                                            extra_bound_points=points)

    run_kwargs = dict(print_progress=printing, dlogz_init=0.1, maxbatch=2)
    reference = make_sampler()
    reference.run_nested(**run_kwargs)

    interrupted = make_sampler()
    # Step into the base run the same way run_nested() does, then save as
    # a checkpoint would.
    for it, _ in enumerate(interrupted.sample_initial(nlive=NLIVE,
                                                      dlogz=0.1)):
        if it == 1000:
            break
    interrupted.save(fname)
    restored = dynesty.DynamicNestedSampler.restore(fname)
    assert restored.extra_bound_method == 'union'
    npt.assert_array_equal(restored.extra_bound_points[0], points)
    restored.run_nested(resume=True, **run_kwargs)
    assert restored.results.logz[-1] == reference.results.logz[-1]
    assert (abs(narrow_mode_mass(restored.results) - NARROW_MASS_TRUTH) <
            0.1)


def test_extra_bound_points_pool():
    # Extra points evaluated through a dynesty pool get the same
    # log-likelihoods as direct evaluation, and the bound updates use them.
    # one point per peak, plus a far corner that drops below the threshold
    points = np.array([NARROW_CENTER, BROAD_CENTER, [0.05, 0.95]])
    with dypool.Pool(2, loglike, prior_transform) as pool:
        sampler = dynesty.NestedSampler(pool.loglike,
                                        pool.prior_transform,
                                        NDIM,
                                        nlive=NLIVE,
                                        bound='multi',
                                        sample='unif',
                                        pool=pool,
                                        queue_size=10,
                                        rstate=get_rstate(),
                                        extra_bound_points=points)
        npt.assert_allclose(sampler.extra_bound_logl,
                            [loglike(prior_transform(p)) for p in points])
        sampler.run_nested(print_progress=printing, maxiter=3000)
        # see test_pool.py: close explicitly for coverage
        pool.close()
        pool.join()
    assert sampler.nbound > 1
