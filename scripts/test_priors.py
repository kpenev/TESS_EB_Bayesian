#!/usr/bin/env python3
"""
Check that sampling with the likelihood switched off reproduces the priors.

`LogLikelihoodPriorsOnly` drops the light-curve and SED terms, so emcee should
draw every physical parameter from its intended prior:

- uniform on its range, or log-uniform for `LogLikelihood._log_uniform`;
- the age uniform in age (not log age) over its range;
- [M/H] a normal of width 0.5 truncated to its range;
- the Kipping (2013) limb darkening q1, q2 uniform on (0, 1);
- e uniform on (0, e_max) and w uniform on (-180, 180), independent of e.

Each is compared with its expected distribution by a Kolmogorov-Smirnov test
on the chain, thinned by the autocorrelation time. Exits with status 1 if any
test fails.
"""

from argparse import ArgumentParser
from multiprocessing import get_context
import logging
import sys

import emcee
import numpy
from scipy.stats import kstest, truncnorm, uniform

from log_likelihood_priors import LogLikelihoodPriorsOnly
from sample_params import SampleParams

_target = None


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "tic_id",
        type=int,
        nargs="?",
        default=306272819,
        help="The target whose prior ranges are used.",
    )
    parser.add_argument("--num-walkers", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument(
        "--burn-in-fraction",
        type=float,
        default=0.2,
        help="Fraction of the steps discarded at the start.",
    )
    parser.add_argument("--num-parallel", type=int, default=16)
    parser.add_argument(
        "--min-p-value",
        type=float,
        default=1e-3,
        help="A test fails if its p-value is below this.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _log_prob(sample):
    """The prior-only log-probability and physical parameters (worker)."""

    return _target(sample)


def get_expected_cdfs(target):
    """Return {name: (values getter, expected CDF)} for every test."""

    def uniform_cdf(low, high):
        return uniform(loc=low, scale=high - low).cdf

    def field(name, transform=None):
        index = SampleParams._fields.index(name)
        if transform is None:
            return lambda params: params[:, index]
        return lambda params: transform(params[:, index])

    def limb_darkening_q(component, which):
        def get(params):
            return numpy.array(
                [
                    getattr(
                        target.limb_darkening_to_q(SampleParams(*row)),
                        f"{component}_limb_dark_{which}",
                    )
                    for row in params
                ]
            )

        return get

    result = {}
    for name in SampleParams._fields:
        low, high = target.get_range(name)
        if name == "age_gyr":
            result[name] = (field(name), uniform_cdf(10.0**low, 10.0**high))
        elif name == "meh":
            result[name] = (
                field(name),
                truncnorm(2.0 * low, 2.0 * high, scale=0.5).cdf,
            )
        elif "limb_dark" in name:
            component, which = name.split("_limb_dark_")
            result[f"q{which} ({component})"] = (
                limb_darkening_q(component, which),
                uniform_cdf(0.0, 1.0),
            )
        elif name in target._log_uniform:  # pylint: disable=protected-access
            result[f"log10 {name}"] = (
                field(name, numpy.log10),
                uniform_cdf(low, high),
            )
        else:
            result[name] = (field(name), uniform_cdf(low, high))
    return result


def check_samples(params, target, min_p_value):
    """Print the test results for the given physical samples; return passed."""

    passed = True

    def report(name, values, cdf):
        nonlocal passed
        result = kstest(values, cdf)
        ok = result.pvalue >= min_p_value
        passed &= ok
        print(
            f"{name:>30s}: D = {result.statistic:.4f}, "
            f"p = {result.pvalue:.3g} {'ok' if ok else 'FAILED'}"
        )

    for name, (get_values, cdf) in get_expected_cdfs(target).items():
        report(name, get_values(params), cdf)

    ecc = params[:, SampleParams._fields.index("ecc")]
    w_values = params[:, SampleParams._fields.index("w")]
    median = numpy.median(ecc)
    for label, selected in (
        ("w for e < median", ecc < median),
        ("w for e > median", ecc > median),
    ):
        report(label, w_values[selected], uniform(loc=-180, scale=360).cdf)
    return passed


def main(config):
    """Sample the priors and test them."""

    global _target  # pylint: disable=global-statement
    _target = LogLikelihoodPriorsOnly(config.tic_id)
    num_params = len(SampleParams._fields)
    start = numpy.random.default_rng(config.seed).standard_normal(
        (config.num_walkers, num_params)
    )
    with get_context("fork").Pool(config.num_parallel) as pool:
        sampler = emcee.EnsembleSampler(
            config.num_walkers, num_params, _log_prob, pool=pool
        )
        sampler.run_mcmc(start, config.num_steps, progress=False)

    discard = int(config.burn_in_fraction * config.num_steps)
    autocorr = sampler.get_autocorr_time(discard=discard, quiet=True)
    thin = int(numpy.ceil(autocorr.max()))
    params = sampler.get_blobs(discard=discard, thin=thin, flat=True)
    print(
        f"Autocorrelation time up to {autocorr.max():.1f} steps; "
        f"{params.shape[0]} samples after discarding {discard} steps and "
        f"thinning by {thin}."
    )
    return check_samples(params, _target, config.min_p_value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(0 if main(parse_command_line()) else 1)
