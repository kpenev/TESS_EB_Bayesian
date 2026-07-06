"""MCMC chain analysis helpers (non-plotting).

Functions extracted from ``visualize.py`` that operate on MCMC chain
arrays / config objects without touching matplotlib. ``visualize.py``
imports them back to drive its plots; non-plotting callers (sampler,
diagnostics scripts) can use them without pulling in plotting deps.

``get_convergence_data`` has a side effect: when the chain has
progressed past burn-in for every parameter, it clears the
``skip_review`` flag for the current TIC in every BUI review table.
See ``bui.select_ticids.data_model.clear_skip_review``.
"""

import numpy
import pandas
from asteval import Interpreter

from general_purpose_python_modules.emcee_util import load_initial_positions
from general_purpose_python_modules.emcee_quantile_convergence import (
    find_emcee_quantiles,
    diagnose_emcee_quantile,
)
from general_purpose_python_modules.multi_pickle import MultiPickle

from sample_params import SampleParams
from utils import fit_least_squares
from bui.select_ticids.data_model import clear_skip_review
from hacked_emcee_hdf5_backend import HDFBackend


def get_max_likelihood_params(samples_fname):
    """Return ``(SampleParams, log_prob)`` for the max-likelihood mcmc sample.

    Only the ``mcmc`` chain is read; preliminary chains may use a different
    log-probability definition and must be ignored. The single highest
    ``log_prob`` entry is located, then ``thin`` is used to stride straight to
    that step so only the one winning blob row is read back (never the whole
    blobs array). The returned ``log_prob`` is that maximum value.
    """

    backend = HDFBackend(samples_fname, name="mcmc", read_only=True)
    log_prob = backend.get_log_prob()
    top_index = numpy.unravel_index(
        numpy.nanargmax(log_prob), log_prob.shape
    )
    best_params = SampleParams(
        *backend.get_blobs(discard=0, thin=top_index[0] + 1)[0][
            top_index[1:]
        ]
    )
    return best_params, float(log_prob[top_index])


def get_pickler(filename):
    """Return a MultiPickle instance for the given filename."""

    return MultiPickle(
        filename,
        (
            "samples_fname_pattern",
            "samples_fname",
            "corner_plot_fname",
            "plot_expressions",
            "expression_movie",
            "histogram_movie",
            "plot_lightcurve",
            "show_model_with_lc",
            "show_lc_detrending",
            "eclipse_model_only",
            "plot_convergence",
            "sample_condition",
            "x_range",
            "y_range",
            "histogram_resolution",
            "histogram_range",
            "data_on_top",
        ),
    )


def get_chain_expressions(plot_data, chain_expressions):
    """Evaluate the chain expressions specified on the command line."""

    if "selected" in plot_data:
        plot_data = plot_data[plot_data["selected"]]
    if chain_expressions:
        evaluate = Interpreter(user_symbols=plot_data)
        plot_data = {}
        ranges = {}
        for expression in chain_expressions:
            if ":" in expression:
                value_expression, range_expression = expression.split(":", 1)
                plot_range = tuple(
                    evaluate(v) for v in range_expression.split(":")
                )
            else:
                value_expression = expression
                plot_range = 1.0
            name, value_expression = value_expression.split("=")
            plot_data[name] = evaluate(value_expression)
            ranges[name] = plot_range
    for value in plot_data.values():
        if numpy.atleast_1d(value).size > 1:
            print(f"Plot data: {plot_data!r}")
            plot_data = pandas.DataFrame(plot_data)
            ranges = [ranges[col] for col in plot_data.columns]
            return plot_data, ranges
    return plot_data


def get_convergence_data(plot_data, config, num_walkers):
    """Prepare the data needed for the convergence plot.

    Side effect: when the chain's per-parameter burn-in is below the
    total number of steps, calls ``clear_skip_review(config.tic_id)``
    so a previously-skipped TIC is restored to the review queue once
    its sampling has converged.
    """

    num_steps = plot_data.shape[0] // num_walkers
    assert num_walkers * num_steps == plot_data.shape[0]

    if getattr(config, "check_pickled", False):
        pickler = get_pickler("convergence_data.pickle")
        config.num_steps = num_steps
        pickled = pickler.check_for_pickled(config)
        if pickled is not None:
            return pickled[0]

    num_entries = len(plot_data.columns) * len(config.diagnostic_quantiles)
    print(f"Initializing convergence data with {num_entries} entries")
    convergence_data = {
        "value": numpy.empty(num_entries, dtype=float),
        "burnin": numpy.empty(num_entries, dtype=int),
        "stdev": numpy.empty(num_entries, dtype=float),
        "thin": numpy.empty(num_entries, dtype=int),
        "num_steps": num_steps,
    }
    burnin = 0
    result_ind = 0
    for column in plot_data.columns:
        for cdf_value in config.diagnostic_quantiles:
            print(f"Processing quantile {cdf_value} for column {column}")
            convergence_data["burnin"][result_ind] = find_emcee_quantiles(
                plot_data[column].values.reshape(num_steps, num_walkers),
                cdf_value,
                config.burnin_tolerance,
                0,
                max(1, num_steps // 10),
            )[1]
            result_ind += 1
    burnin = convergence_data["burnin"].max()
    if burnin < num_steps:
        clear_skip_review(config.tic_id)
    print(f"Burnin is {burnin} out of {num_steps} steps.")
    result_ind = 0
    for column in plot_data.columns:
        for cdf_value in config.diagnostic_quantiles:
            if burnin < num_steps:
                samples = plot_data[column].values.reshape(
                    num_steps, num_walkers
                )[burnin:]
            else:
                samples = plot_data[column].values.reshape(
                    num_steps, num_walkers
                )
            print(f"Finding quantile of {column} from {samples.size} samples.")
            quantile = numpy.quantile(samples.flatten(), cdf_value)
            convergence_data["value"][result_ind] = quantile
            print(
                f"Diagnosing {column} CDF({quantile}) = {cdf_value} quantile."
            )
            quantile_info = diagnose_emcee_quantile(
                samples,
                samples.shape[1],
                config.quantile_variance_realizations,
                quantile=quantile,
            )
            if quantile_info[2] is None:
                (
                    convergence_data["stdev"][result_ind],
                    convergence_data["thin"][result_ind],
                ) = (numpy.nan, 1)
            else:
                (
                    convergence_data["stdev"][result_ind],
                    convergence_data["thin"][result_ind],
                ) = quantile_info[1:]
            result_ind += 1

    if getattr(config, "check_pickled", False):
        pickler.add_result(config, convergence_data)

    return convergence_data


def get_lstsq_interpreters(lstsq_data):
    """Return interpreters for least-squares and max-likelihood samples."""

    return {
        "lstsq": Interpreter(
            {
                "logprob": lstsq_data["lstsq_logprob"],
                **lstsq_data["lstsq_params"]._asdict(),
            }
        ),
        "maxlike": Interpreter(
            {
                "logprob": lstsq_data["best_logprob"],
                **lstsq_data["best_params"]._asdict(),
            }
        ),
    }


def get_initial_positions(config, num_walkers):
    """Return the initial positions per the given config."""

    print(
        f"Reading initial positions from {config.samples_fname} with args:\n\t"
        + "\n\t".join(
            [
                f"chain_name={config.chain_name}",
                f"num_walkers={num_walkers}",
                f"num_params={len(SampleParams._fields)}",
                "blobs_dtype="
                + repr(
                    [("log_likelihood", float)]
                    + [
                        (f"s{i:02d}", float)
                        for i, _ in enumerate(SampleParams._fields)
                    ]
                ),
            ]
        )
    )

    result = load_initial_positions(
        config.samples_fname,
        chain_name=config.chain_name,
        num_walkers=num_walkers,
        num_params=len(SampleParams._fields),
        blobs_dtype=[("log_likelihood", float)]
        + [(f"s{i:02d}", float) for i, _ in enumerate(SampleParams._fields)],
    )
    return result[0][result[-1]]


def get_walker_step_params(
    raw_data, selection, config, log_likelihood, num_walkers
):
    """Return parameters for <STEP>,<WALKER> specifications."""

    selection = tuple(int(s) for s in selection.split(","))
    if selection[0] == -1:
        initial_positions = get_initial_positions(config, num_walkers)
        if len(selection) != 1:
            initial_positions = [initial_positions[selection[1]]]
        print(
            f"Filtering initial positions:\n{initial_positions!r} with "
            f"condition: {getattr(config, 'sample_condition', 'True')!r}"
        )
        sample_params = [
            log_likelihood.get_sample_params(sample)
            for sample in initial_positions
            if Interpreter(
                user_symbols=(
                    dict(zip(SampleParams._fields, sample))
                    | {
                        "bls_porb": log_likelihood.bls_porb,
                        "bls_duration": log_likelihood.best_fit_bls["duration"],
                    }
                )
            )(getattr(config, "sample_condition", "True"))
        ]
        print(f"Surviving params: {sample_params!r}")
    else:
        if len(selection) == 1:
            if include is not None:
                include = include[
                    selection[0]
                    * num_walkers : (selection[0] + 1)
                    * num_walkers
                ]
                selection = raw_data[selection][include]
            else:
                selection = raw_data[selection]
        else:
            assert selection[0] >= 0
            selection = [raw_data[selection]]

        sample_params = [SampleParams(*sample) for sample in selection]

    return sample_params


def get_lstsq(backend, log_likelihood, config):
    """Max likelihood mcmc sample, parameters, log prob and LSQ fit versions."""

    pickler = MultiPickle("lstsq_data.pickle")
    pickler_config = {
        "samples_fname": backend.filename,
        "num_steps": backend.iteration,
    }
    pickled = pickler.check_for_pickled(pickler_config)
    if pickled is not None:
        return pickled[0]

    log_prob = backend.get_log_prob(discard=config.burn_in, thin=config.thin)
    best_index = numpy.unravel_index(numpy.nanargmax(log_prob), log_prob.shape)
    best_mcmc = backend.get_chain(discard=config.burn_in, thin=config.thin)[
        best_index
    ]
    best_params = backend.get_blobs(discard=config.burn_in, thin=config.thin)[
        best_index
    ]
    lstsq_result = fit_least_squares(log_likelihood, best_mcmc)
    print(f"Least squares fit result: {lstsq_result!r}")

    min_log_prob = log_prob[numpy.isfinite(log_prob)].min()

    result = {
        "best_mcmc": best_mcmc,
        "best_params": SampleParams(*best_params),
        "best_logprob": log_prob[best_index] - min_log_prob,
        "lstsq_mcmc": lstsq_result.x,
        "lstsq_params": log_likelihood.get_sample_params(lstsq_result.x),
        "lstsq_logprob": log_likelihood(lstsq_result.x)[0] - min_log_prob,
    }
    print(
        "LSTSQ result: "
        + "\n\t".join([f"{key}: {value!r}" for key, value in result.items()])
    )
    pickler.add_result(pickler_config, result)
    return result
