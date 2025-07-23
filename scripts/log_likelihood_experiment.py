"""Experiment with the log likelihood (used for debugging)."""

from argparse import Namespace

import numpy

from log_likelihood import LogLikelihood
from binary import Binary
from paths import samples
from hacked_emcee_hdf5_backend import HDFBackend
from light_curve_plotter import LightCurvePlotter
from sample_params import SampleParams


def experiment():
    """Manually experiment with things."""

    test_tic = 26489741
    log_likelihood = LogLikelihood(test_tic)

    # logging.basicConfig(level=logging.DEBUG)
    backend = HDFBackend(
        samples.format(tic_id=test_tic),
        # name="prelim_mcmc_4",
        read_only=True,
    )
    log_prob = backend.get_log_prob()
    best_index = numpy.unravel_index(numpy.argmax(log_prob), log_prob.shape)
    best_params = SampleParams(*backend.get_blobs()[best_index])
    best_params = best_params._replace(
        eclipse_time=best_params.eclipse_time - 1.0
    )
    best_binary = Binary(from_mcmc=best_params)
    for header, lightcurve in log_likelihood.lcs:
        log_likelihood.get_eclipse_model(best_binary, header, lightcurve, 0.0)

    print(f"Best binary: {best_binary!s}")
    print(
        "Max log-likelihood: "
        + repr(
            log_likelihood.calc_lc_log_likelihood(
                best_binary, best_params.lc_sys
            )
        )
    )
    LightCurvePlotter(
        Namespace(
            tic_id=test_tic,
            plot_lightcurve=[
                "test.pdf",
                "[[full, full],"
                " [folded, folded],"
                " [zoom_primary, zoom_secondary]]",
            ],
            # highlight_first_model=True,
        )
    )(tic_id=test_tic, binaries=[best_binary])


# pylint: enable=import-outside-toplevel


if __name__ == "__main__":
    experiment()
