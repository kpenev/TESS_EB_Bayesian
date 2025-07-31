"""Add visualizations for the functioning of class:`LogLikelihood`."""

from matplotlib import pyplot
import numpy
from configargparse import ArgumentParser, DefaultsFormatter

from log_likelihood import LogLikelihood
from paths import samples as samples_fname_pattern
from hacked_emcee_hdf5_backend import HDFBackend
from sample_params import SampleParams
from binary import Binary

class LogLikelihoodVisualize(LogLikelihood):
    """Add plotting to show some inner workings of class:`LogLikelihood`."""

    def _add_eclipse_flags(self, *args):
        """Add plotting to the `_add_eclipse_flags` method."""

        super()._add_eclipse_flags(*args)

        for header, lightcurve in self._lcs:
            porb = header["bls_period"]
            for eclipse_flag in numpy.unique(lightcurve["eclipse_flags"]):
                pyplot.plot(
                    lightcurve["time"][
                        lightcurve["eclipse_flags"] == eclipse_flag
                    ],
                    lightcurve["flux"][
                        lightcurve["eclipse_flags"] == eclipse_flag
                    ],
                    "o"
                    + (
                        "r"
                        if eclipse_flag < 0
                        else ("k" if eclipse_flag == 0 else "b")
                    ),
                    zorder=abs(eclipse_flag),
                )
            pyplot.show()

            for eclipse_flag in numpy.unique(lightcurve["eclipse_flags"]):
                if eclipse_flag == 0:
                    continue
                pyplot.plot(
                    lightcurve["time"][
                        lightcurve["eclipse_flags"] == eclipse_flag
                    ]
                    % porb
                    / porb,
                    lightcurve["flux"][
                        lightcurve["eclipse_flags"] == eclipse_flag
                    ],
                    "o",
                    zorder=abs(eclipse_flag),
                )
                pyplot.show()

    def get_eclipse_model(self, binary, header, lightcurve, lc_sys_err):
        """Add plotting to the `get_eclipse_model` method."""

        try:
            result = super().get_eclipse_model(
                binary, header, lightcurve, lc_sys_err
            )
        except ValueError:
            eclipse_model = self._evaluate_eclipse_model(
                binary, header, lightcurve, False
            )[0]
            pyplot.plot(lightcurve["time"], lightcurve["flux"], "o")
            pyplot.plot(lightcurve["time"], eclipse_model, "-k")
            pyplot.show()
            raise

        eclipse_model, _, model_mask = result
        for eclipse_flag in numpy.unique(lightcurve["eclipse_flags"]):
            pyplot.plot(
                lightcurve["time"][lightcurve["eclipse_flags"] == eclipse_flag],
                lightcurve["flux"][lightcurve["eclipse_flags"] == eclipse_flag],
                "o",
                zorder=abs(eclipse_flag),
            )
        pyplot.show()

        for eclipse_flag in numpy.unique(lightcurve["eclipse_flags"]):
            if eclipse_flag == 0:
                continue
            eclipse_mask = lightcurve["eclipse_flags"] == eclipse_flag
            pyplot.plot(
                lightcurve["time"][eclipse_mask]
                % header["bls_period"]
                / header["bls_period"],
                lightcurve["flux"][eclipse_mask],
                "o",
                zorder=10,
            )
            if eclipse_model.size > 0:
                pyplot.plot(
                    lightcurve["time"][eclipse_mask]
                    % header["bls_period"]
                    / header["bls_period"],
                    eclipse_model[eclipse_mask[model_mask]],
                    "-k",
                    zorder=20,
                )
            pyplot.show()

        return result


def parse_command_line():
    """Parse command line arguments."""

    parser = ArgumentParser(
        description="Visualize the log likelihood calculation of best model for"
        " given TIC Id."
    )
    parser.add_argument(
        "tic_id",
        type=int,
        help="The TIC Id to show likelihood calculation for.",
    )
    parser.add_argument(
        "--samples-fname-pattern",
        default=samples_fname_pattern,
        help="The filename where to save samples. If the file already exists, "
        "sampling continues, adding more points to the existing chain.",
    )
    parser.add_argument(
        "--chain-name",
        default="mcmc",
        help="The name of the HDF5 group containin the MCMC chain to "
        "visualize.",
    )

    return parser.parse_args()


def main(config):
    """Show the calculation of the max likelihood point in given samples."""

    samples_fname = config.samples_fname_pattern.format(tic_id=config.tic_id)
    log_likelihood = LogLikelihoodVisualize(config.tic_id)
    backend = HDFBackend(samples_fname, name=config.chain_name, read_only=True)
    log_prob = backend.get_log_prob()
    best_index = numpy.argmax(log_prob)
    best_index = numpy.unravel_index(best_index, log_prob.shape)
    best_sample = backend.get_blobs()[best_index]
    best_sample = SampleParams(*best_sample)
    binary = Binary(from_mcmc=best_sample)
    for header, lightcurve in log_likelihood._lcs:
        log_likelihood.get_eclipse_model(binary, header, lightcurve, 0.0)

if __name__ == "__main__":
    main(parse_command_line())
