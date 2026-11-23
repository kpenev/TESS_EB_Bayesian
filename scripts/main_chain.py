"""
The main chain of one target, as post-processing reads it (TEC-142).

Post-processing re-evaluates the posterior the main chain sampled (local
chains, bridge sampling draws), so `MainChain` also provides that function.
Tests on toy posteriors subclass it and override `log_probability`.
"""

from functools import cached_property

from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import LogLikelihood
from mcmc_sampling import get_isochrone_interpolation


class MainChain:
    """
    The window of a target's main chain used by post-processing.

    Attributes:
        tic_id(int):    The target.

        backend(HDFBackend):    The samples file, read only.

        positions(array):    The walker positions after the burn-in
            (thinned), shape (steps, walkers, parameters).

        log_prob(array):    Their stored log-probabilities, shape (steps,
            walkers).
    """

    def __init__(self, config):
        """
        Read the window of the main chain after the burn-in.

        Args:
            config:    The command line configuration (``tic_id``,
                ``samples_fname_pattern``, ``burn_in`` and ``thin``).
        """

        self.tic_id = config.tic_id
        self.backend = HDFBackend(
            config.samples_fname_pattern.format(tic_id=config.tic_id),
            read_only=True,
        )
        self.positions = self.backend.get_chain(
            discard=config.burn_in, thin=config.thin
        )
        self.log_prob = self.backend.get_log_prob(
            discard=config.burn_in, thin=config.thin
        )

    @cached_property
    def log_probability(self):
        """
        The function the main chain sampled, built on first use (slow).

        It is called with one position and returns the log-probability
        followed by the blobs, as `LogLikelihood` does, must be picklable
        (it is evaluated in worker processes), and has an
        ``isochrone_interpolation`` attribute, recorded as provenance.
        """

        return LogLikelihood(
            self.tic_id,
            isochrone_interpolation=get_isochrone_interpolation(
                self.backend.filename
            ),
        )
