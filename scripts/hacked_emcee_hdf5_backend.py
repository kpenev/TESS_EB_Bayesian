# -*- coding: utf-8 -*-

"""
Modified HDF5 EMCEE backend that buffers steps in case saving fails.

Each step is appended to a buffer file (``<samples>.unsaved_steps``) as soon
as it is generated, and the buffered steps are written to the HDF5 file in
batches. The buffer starts with the HDF5 iteration at which its first step
belongs, followed by the pickled steps. It is best-effort: a buffer that is
inconsistent with the HDF5 file is discarded, since the few steps it holds
are quick to redo.

Writes are ordered so that an interrupted write leaves the files consistent:
chain rows are on disk before the iteration counting them, and the buffer is
replaced atomically only after that. Damage the storage may still cause is
repaired when a samples file is opened for writing, where possible (logged as
ERROR), otherwise ``CorruptSamplesError`` is raised. See TEC-145.

The random state of the sampler is not stored, so each resumed run starts
from a fresh one.
"""

from __future__ import division, print_function

__all__ = ["HDFBackend", "TempHDFBackend", "CorruptSamplesError"]

import os
import os.path
import re
from tempfile import NamedTemporaryFile
import pickle
import logging
from time import monotonic

import numpy as np

from emcee import __version__
from emcee.backends import Backend

try:
    import h5py
except ImportError:
    h5py = None

_logger = logging.getLogger(__name__)

# How HDF5 reports a file shorter than the size recorded in it.
_truncated_file_message = re.compile(
    r"truncated file: eof = (\d+).*stored_eof = (\d+)"
)


class CorruptSamplesError(RuntimeError):
    """The samples file is damaged beyond automatic repair."""


def _fsync_path(path):
    """Make sure everything written to a file or directory is on disk."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# The buffer and batching state adds to the attributes inherited from emcee.
# pylint: disable-next=too-many-instance-attributes
class HDFBackend(Backend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note::
        You must install `h5py <http://www.h5py.org/>`_ to use this backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        name (str; optional): The name of the group where the chain will
            be saved.
        read_only (bool; optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.

    """

    @staticmethod
    def _write_steps(steps, group, iteration):
        """
        Write steps to the chain rows starting at the given iteration.

        Does not update the iteration attribute: the caller does that once
        the rows are on disk.

        Args:
            steps(list):    The ``(state, accepted)`` steps to write.

            group(h5py.Group):    The chain group of the open samples file.

            iteration(int):    The row for the first step.

        Returns:
            int:
                The iteration after the last written step.
        """

        end = iteration + len(steps)
        group["chain"][iteration:end] = np.stack(
            [state.coords for state, _ in steps]
        )
        group["log_prob"][iteration:end] = np.stack(
            [state.log_prob for state, _ in steps]
        )
        if steps[0][0].blobs is not None:
            group["blobs"][iteration:end] = np.stack(
                [state.blobs for state, _ in steps]
            )
        group["accepted"][:] += np.sum(
            [accepted for _, accepted in steps], axis=0
        )
        return end

    def _load_buffer(self):
        """
        Return the contents of the buffer, dropping what cannot be used.

        Reading stops at the first unreadable record (normally a step whose
        writing was interrupted); the records before it are kept.

        Returns:
            (int or None, list):
                The iteration of the first buffered step (None if the buffer
                is missing or unusable) and the buffered ``(state, accepted)``
                steps.
        """

        if not os.path.exists(self.unsaved_steps_fname):
            return None, []
        records = []
        with open(self.unsaved_steps_fname, "rb") as buffer_file:
            while True:
                try:
                    records.append(pickle.load(buffer_file))
                except EOFError:
                    break
                # Unpickling damaged data can raise almost anything.
                # pylint: disable=broad-exception-caught
                except Exception:
                    # pylint: enable=broad-exception-caught
                    _logger.error(
                        "Dropping unreadable data after %d records of %s",
                        len(records),
                        repr(self.unsaved_steps_fname),
                        exc_info=True,
                    )
                    break
        if not records:
            return None, []
        if not isinstance(records[0], (int, np.integer)) or not all(
            isinstance(step, tuple) and len(step) == 2 for step in records[1:]
        ):
            _logger.error(
                "Discarding %s: it does not start with an iteration followed "
                "by steps",
                repr(self.unsaved_steps_fname),
            )
            return None, []
        return int(records[0]), records[1:]

    def _start_buffer(self, iteration):
        """Atomically replace the buffer by an empty one at the iteration."""

        directory = os.path.dirname(os.path.abspath(self.unsaved_steps_fname))
        with NamedTemporaryFile(
            dir=directory,
            prefix=os.path.basename(self.unsaved_steps_fname) + ".",
            suffix=".tmp",
            delete=False,
        ) as new_buffer:
            pickle.dump(int(iteration), new_buffer)
            new_buffer.flush()
            os.fsync(new_buffer.fileno())
        os.replace(new_buffer.name, self.unsaved_steps_fname)
        _fsync_path(directory)
        self._steps_since_flush = 0
        self._last_flush = monotonic()

    def flush(self):
        """
        Write the buffered steps to the HDF5 file and start a new buffer.

        The chain rows are made durable before the iteration attribute that
        counts them, and the buffer is replaced only after that, so an
        interruption at any point leaves files that the next flush handles:
        rows written past the iteration are overwritten, and steps already
        counted by the iteration are skipped.
        """

        start, steps = self._load_buffer()
        if not os.path.exists(self.filename):
            if steps:
                _logger.error(
                    "Discarding %d buffered steps: %s does not exist",
                    len(steps),
                    repr(self.filename),
                )
            self._start_buffer(0)
            return
        with self.open("r+" if steps else "r") as samples_file:
            if self.name not in samples_file:
                return
            group = samples_file[self.name]
            saved = int(group.attrs["iteration"])
            if start is None:
                if steps:
                    _logger.error(
                        "Discarding %d buffered steps of unknown iteration",
                        len(steps),
                    )
                steps = []
            elif start > saved:
                _logger.error(
                    "Discarding %d buffered steps from iteration %d: %s has "
                    "only %d, the steps in between were lost",
                    len(steps),
                    start,
                    repr(self.filename),
                    saved,
                )
                steps = []
            else:
                steps = steps[saved - start :]
            if steps:
                end = self._write_steps(steps, group, saved)
                samples_file.flush()
                _fsync_path(self.filename)
                group.attrs["iteration"] = end
                samples_file.flush()
                _fsync_path(self.filename)
                _logger.debug(
                    "Added %d steps to %s", len(steps), repr(self.filename)
                )
                saved = end
        self._start_buffer(saved)

    # Inherited from EMCEE
    # pylint: disable=super-init-not-called
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        *,
        flush_steps=100,
        flush_seconds=600.0,
    ):
        """
        Open (and if writable, check and repair) the given samples file.

        Args:
            flush_steps(int):    Write buffered steps to the HDF5 file after
                this many steps.

            flush_seconds(float):    Write buffered steps to the HDF5 file
                after this many seconds, even if fewer than ``flush_steps``.

        See the class docstring for the other arguments.
        """

        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name
        self.read_only = read_only
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

        self._has_blobs = None
        self._shape = None
        self.flush_steps = flush_steps
        self.flush_seconds = flush_seconds
        self._steps_since_flush = 0
        self._last_flush = monotonic()
        self.unsaved_steps_fname = (
            os.path.splitext(filename)[0] + ".unsaved_steps"
        )
        if not read_only:
            if os.path.exists(filename):
                self._check_and_repair()
            self.flush()

    # pylint: enable=too-many-arguments
    # pylint: enable=super-init-not-called

    def _open_for_check(self):
        """Open the samples file for reading, padding it if truncated."""

        try:
            return h5py.File(self.filename, "r")
        except OSError as error:
            truncated = _truncated_file_message.search(str(error))
            if truncated is None or int(truncated[2]) <= int(truncated[1]):
                raise
            _logger.error(
                "%s is %d bytes shorter than the size recorded in it (the "
                "storage lost a write); padding it with zeros",
                repr(self.filename),
                int(truncated[2]) - int(truncated[1]),
            )
            os.truncate(self.filename, int(truncated[2]))
            _fsync_path(self.filename)
            return h5py.File(self.filename, "r")

    def _check_and_repair(self):
        """
        Repair the damage interrupted writes can leave in the samples file.

        Each repair is logged as ERROR:

            * a file shorter than the size recorded in it is padded with
              zeros;

            * unreadable random state attributes (no longer used) are
              deleted;

            * rows counted by the iteration but never written (all zero) are
              uncounted.

        Raises:
            CorruptSamplesError:    If the file cannot be read for any other
                reason. This may also be a storage outage, so callers should
                retry a few times before giving up on the file.
        """

        try:
            with self._open_for_check() as samples_file:
                if self.name not in samples_file:
                    return
                group = samples_file[self.name]
                bad_attributes = []
                for attribute in list(group.attrs):
                    try:
                        group.attrs[attribute]
                    except OSError:
                        if not attribute.startswith("random_state_"):
                            raise
                        bad_attributes.append(attribute)
                iteration = int(group.attrs["iteration"])
                valid = min(iteration, group["chain"].shape[0])
                while valid > 0 and not group["chain"][valid - 1].any():
                    valid -= 1
            if not bad_attributes and valid == iteration:
                return
            with h5py.File(self.filename, "r+") as samples_file:
                group = samples_file[self.name]
                for attribute in bad_attributes:
                    _logger.error(
                        "Deleting unreadable attribute %s of %s",
                        attribute,
                        repr(self.filename),
                    )
                    del group.attrs[attribute]
                if valid < iteration:
                    _logger.error(
                        "Rolling back %s from iteration %d to %d: the last "
                        "rows were never written",
                        repr(self.filename),
                        iteration,
                        valid,
                    )
                    group.attrs["iteration"] = valid
                samples_file.flush()
            _fsync_path(self.filename)
        except OSError as error:
            _logger.critical(
                "%s is damaged beyond automatic repair",
                repr(self.filename),
                exc_info=True,
            )
            raise CorruptSamplesError(
                f"{self.filename!r} is damaged beyond automatic repair"
            ) from error

    # Inherited from EMCEE
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )
        f = h5py.File(self.filename, mode)
        if not self.dtype_set and self.name in f:
            g = f[self.name]
            if "chain" in g:
                self.dtype = g["chain"].dtype
                self.dtype_set = True
            self._has_blobs = g.attrs["has_blobs"]
        return f

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]

            self._has_blobs = False

            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset("accepted", data=np.zeros(nwalkers))
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=self.dtype,
            )
            g.create_dataset(
                "log_prob",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=self.dtype,
            )
        self._shape = (nwalkers, ndim)
        if os.path.exists(self.unsaved_steps_fname):
            os.remove(self.unsaved_steps_fname)
        self.flush()

    def has_blobs(self):
        if self._has_blobs is None:
            raise RuntimeError("HDF5 backend not ready for use.")

        return self._has_blobs

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            assert g.attrs["has_blobs"] == self._has_blobs
            if name == "blobs" and not self._has_blobs:
                return None

            v = g[name][discard + thin - 1 : self.iteration : thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)
            return v

    @property
    def shape(self):
        # Read once: emcee checks every step against it, and it changes only
        # through reset().
        if self._shape is None:
            with self.open() as f:
                g = f[self.name]
                self._shape = (g.attrs["nwalkers"], g.attrs["ndim"])
        return self._shape

    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def random_state(self):
        """
        Always None: the random state is not stored (TEC-145).

        Files written before still hold it as attributes; it is ignored so
        that resumed runs do not all replay the same stale state. emcee then
        starts from a fresh random state.
        """

        return None

    # pylint: enable=missing-function-docstring

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current list of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        with self.open("r+") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            if blobs is not None:
                assert self._has_blobs == g.attrs["has_blobs"]
                if not self._has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                    g.create_dataset(
                        "blobs",
                        (ntot, nwalkers),
                        maxshape=(None, nwalkers),
                        dtype=dt,
                    )
                else:
                    g["blobs"].resize(ntot, axis=0)
                self._has_blobs = True
                g.attrs["has_blobs"] = True

    # pylint: enable=invalid-name

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """

        self._check(state, accepted)

        with open(self.unsaved_steps_fname, "ab") as buffer_file:
            pickle.dump((state, accepted), buffer_file)
            buffer_file.flush()
            os.fsync(buffer_file.fileno())
        self._steps_since_flush += 1

        if (
            self._steps_since_flush >= self.flush_steps
            or monotonic() - self._last_flush >= self.flush_seconds
        ):
            try:
                self.flush()
            except OSError:
                _logger.error(
                    "Failed to save buffered steps to %s, will try again "
                    "after the next %d steps or %s seconds",
                    repr(self.filename),
                    self.flush_steps,
                    self.flush_seconds,
                    exc_info=True,
                )
                self._steps_since_flush = 0
                self._last_flush = monotonic()


class TempHDFBackend:
    """HDF5 backend based on a temporary file."""

    def __init__(self, dtype=None):
        self.dtype = dtype
        self.filename = None

    def __enter__(self):
        # pylint: disable=invalid-name
        f = NamedTemporaryFile(
            prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
        )
        f.close()
        self.filename = f.name
        return HDFBackend(f.name, "test", dtype=self.dtype)
        # pylint: enable=invalid-name

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
