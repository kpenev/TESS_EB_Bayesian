#!/usr/bin/env python3

"""
Plot the starting models for sampling, one plot per family (mode).

Reads the starting positions that initial_walkers.py saved in a samples file.
Each walker's ``log_prob_results`` row holds its log-probability and the
log-probability and latent coordinates of its family's best point, so walkers
sharing a best point form a family. For each family the plot shows the best
point highlighted (the red model of the BUI "best" plot) and the family's
walkers as the faint models, over the light curve (full, folded, residuals,
zoomed eclipses) and the SED.

The plots are PNG (PDFs with tens of thousands of light curve points are slow
to open), each with a header listing the family's best point, for the BUI's
prepare review page. They are not stacked into one image: with many families
that image gets taller than browsers draw (TIC 27006880, 22 families: 214,639
pixels). Last, a list of the families is written.
"""

from argparse import ArgumentParser, Namespace
import logging
from os import path

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import h5py
import numpy
from emcee import State
from PIL import Image, ImageDraw, ImageFont

from general_purpose_python_modules.emcee_util import load_initial_positions
from binary import Binary
from exclude_data import exclude_data
from hacked_emcee_hdf5_backend import HDFBackend
from log_likelihood import LogLikelihood
from sample_params import SampleParams
from visualize import main as visualize

# pylint: enable=wrong-import-position

_logger = logging.getLogger(__name__)


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tic_id", type=int, help="The TIC ID.")
    parser.add_argument(
        "samples",
        help="The samples file with the starting positions saved by "
        "initial_walkers.py.",
    )
    parser.add_argument(
        "output_prefix",
        help="Family k (best first) is plotted to OUTPUT_PREFIX_k_plot.png "
        "(its temporary samples file is OUTPUT_PREFIX_k_samples.h5), and with "
        "a header listing its best point to OUTPUT_PREFIX_k.png; the list of "
        "families (image, walkers, best log_prob) goes to OUTPUT_PREFIX.txt, "
        "written last.",
    )
    parser.add_argument(
        "--reuse-plots",
        action="store_true",
        help="Do not plot again the families whose OUTPUT_PREFIX_k_plot.png "
        "exists; only redo their headers (seconds instead of minutes per "
        "family).",
    )
    parser.add_argument(
        "--max-walker-models",
        type=int,
        default=50,
        help="Show at most this many of each family's walkers (a random "
        "subset if there are more).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for choosing the walkers shown.",
    )
    return parser.parse_args()


def load_families(samples_fname):
    """
    Return the families of the starting positions in a samples file.

    Every walker is saved with an exact copy of its family's best point, so
    walkers with identical best points form a family.

    Returns:
        [(array, float, array, array)]:
            For each family, best first: its best point, the log-probability
            of that point, its walkers (one per row) and their
            log-probabilities.
    """

    with h5py.File(samples_fname, "r") as samples_file:
        num_walkers = len(samples_file["mcmc/starting_positions/positions"])
    num_params = len(SampleParams._fields)
    positions, log_prob, best, found = load_initial_positions(
        samples_fname,
        num_params=num_params,
        num_walkers=num_walkers,
        blobs_dtype=[("log_likelihood", float)]
        + [(f"s{index:02d}", float) for index in range(num_params)],
    )
    assert found.all(), f"Not all starting positions saved in {samples_fname}"
    best_log_prob = best["log_likelihood"]
    best = numpy.array([list(row)[1:] for row in best])
    families = []
    for center in numpy.unique(best, axis=0):
        members = (best == center).all(axis=1)
        families.append(
            (
                center,
                best_log_prob[members][0],
                positions[members],
                log_prob[members],
            )
        )
    return sorted(families, key=lambda family: -family[1])


def write_samples_file(fname, tic_id, samples, log_prob, log_likelihood):
    """Write the samples as a one-step samples file (one walker each)."""

    physical = numpy.array(
        [log_likelihood.get_sample_params(sample) for sample in samples]
    )
    backend = HDFBackend(fname, name="mcmc")
    backend.reset(len(samples), samples.shape[1])
    backend.grow(1, physical)
    backend.save_step(
        State(
            samples,
            log_prob=log_prob,
            blobs=physical,
            random_state=numpy.random.get_state(),
        ),
        numpy.ones(len(samples), dtype=bool),
    )
    with h5py.File(fname, "a") as samples_file:
        samples_file.attrs["TICID"] = tic_id


def plot_models(tic_id, samples_fname, num_models, output):
    """
    Plot all walkers of a one-step samples file, the first one highlighted.

    Uses the configuration of bui/select_ticids/plots.py:best(), except the
    models: walker 0 is listed first, so it is drawn as the highlighted model
    whatever its log-probability, and the rest are drawn faint.
    """

    visualize(
        Namespace(
            tic_id=tic_id,
            remove_lc_trend="moving_median",
            plot_lightcurve=[
                output,
                "[[full,         full,        zoom_primary],"
                " [folded,       folded,      zoom_secondary],"
                " [folded_diff,  folded_diff, sed]]",
            ],
            show_model_with_lc=[f"0,{walker}" for walker in range(num_models)],
            data_on_top=False,
            samples_fname=samples_fname,
            chain_name="mcmc",
            burn_in=0,
            thin=1,
            sample_condition=None,
            eclipse_model_only="OOE" in exclude_data.get(tic_id, []),
        )
    )


def family_parameters(sample_params):
    """Return the header text with the parameters of a family's best point."""

    return (
        f"age={sample_params.age_gyr:.4g} Gyr, [M/H]={sample_params.meh:.4g}, "
        + Binary(from_mcmc=sample_params).summary()
    )


def _wrap_text(text, font, width):
    """Return the text with line breaks so no line is wider than ``width``."""

    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if line and font.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def add_header(fname, header, output, *, header_size=30, margin=10):
    """
    Write the image in the given file with a header above it.

    Args:
        fname(str):    The image (PNG) to add the header to.
        header(str):    The header text, wrapped to the width of the image.
        output(str):    The file to write the image with the header to.
        header_size(int):    The font size of the header [pixels].
        margin(int):    The space around the header [pixels].
    """

    image = Image.open(fname).convert("RGB")
    font = ImageFont.load_default(size=header_size)
    header = _wrap_text(header, font, image.width - 2 * margin)
    header_height = (
        ImageDraw.Draw(image).multiline_textbbox((0, 0), header, font=font)[3]
        + 2 * margin
    )
    result = Image.new(
        "RGB", (image.width, image.height + header_height), "white"
    )
    ImageDraw.Draw(result).multiline_text(
        (margin, margin), header, fill="black", font=font
    )
    result.paste(image, (0, header_height))
    result.save(output)


def main(config):
    """Make the plot of each family."""

    log_likelihood = LogLikelihood(config.tic_id)
    rng = numpy.random.default_rng(config.seed)
    families = load_families(config.samples)
    family_list = []
    for family, (best, best_log_prob, walkers, log_prob) in enumerate(families):
        shown = numpy.arange(len(walkers))
        if len(walkers) > config.max_walker_models:
            shown = rng.choice(
                len(walkers), config.max_walker_models, replace=False
            )
        prefix = f"{config.output_prefix}_{family:d}"
        if not (config.reuse_plots and path.exists(prefix + "_plot.png")):
            write_samples_file(
                prefix + "_samples.h5",
                config.tic_id,
                numpy.concatenate([best[None, :], walkers[shown]]),
                numpy.concatenate([[best_log_prob], log_prob[shown]]),
                log_likelihood,
            )
            plot_models(
                config.tic_id,
                prefix + "_samples.h5",
                len(shown) + 1,
                prefix + "_plot.png",
            )
        add_header(
            prefix + "_plot.png",
            f"Family {family:d}: {len(walkers):d} walkers, best log_prob "
            f"{best_log_prob:.2f} ({families[0][1] - best_log_prob:.2f} "
            "below family 0)\n"
            + family_parameters(log_likelihood.get_sample_params(best)),
            prefix + ".png",
        )
        family_list.append(
            f"{path.basename(prefix)}.png {len(walkers):d} "
            f"{float(best_log_prob)!r}"
        )
        _logger.info(
            "Family %d: best point log_prob %s, %d of its %d walkers shown "
            "(log_prob %s to %s), plotted to %s.png",
            family,
            repr(best_log_prob),
            len(shown),
            len(walkers),
            repr(log_prob.min()),
            repr(log_prob.max()),
            prefix,
        )
    # Written last: its existence means all family plots are done.
    with open(config.output_prefix + ".txt", "w", encoding="ascii") as listf:
        listf.write("# image walkers best_log_prob\n")
        listf.write("\n".join(family_list) + "\n")
    _logger.info("Families listed in %s.txt", config.output_prefix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_command_line())
