#!/usr/bin/env python3

"""Query and assemble stellar evolution data from the CMD web interface."""

from os import path
from tempfile import TemporaryDirectory

import numpy
from configargparse import ArgumentParser, DefaultsFormatter
from astropy import units

from general_purpose_python_modules.cmd_utils import query_cmd


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        formatter_class=DefaultsFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--log-age-grid",
        type=float,
        nargs=3,
        metavar=("min", "max", "step"),
        default=(6.0, 6.13, 0.02),
        help='The grid of log10(age) to generate isochrones for.'
    )
    parser.add_argument(
        "--feh-grid",
        type=float,
        nargs=3,
        metavar=("min", "max", "step"),
        default=(-2.5, -2.0, 0.1),
        help='The grid of [Fe/H] to generate isochrones for.'
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=path.join(
            path.dirname(path.dirname(path.abspath(__file__))),
            "data",
            "isochrone_data.ssv",
        ),
        help='The output file to write the isochrone data to.'
    )
    return parser.parse_args()


def get_isochrones(log_age_grid, feh_grid, temp_dir):
    """Download all the isochrones from the CMD interface to given directory."""

    feh_grid = numpy.arange(*feh_grid)
    feh_slices = []
    for feh in feh_grid:
        feh_slices.append(path.join(temp_dir, f"{feh!r}"))
        query_cmd(
            age=tuple(value * units.yr * units.dex for value in log_age_grid),
            feh=feh,
            cmd_version="3.7",
            output_fname=feh_slices[-1],
        )
    return feh_slices


def main(config):
    """Download and assemble the isochrone file."""

    with TemporaryDirectory() as temp_dir:
        feh_slices = get_isochrones(
            config.log_age_grid, config.feh_grid, temp_dir
        )
        with open(config.output, "w", encoding='utf-8') as destination:
            skip = False
            for feh_slice_fname in feh_slices:
                with open(feh_slice_fname, 'r', encoding='utf-8') as feh_slice:
                    for line in feh_slice:
                        if line.strip() == "#isochrone terminated":
                            pass
                        if skip:
                            if line[0] == "#":
                                header = line
                            else:
                                skip = False
                                destination.write(header)
                                destination.write(line)
                        else:
                            destination.write(line)
                skip = True
            destination.write("#isochrone terminated\n")


if __name__ == "__main__":
    main(parse_command_line())
