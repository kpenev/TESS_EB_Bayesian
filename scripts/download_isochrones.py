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
        default=(6.0, 10.13, 0.02),
        help="The grid of log10(age) to generate isochrones for.",
    )
    parser.add_argument(
        "--feh-grid",
        type=float,
        nargs=3,
        metavar=("min", "max", "step"),
        default=(-2.1, 1.0, 0.1),
        help="The grid of [Fe/H] to generate isochrones for.",
    )
    parser.add_argument(
        "--photometric-system",
        "--photsys",
        default="panstarrs1",
        choices=[
            "2mass_spitzer",
            "2mass_spitzer_wise",
            "2mass",
            "ogle_2mass_spitzer",
            "ubvrijhk",
            "bessel",
            "akari",
            "batc",
            "megacam_wircam",
            "wircam",
            "megacam_post2014",
            "megacam",
            "ciber",
            "clue_galex",
            "CSST",
            "decam",
            "denis",
            "dmc14",
            "dmc15",
            "eis",
            "wfi",
            "wfi2",
            "euclid_nisp",
            "galex_sloan",
            "galex",
            "gaia_tycho2_2mass",
            "gaiaDR2_tycho2_2mass",
            "gaiaDRweiler_tycho2_2mass",
            "gaiaEDR3",
            "gaia",
            "gaiaDR2",
            "gaiaDR2maiz",
            "gaiaDR2weiler",
            "UVbright",
            "acs_hrc",
            "acs_wfc_pos04jul06",
            "acs_wfc_202101",
            "nicmosab",
            "nicmosvega",
            "stis",
            "wfc3_wideverywide",
            "wfc3_202101_verywide",
            "wfc3_202101_medium",
            "wfc3_202101_wide",
            "wfpc2",
            "hipparcos",
            "int_wfc",
            "iphas",
            "jwst_miri_wide",
            "jwst_nircam_wide",
            "jwst_nircam_widemedium_nov22",
            "jwst_nircam_widemedium",
            "jwst_niriss_nov22",
            "jwst_nirspec",
            "kepler",
            "kepler_2mass",
            "vst_vista",
            "lbt_lbc",
            "lsst_wfirst_proposed2017",
            "lsst",
            "lsstDP0",
            "lsstR1.9",
            "noao_ctio_mosaic2",
            "ogle",
            "panstarrs1",
            "Roman2021",
            "splus",
            "sloan",
            "sloan_2mass",
            "sloan_ukidss",
            "swift_uvot",
            "skymapper",
            "spitzer",
            "stroemgren",
            "hsc",
            "suprimecam",
            "SuperBIT",
            "TESS_2mass",
            "TESS_2mass_kepler",
            "ukidss",
            "uvit",
            "visir",
            "vispa",
            "vphas",
            "vst_omegacam",
            "vilnius",
            "wfc3_uvisCaHK",
            "washington_ddo51",
            "ztf",
            "deltaa",
        ],
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=path.join(
            path.dirname(path.dirname(path.abspath(__file__))),
            "data",
            "isochrone_data_{photsys}.ssv",
        ),
        help="The output file to write the isochrone data to.",
    )
    return parser.parse_args()


def get_isochrones(log_age_grid, feh_grid, photsys, temp_dir):
    """Download all the isochrones from the CMD interface to given directory."""

    feh_grid = numpy.arange(*feh_grid)
    feh_slices = []
    for feh in feh_grid:
        feh_slices.append(path.join(temp_dir, f"{feh!r}"))
        query_cmd(
            age=tuple(value * units.yr * units.dex for value in log_age_grid),
            feh=feh,
            cmd_version="3.7",
            photsys=photsys,
            output_fname=feh_slices[-1],
        )
    return feh_slices


def main(config):
    """Download and assemble the isochrone file."""

    with TemporaryDirectory() as temp_dir:
        feh_slices = get_isochrones(
            config.log_age_grid,
            config.feh_grid,
            config.photometric_system,
            temp_dir,
        )
        with open(
            config.output.format(photsys=config.photometric_system),
            "w",
            encoding="utf-8",
        ) as destination:
            skip = False
            for feh_slice_fname in feh_slices:
                with open(feh_slice_fname, "r", encoding="utf-8") as feh_slice:
                    for line in feh_slice:
                        if line.strip() == "#isochrone terminated":
                            continue
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
