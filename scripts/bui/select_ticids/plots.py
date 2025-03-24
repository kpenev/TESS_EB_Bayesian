"""Plots that can be used for selecting TESS objects."""

from collections import namedtuple
from base64 import b64encode
from io import BytesIO
from os import path, makedirs
from functools import partial

from multiprocessing import Pool
import matplotlib
from matplotlib.pyplot import savefig
from sqlalchemy import select, update

from light_curve_plotter import LightCurvePlotter

# False positive
# pylint: disable=import-error
from bui.db_interface import Session

# pylint: enable=import-error
from bui.select_ticids.data_model import get_ticid_select_table

matplotlib.use("Agg")


def lightcurve(tic_id, fname=None):
    """Plot the lightcurves available for the given TIC."""

    # False positive
    # pylint: disable=possibly-used-before-assignment
    if fname is None:
        png_stream = BytesIO()
        destination = (png_stream, "png")
    else:
        destination = fname
        if not path.exists(path.dirname(fname)):
            makedirs(path.dirname(fname))
    # pylint: enable=possibly-used-before-assignment
    if fname is None or not path.exists(fname):
        # pylint: enable=possibly-used-before-assignment

        config = namedtuple("ConfigType", ["plot_lightcurve", "data_on_top"])(
            (
                destination,
                "[[full, full],"
                " [folded, folded],"
                " [zoom_odd, zoom_even],"
                " [sed, zoom_masked]]",
            ),
            False,
        )
        LightCurvePlotter(config)(tic_id)
        if fname is not None:
            savefig(fname)

    if fname is None:
        return b64encode(png_stream.getvalue()).decode("utf-8")

    with open(fname, "rb") as f:
        return b64encode(f.read()).decode("utf-8")


def render_one(tic_id, tablename, render_dir):
    """Render the lightcurve for a single TIC ID."""

    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(tablename)
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    try:
        lightcurve(tic_id, path.join(render_dir, f"tess{tic_id}.png"))
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            db_session.execute(
                update(SelectTICIDs).filter_by(id=tic_id).values(rendered=1)
            )
    except:
        pass


def render_all_plots(tablename, render_dir, num_parallel):
    """Render the lightcurves plots for a list of TIC IDs for faster review."""

    # That's the whole point
    # pylint: disable=no-member
    # This is actually a class
    # pylint: disable=invalid-name
    SelectTICIDs = get_ticid_select_table(tablename)
    # pylint: enable=no-member
    # pylint: enable=invalid-name

    # False positive
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        tic_id_list = list(
            db_session.execute(select(SelectTICIDs.id)).scalars()
        )

    with Pool(num_parallel) as pool:
        pool.map(
            partial(render_one, tablename=tablename, render_dir=render_dir),
            tic_id_list,
        )


if __name__ == "__main__":
    render_all_plots("prsa_ebs", "/mnt/md2/TESS_EBs/prsa_ebs", 16)
