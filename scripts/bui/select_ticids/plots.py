"""Plots that can be used for selecting TESS objects."""

from collections import namedtuple
from base64 import b64encode
from io import BytesIO

from visualize import create_lightcurve_plot


def lightcurve(tic_id):
    """Plot the lightcurves available for the given TIC."""

    png_stream = BytesIO()

    config = namedtuple("ConfigType", ["tic_id", "plot_lightcurve"])(
        tic_id,
        (
            (png_stream, "png"),
            "[[full]]",
            # "[[full, full, full],"
            # " [folded, folded, folded],"
            # " [zoom_default, zoom_even, zoom_odd]]",
        ),
    )
    create_lightcurve_plot(config)
    return b64encode(png_stream.getvalue()).decode("utf-8")
