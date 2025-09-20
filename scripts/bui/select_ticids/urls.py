"""Define the URL patterns for the sampling TICId selector."""

import os.path

from django.urls import path

from paths import render_dir
from bui.select_ticids.views import TICIdSelectorView


def get_review_urls(mode, table_name, states):
    """Return a list of URL patterns."""

    if mode == "lightcurve":
        plot_dirs = (
            (
                os.path.join(render_dir, table_name, "lightcurve"),
                "1 / 1 / 1 / 1",
            ),
        )
    elif mode == "sampling":
        plot_dirs = (
            (os.path.join(render_dir, table_name, "best"), "1 / 1 / 2 / 2"),
            (
                os.path.join(render_dir, table_name, "convergence"),
                "1 / 2 / 2 / 3",
            ),
            # (
            #    os.path.join(render_dir, table_name, "starting"),
            #    "2 / 2 / 3 / 3"
            # ),
        )
    return [
        path(
            f"{table_name}/{mode}/{urltail}",
            TICIdSelectorView.as_view(
                reviewing=f"{table_name}_{mode}",
                tablename=table_name,
                plot_dirs=plot_dirs,
                rendered_only=False,
                states=states,
                grid={"columns": "1fr 1fr", "rows": "1fr"},
            ),
            name=urlname,
        )
        for urltail, urlname in [
            ("", ""),
            ("<slug:sort_state>/", f"{table_name}_{mode}_index"),
            (
                "<slug:sort_state>/<int:displayed_ticid>/",
                f"{table_name}_{mode}_jump",
            ),
            (
                "<slug:sort_state>/<int:displayed_ticid>/<slug:decision>/",
                f"{table_name}_{mode}_decision",
            ),
        ]
    ]


urlpatterns = get_review_urls(
    "lightcurve",
    "sample_prsa",
    ("bad", "full_model", "discard_ooe", "discard_sectors"),
) + get_review_urls(
    "sampling",
    "sample_prsa",
    ("selected", "running_ls6", "running_juno", "restart", "stop", "finished"),
)
