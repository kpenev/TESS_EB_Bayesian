"""Define the URL patterns for the sampling TICId selector."""

from django.urls import path

from .views import TICIdSelectorView, toggle_data, replotlc
from .path_util import get_render_dir


def get_review_urls(mode, table_name, states):
    """Return a list of URL patterns."""

    if mode in ["lightcurve", "starting"]:
        plot_dirs = (
            (
                get_render_dir(table_name, mode),
                "1 / 1 / 1 / 1",
            ),
        )
    elif mode == "sampling":
        plot_dirs = (
            (get_render_dir(table_name, "best"), "1 / 1 / 2 / 2"),
            (get_render_dir(table_name, "convergence"), "1 / 2 / 2 / 3"),
            # (get_render_dir(table_name, "starting"), "2 / 2 / 3 / 3"),
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
            kwargs={"update_rendered": urltail.startswith("update_rendered/")},
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
            (
                "update_rendered/<int:displayed_ticid>",
                f"{table_name}_{mode}_update_rendered",
            ),
        ]
    ]


urlpatterns = (
    get_review_urls(
        "lightcurve",
        "sample_prsa",
        (
            "bad",
            "sample",
            "fix",
            "old_finished",
            "old_ls6_sampling",
            "old_juno_sampling",
        ),
    )
    + get_review_urls(
        "starting",
        "sample_prsa",
        (
            "bad",
            "continue",
            "fix",
            "old_finished",
            "old_ls6_sampling",
            "old_juno_sampling",
        ),
    )
    + get_review_urls(
        "sampling",
        "sample_prsa",
        (
            "selected",
            "running_ls6",
            "running_juno",
            "restart",
            "stop",
            "finished",
        ),
    )
    + [
        path(
            "toggle_data/<int:ticid>/<slug:selection>/<slug:review_table>",
            toggle_data,
            name="toggle_data",
        ),
        path(
            "replotlc/<int:ticid>/<slug:review_table>",
            replotlc,
            name="replotlc",
        ),
    ]
)
