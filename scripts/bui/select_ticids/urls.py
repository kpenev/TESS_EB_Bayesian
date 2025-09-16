"""Define the URL patterns for the sampling TICId selector."""

from django.urls import path

from bui.select_ticids.views import TICIdSelectorView


def get_review_urls(table_name, states):
    """Return a list of URL patterns."""

    return [
        path(
            f"{table_name}/{urltail}",
            TICIdSelectorView.as_view(
                reviewing=table_name,
                tablename=table_name,
                plot_dirs=(
                    ("best", "1 / 1 / 2 / 2"),
                    ("convergence", "1 / 2 / 2 / 3"),
                    # ("starting", "2 / 2 / 3 / 3"),
                ),
                rendered_only=False,
                states=states,
                grid={"columns": "1fr 1fr", "rows": "1fr"},
            ),
            name=urlname,
        )
        for urltail, urlname in [
            ("", ""),
            ("<slug:sort_state>/", f"{table_name}_index"),
            ("<slug:sort_state>/<int:displayed_ticid>/", f"{table_name}_jump"),
            (
                "<slug:sort_state>/<int:displayed_ticid>/<slug:decision>/",
                f"{table_name}_decision",
            ),
        ]
    ]


urlpatterns = get_review_urls(
    "sampling",
    (
        "finished",
        "continue ls6",
        "continue juno",
        "restart juno",
        "changed likelihood",
        "unsuitable",
    ),
) + get_review_urls(
    "sample_prsa",
    (
        "finished",
        "continue",
        "restart",
        "changed likelihood",
        "unsuitable",
    ),
)
