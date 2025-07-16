"""Define the URL patterns for the sampling TICId selector."""

from django.urls import path

from bui.select_ticids.views import TICIdSelectorView

urlpatterns = [
    path(
        "sampling/" + urltail,
        TICIdSelectorView.as_view(
            reviewing="sampling",
            tablename="sampling",
            plot_dirs=(
                ("best", "1 / 1 / 2 / 2"),
                ("convergence", "1 / 2 / 2 / 3"),
                # ("starting", "2 / 2 / 3 / 3"),
            ),
            rendered_only=True,
            states=(
                "finished",
                "continue",
                "bad period",
                "bad minimum",
                "changed likelihood",
            ),
            grid={"columns": "1fr 1fr", "rows": "1fr"},
        ),
        name=urlname,
    )
    for urltail, urlname in [
        ("<slug:sort_state>/", "sampling_index"),
        ("<slug:sort_state>/<int:displayed_ticid>/", "sampling_jump"),
        (
            "<slug:sort_state>/<int:displayed_ticid>/<slug:decision>/",
            "sampling_decision",
        ),
    ]
]
