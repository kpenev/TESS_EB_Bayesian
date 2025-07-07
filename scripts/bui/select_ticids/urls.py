from django.urls import path

from bui.select_ticids.views import TICIdSelectorView
from bui.select_ticids import plots

urlpatterns = [
    path(
        "sampling/" + urltail,
        TICIdSelectorView.as_view(
            reviewing="sampling",
            tablename="sampling",
            plot_dirs=(
                ("best", "1 / 1 / 3 / 2"),
                ("convergence", "1 / 2 / 2 / 3"),
                ("starting", "2 / 2 / 3 / 3"),
            ),
            rendered_only=True,
            states=("finished", "continue", "bad period", "bad minimum"),
            grid={"columns": "1fr 1fr", "rows": "1fr 1fr"},
        ),
        name=urlname,
    )
    for urltail, urlname in [
        ("", "sampling_index"),
        ("<int:displayed_ticid>/", "sampling_jump"),
        ("<int:displayed_ticid>/<slug:decision>/", "sampling_decision"),
    ]
]
