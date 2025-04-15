from django.urls import path

from bui.select_ticids.views import TICIdSelectorView
from bui.select_ticids import plots

urlpatterns = [
    path(
        "",
        TICIdSelectorView.as_view(
            reviewing="prsa",
            tablename="prsa_ebs",
            plot=plots.lightcurve,
            rendered_only=True,
        ),
        name="prsa_index",
    ),
    path(
        "<int:displayed_ticid>/",
        TICIdSelectorView.as_view(
            reviewing="prsa",
            tablename="prsa_ebs",
            plot=plots.lightcurve,
            rendered_only=True,
        ),
        name="prsa_jump",
    ),
    path(
        "<int:displayed_ticid>/<slug:decision>/",
        TICIdSelectorView.as_view(
            reviewing="prsa",
            tablename="prsa_ebs",
            plot=plots.lightcurve,
            rendered_only=True,
        ),
        name="prsa_decision",
    ),
] + sum(
    (
        [
            path(
                plot_type,
                TICIdSelectorView.as_view(
                    reviewing=plot_type,
                    tablename="attempted_sampling",
                    plot=getattr(plots, plot_type),
                    rendered_only=False,
                ),
                name=f"{plot_type}_index",
            ),
            path(
                f"{plot_type}/<int:displayed_ticid>/",
                TICIdSelectorView.as_view(
                    reviewing=plot_type,
                    tablename="attempted_sampling",
                    plot=getattr(plots, plot_type),
                    rendered_only=False,
                ),
                name=f"{plot_type}_jump",
            ),
            path(
                f"{plot_type}/<int:displayed_ticid>/<slug:decision>/",
                TICIdSelectorView.as_view(
                    reviewing=plot_type,
                    tablename="attempted_sampling",
                    plot=getattr(plots, plot_type),
                    rendered_only=False,
                ),
                name=f"{plot_type}_decision",
            ),
        ]
        for plot_type in ["starting", "best"]
    ),
    [],
)
