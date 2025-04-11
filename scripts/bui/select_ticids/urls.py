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
    path(
        "starting",
        TICIdSelectorView.as_view(
            reviewing="starting",
            tablename="attempted_sampling",
            plot=plots.starting,
            rendered_only=False,
        ),
        name="starting_index",
    ),
    path(
        "starting/<int:displayed_ticid>/",
        TICIdSelectorView.as_view(
            reviewing="starting",
            tablename="attempted_sampling",
            plot=plots.starting,
            rendered_only=False,
        ),
        name="starting_jump",
    ),
    path(
        "starting/<int:displayed_ticid>/<slug:decision>/",
        TICIdSelectorView.as_view(
            reviewing="starting",
            tablename="attempted_sampling",
            plot=plots.starting,
            rendered_only=False,
        ),
        name="starting_decision",
    ),

]
