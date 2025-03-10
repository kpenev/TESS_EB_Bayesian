from django.urls import path

from bui.select_ticids.views import TICIdSelectorView
from bui.select_ticids import plots

urlpatterns = [
    path(
        "",
        TICIdSelectorView.as_view(
            tablename="prsa_ebs", plot=plots.lightcurve
        ),
        name="index",
    ),
    path(
        "<int:displayed_ticid>/",
        TICIdSelectorView.as_view(
            tablename="prsa_ebs", plot=plots.lightcurve
        ),
        name="jump",
    ),
    path(
        "<int:displayed_ticid>/<slug:decision>/",
        TICIdSelectorView.as_view(
            tablename="prsa_ebs", plot=plots.lightcurve
        ),
        name="decision",
    ),
]
