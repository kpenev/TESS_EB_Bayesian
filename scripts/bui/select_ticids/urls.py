from django.urls import path

from .views import TICIdSelectorView

urlpatterns = [
    path("", TICIdSelectorView.as_view(tablename='test_batch'), name="index"),
    path("<int:displayed_ticid>/",
         TICIdSelectorView.as_view(tablename='test_batch'), name="jump"),
    path("<int:displayed_ticid>/<slug:decision>/",
         TICIdSelectorView.as_view(tablename='test_batch'), name="decision")
]
