from django.urls import path

from .views import TICIdSelectorView

urlpatterns = [
    path("", TICIdSelectorView.as_view(tablename='test_batch'), name="index"),
]
