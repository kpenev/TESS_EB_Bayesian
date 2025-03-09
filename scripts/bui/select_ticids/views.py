"""The available views for the app allowing the selection of TIC IDs."""

from django.shortcuts import render

# Create your views here.
def index(request):
    """Allow user to review LCs from Villanova catalog and select some."""

    return render(
        request, "select_ticids/index.html", {"review_ticids": range(100)}
    )
