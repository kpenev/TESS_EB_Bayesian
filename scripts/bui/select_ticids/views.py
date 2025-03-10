"""The available views for the app allowing the selection of TIC IDs."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(__file__)))

# pylint: disable=wrong-import-position
from sqlalchemy import select, update
from django.shortcuts import render
from django.views import View

# False positive
# pylint: disable=import-error
from db_interface import Session

# pylint: enable=import-error

from .data_model import get_ticid_select_table


class TICIdSelectorView(View):
    """Base for views that displays plots per TIC and allows selecting some."""

    tablename = None

    def get(self, request, displayed_ticid=None, decision=None):
        """Allow user to review LCs from Villanova catalog and select some."""

        # That's the whole point
        # pylint: disable=no-member
        SelectTICIDs = get_ticid_select_table(self.tablename)
        # pylint: enable=no-member

        # False positive
        # pylint: disable=no-member
        with Session.begin() as db_session:
            if decision is not None:
                assert displayed_ticid is not None
                if decision == "select":
                    flag = 1
                elif decision == "discard":
                    flag = -1
                else:
                    assert decision == "skip"
                    flag = 0
                if flag:
                    db_session.execute(
                        update(SelectTICIDs)
                        .filter_by(id=displayed_ticid)
                        .values(flag=flag)
                    )
                displayed_ticid = db_session.scalar(
                    select(SelectTICIDs.id)
                    .filter_by(flag=0)
                    .where(SelectTICIDs.id > displayed_ticid)
                    .order_by(SelectTICIDs.id)
                )

            # pylint: enable=no-member
            context = {
                state: db_session.scalars(
                    select(SelectTICIDs.id)
                    .filter_by(flag=flag)
                    .order_by(SelectTICIDs.id)
                ).all()
                for state, flag in [
                    ("selected", "1"),
                    ("pending", "0"),
                    ("discarded", "-1"),
                ]
            }
        if displayed_ticid is None:
            displayed_ticid = context["pending"][0]
        context["displayed_ticid"] = displayed_ticid

        return render(request, "select_ticids/index.html", context)
