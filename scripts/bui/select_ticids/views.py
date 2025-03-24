"""The available views for the app allowing the selection of TIC IDs."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(__file__)))

# pylint: disable=wrong-import-position
from sqlalchemy import select, update, func
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
    plot = None
    render_root = "/mnt/md2/TESS_EBs/"

    def get(self, request, displayed_ticid=None, decision=None):
        """Allow user to review LCs from Villanova catalog and select some."""

        # That's the whole point
        # pylint: disable=no-member
        # This is actually a class
        # pylint: disable=invalid-name
        SelectTICIDs = get_ticid_select_table(self.tablename)
        # pylint: enable=no-member
        # pylint: enable=invalid-name

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
            rendered_progress = dict(
                db_session.execute(
                    select(
                        # False positive
                        # pylint: disable=not-callable
                        SelectTICIDs.rendered,
                        func.count(SelectTICIDs.id),
                        # pylint: enable=not-callable
                    ).group_by(SelectTICIDs.rendered)
                ).all()
            )
            if 1 not in rendered_progress:
                rendered_progress[1] = 0

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
        plot_fname = path.join(
            self.render_root, self.tablename, f"tess{displayed_ticid}.png"
        )
        new_plot = not path.exists(plot_fname)
        # False positive
        # pylint: disable=not-callable
        context["image"] = self.plot(displayed_ticid, fname=plot_fname)
        # pylint: enable=not-callable

        if new_plot:
            # False positivie
            # pylint: disable=no-member
            with Session.begin() as db_session:
                # pylint: enable=no-member
                db_session.execute(
                    update(SelectTICIDs)
                    .filter_by(id=displayed_ticid)
                    .values(rendered=1)
                )
                rendered_progress[1] += 1
                rendered_progress[0] -= 1

        context["render_progress"] = (
            rendered_progress[1],
            rendered_progress[0] + rendered_progress[1],
        )

        return render(request, "select_ticids/index.html", context)
