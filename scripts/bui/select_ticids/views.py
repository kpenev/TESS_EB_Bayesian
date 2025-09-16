"""The available views for the app allowing the selection of TIC IDs."""

import sys
from os import path
from base64 import b64encode

sys.path.append(path.dirname(path.dirname(__file__)))

# pylint: disable=wrong-import-position
from sqlalchemy import select, update, func
from django.shortcuts import render
from django.views import View
from django.template.defaultfilters import slugify

# False positive
# pylint: disable=import-error
from db_interface import Session

# pylint: enable=import-error

from paths import render_dir
from .data_model import get_ticid_select_table


class TICIdSelectorView(View):
    """Base for views that displays plots per TIC and allows selecting some."""

    reviewing = None
    tablename = None
    plot_dirs = ()
    rendered_only = False
    states = ("selected", "discarded")
    grid = {"columns": "1fr", "rows": "1fr"}

    def get(
        self, request, sort_state="pending", displayed_ticid=None, decision=None
    ):
        """Allow user to review LCs from Villanova catalog and select some."""

        state_slugs = ["pending"] + [slugify(state) for state in self.states]
        print(f"State slugs: {state_slugs!r}")
        sort_state_index = state_slugs.index(sort_state)
        # That's the whole point
        # pylint: disable=no-member
        # This is actually a class
        # pylint: disable=invalid-name
        SelectTICIDs = get_ticid_select_table(
            self.tablename, tuple(e[0] for e in self.plot_dirs)
        )
        # pylint: enable=no-member
        # pylint: enable=invalid-name

        # False positive
        # pylint: disable=no-member
        with Session.begin() as db_session:
            if decision is not None:
                assert displayed_ticid is not None
                if decision == "skip":
                    status = sort_state_index
                else:
                    status = state_slugs.index(decision)
                if status:
                    db_session.execute(
                        update(SelectTICIDs)
                        .filter_by(id=displayed_ticid)
                        .values(status=status)
                    )
                displayed_ticid = db_session.scalar(
                    select(SelectTICIDs.id)
                    .filter_by(status=sort_state_index, rendered=1)
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

            select_expr = select(SelectTICIDs.id, SelectTICIDs.rendered)
            if self.rendered_only:
                select_expr = select_expr.filter_by(rendered=1)
            context = {
                "sort_state": sort_state,
                "by_state": [
                    (
                        state_index,
                        db_session.execute(
                            select_expr.where(
                                SelectTICIDs.status == state
                            ).order_by(SelectTICIDs.id)
                        ).all(),
                    )
                    for state, state_index in enumerate(
                        ("pending",) + self.states
                    )
                ],
                "decisions": self.states + ("skip",),
                "review": self.reviewing,
                "grid": self.grid,
                "images": [],
            }
            # pylint: enable=no-member

        if displayed_ticid is None:
            displayed_ticid = dict(context["by_state"])[
                (
                    self.states[sort_state_index - 1]
                    if sort_state_index
                    else "pending"
                )
            ][0][0]
        context["displayed_ticid"] = displayed_ticid
        for dirname, area in self.plot_dirs:
            plot_fname = path.join(dirname, f"tess{displayed_ticid}.png")
            if path.exists(plot_fname):
                print(f"Loading {plot_fname}")
                with open(plot_fname, "rb") as plotf:
                    context["images"].append(
                        (
                            dirname,
                            b64encode(plotf.read()).decode("utf-8"),
                            area,
                        )
                    )

        context["render_progress"] = (
            rendered_progress.get(1, 0),
            rendered_progress.get(0, 0) + rendered_progress.get(1, 0),
        )

        return render(request, "select_ticids/index.html", context)
