"""The available views for the app allowing the selection of TIC IDs."""

import sys
from os import path, remove
from base64 import b64encode

sys.path.append(path.dirname(path.dirname(__file__)))

# pylint: disable=wrong-import-position
from sqlalchemy import select, update, delete, func, and_, or_
from django.shortcuts import render, redirect
from django.views import View
from django.template.defaultfilters import slugify

# False positive
# pylint: disable=import-error
from db_interface import Session

# pylint: enable=import-error
from download_lcs import get_available_sectors
from exclude_data import exclude_data
from cache_interface import CacheSession, CachedBLS

from .data_model import get_ticid_select_tables
from .path_util import get_render_dir
from . import plots


class TICIdSelectorView(View):
    """Base for views that displays plots per TIC and allows selecting some."""

    reviewing = None
    tablename = None
    plot_dirs = ()
    rendered_only = False
    states = ("selected", "discarded")
    grid = {"columns": "1fr", "rows": "1fr"}

    def _get_sector_context(self, ticid):
        """Return the sector info to add to context for given TIC ID."""

        tic_excluded = exclude_data[ticid]
        spoc_sectors = get_available_sectors(ticid, "SPOC")
        qlp_sectors = set(get_available_sectors(ticid, "QLP")) - set(
            spoc_sectors
        )
        enabled_prov = [
            prov for prov in ["SPOC", "QLP"] if prov not in tic_excluded
        ]
        if "OOE" not in tic_excluded:
            enabled_prov.append("OOE")

        return {
            "enabled_prov": enabled_prov,
            "spoc_sectors": [
                (
                    "SPOC" in enabled_prov and sector not in tic_excluded,
                    sector,
                )
                for sector in spoc_sectors
            ],
            "qlp_sectors": [
                (
                    "QLP" in enabled_prov and sector not in tic_excluded,
                    sector,
                )
                for sector in qlp_sectors
            ],
            "ooe_flags": [
                (
                    "OOE" in enabled_prov and "BLSOOE" not in tic_excluded,
                    "BLS",
                ),
            ],
        }

    def _join_rendered(
        self,
        select_stmt,
        SelectTICIDs,  # pylint: disable=invalid-name
        RenderedTable,  # pylint: disable=invalid-name
    ):
        """Join the given select with RenderedTable to match plot type."""

        mode = self.reviewing[  # pylint: disable=unsubscriptable-object
            len(self.tablename) + 1 :
        ]
        if mode == 'sampling':
            match_plot = or_(
                *(
                    RenderedTable.plot == plot_type
                    for plot_type in ['best', 'convergence', 'starting']
                )
            )
        else:
            match_plot = RenderedTable.plot == mode
        return select_stmt.outerjoin(
            RenderedTable,
            and_(
                SelectTICIDs.id == RenderedTable.id,
                match_plot,
            ),
        )

    def _get_context(  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self,
        displayed_ticid,
        sort_state,
        SelectTICIDs,  # pylint: disable=invalid-name
        Rendered,  # pylint: disable=invalid-name
        db_session,
    ):
        """Return the context to render the view with."""

        select_expr = self._join_rendered(
            select(
                SelectTICIDs.id,
                Rendered.id != None,  # pylint: disable=singleton-comparison
            ),
            SelectTICIDs,
            Rendered,
        )
        if self.rendered_only:
            select_expr = select_expr.where(
                Rendered.id != None  # pylint: disable=singleton-comparison
            )

        context = {
            "review_table": self.tablename,
            "sort_state": sort_state,
            "by_state": [
                (
                    state_index,
                    db_session.execute(
                        select_expr.where(
                            SelectTICIDs.status == state
                        ).group_by(SelectTICIDs.id)
                    ).all(),
                )
                for state, state_index in enumerate(("pending",) + self.states)
            ],
            "decisions": self.states + ("skip",),
            "review": self.reviewing,
            "mode": self.reviewing.rsplit("_", 1)[1],
            "grid": self.grid,
            "images": [],
        }
        # pylint: enable=no-member

        with CacheSession.begin() as cache:  # pylint: disable=no-member
            context["needs_replot"] = (
                cache.scalar(
                    select(
                        func.count(  # pylint: disable=not-callable
                            CachedBLS.tic_id
                        )
                    ).filter_by(tic_id=displayed_ticid)
                )
                == 0
            )
        context["displayed_ticid"] = displayed_ticid
        context.update(self._get_sector_context(displayed_ticid))
        for dirname, area in self.plot_dirs:
            plot_fname = path.join(dirname, f"tess{displayed_ticid}.png")
            if path.exists(plot_fname):
                with open(plot_fname, "rb") as plotf:
                    context["images"].append(
                        (
                            dirname,
                            b64encode(plotf.read()).decode("utf-8"),
                            area,
                        )
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

        context["render_progress"] = (
            rendered_progress.get(1, 0),
            rendered_progress.get(0, 0) + rendered_progress.get(1, 0),
        )
        return context

    def get(self, request, **kwargs):
        """Allow user to review LCs from Villanova catalog and select some."""

        print(f"Select TIC ID view with kwargs: {kwargs}")
        if "sort_state" not in kwargs:
            kwargs["sort_state"] = "pending"

        state_slugs = ["pending"] + [slugify(state) for state in self.states]
        print(f"State slugs: {state_slugs!r}")
        # That's the whole point
        # pylint: disable=no-member
        # This is actually a class
        # pylint: disable=invalid-name
        SelectTICIDs, Rendered = get_ticid_select_tables(
            self.tablename,
            tuple(e[0] for e in self.plot_dirs),
            refresh_rendered=kwargs.get("update_rendered", False),
        )
        # pylint: enable=no-member
        # pylint: enable=invalid-name

        # False positive
        # pylint: disable=no-member
        with Session.begin() as db_session:
            if "displayed_ticid" not in kwargs:
                sort_state_index = state_slugs.index(kwargs["sort_state"])
            else:
                sort_state_index = db_session.scalar(
                    select(SelectTICIDs.status).filter_by(
                        id=kwargs["displayed_ticid"]
                    )
                )
                kwargs["sort_state"] = state_slugs[sort_state_index]

            if "decision" in kwargs:
                assert "displayed_ticid" in kwargs
                if kwargs["decision"] == "skip":
                    status = sort_state_index
                else:
                    status = state_slugs.index(kwargs["decision"])
                if status:
                    db_session.execute(
                        update(SelectTICIDs)
                        .filter_by(id=kwargs["displayed_ticid"])
                        .values(status=status)
                    )
            if "displayed_ticid" not in kwargs or "decision" in kwargs:
                select_ticid = (
                    self._join_rendered(
                        select(SelectTICIDs.id), SelectTICIDs, Rendered
                    )
                    .where(SelectTICIDs.status == sort_state_index)
                    .where(
                        Rendered.id  # pylint: disable=singleton-comparison
                        != None
                    )
                )
                if "displayed_ticid" in kwargs:
                    select_ticid = select_ticid.where(
                        SelectTICIDs.id > kwargs["displayed_ticid"]
                    )

                kwargs["displayed_ticid"] = db_session.scalar(
                    select_ticid.order_by(SelectTICIDs.id)
                ) or db_session.scalar(
                    select(SelectTICIDs.id).where(
                        SelectTICIDs.status == sort_state_index
                    )
                )

                print(f"Selected TIC ID: {kwargs['displayed_ticid']}")

            context = self._get_context(
                kwargs["displayed_ticid"],
                kwargs["sort_state"],
                SelectTICIDs,
                Rendered,
                db_session,
            )
        return render(request, "select_ticids/index.html", context)


def toggle_data(_, ticid, selection, review_table, mode):
    """Switch the state (enabled/disabled) for given data for given TIC ID."""

    excluded = exclude_data[ticid]

    if selection == "BLS":
        selection = "BLSOOE"
    elif selection not in ["OOE", "SPOC", "QLP"]:
        selection = int(selection)

    if selection in excluded:
        excluded.remove(selection)
    else:
        excluded.add(selection)

    if mode in ["starting", "best", "convergence", "sampling"]:
        return redirect(
            f"{review_table}_{mode}",
            sort_state="continue",
            decision="changed_likelihood",
            displayed_ticid=ticid,
        )

    if mode == "lightcurve":
        with CacheSession.begin() as cache:  # pylint: disable=no-member
            cache.execute(delete(CachedBLS).filter_by(tic_id=ticid))

    return redirect(
        f"{review_table}_lightcurve_jump",
        sort_state="pending",
        displayed_ticid=ticid,
    )


def replotlc(_, ticid, review_table):
    """Re-generate the lightcurve plot for the given TIC ID."""

    plot_fname = path.join(
        get_render_dir(review_table, "lightcurve"), f"tess{ticid}.png"
    )
    if path.exists(plot_fname):
        remove(plot_fname)

    plots.lightcurve(ticid, plot_fname)
    return redirect(
        f"{review_table}_lightcurve_jump",
        sort_state="pending",
        displayed_ticid=ticid,
    )
