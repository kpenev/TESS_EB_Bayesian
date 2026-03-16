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
from cache_interface import CacheSession, CachedBLS, CachedSED

from .data_model import get_ticid_select_tables
from .path_util import get_render_dir
from . import plots


_SAMPLING_PLOT_MODES = frozenset(
    {"starting", "best", "convergence", "sampling"}
)


class TICIdSelectorView(View):
    """Base for views that displays plots per TIC and allows selecting some."""

    reviewing = None
    tablename = None
    plot_dirs = ()
    rendered_only = False
    states = [(1, "selected"), (2, "discarded")]
    grid = {"columns": "1fr", "rows": "1fr"}

    def _get_sector_context(self, ticid, pending_toggles=()):
        """Return the sector info to add to context for given TIC ID."""

        tic_excluded = set(exclude_data[ticid])
        for sel in pending_toggles:
            if sel in tic_excluded:
                tic_excluded.remove(sel)
            else:
                tic_excluded.add(sel)
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
        if mode == "sampling":
            match_plot = or_(
                *(
                    RenderedTable.plot == plot_type
                    for plot_type in ["best", "convergence", "starting"]
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
        request,
        displayed_ticid,
        sort_state,
        SelectTICIDs,  # pylint: disable=invalid-name
        Rendered,  # pylint: disable=invalid-name
        db_session,
    ):
        """Return the context to render the view with."""

        mode = self.reviewing.rsplit("_", 1)[
            1
        ]  # pylint: disable=unsubscriptable-object
        pending_changes = {}
        if mode in _SAMPLING_PLOT_MODES:
            pending_changes = request.session.get("pending_changes", {}).get(
                str(displayed_ticid), {}
            )
        pending_toggles = pending_changes.get("toggles", [])

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
                for state, state_index in ([(0, "pending")] + self.states)
            ],
            "decisions": tuple(s[1] for s in self.states) + ("skip",),
            "review": self.reviewing,
            "mode": mode,
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
            cached_sed = cache.execute(
                select(CachedSED).filter_by(tic_id=displayed_ticid)
            ).scalar_one_or_none()
            db_threshold = (
                cached_sed.bad_sed_threshold
                if cached_sed and cached_sed.bad_sed_threshold is not None
                else ""
            )
        pending_threshold = pending_changes.get("bad_sed_threshold")
        context["bad_sed_threshold"] = (
            pending_threshold
            if "bad_sed_threshold" in pending_changes
            else db_threshold
        )
        context["has_pending_changes"] = bool(
            pending_toggles or "bad_sed_threshold" in pending_changes
        )
        context["displayed_ticid"] = displayed_ticid
        context.update(
            self._get_sector_context(displayed_ticid, pending_toggles)
        )
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

        state_slug_to_ind = {
            slugify(state[1]): state[0] for state in self.states
        }
        state_slug_to_ind["pending"] = 0
        state_slugs = {v: k for k, v in state_slug_to_ind.items()}
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
        status = 0
        with Session.begin() as db_session:
            if "displayed_ticid" not in kwargs:
                sort_state_index = state_slug_to_ind[kwargs["sort_state"]]
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
                    status = state_slug_to_ind[kwargs["decision"]]
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

                kwargs["displayed_ticid"] = (
                    db_session.scalar(select_ticid.order_by(SelectTICIDs.id))
                    or db_session.scalar(
                        select(SelectTICIDs.id).where(
                            SelectTICIDs.status == sort_state_index
                        )
                    )
                    or db_session.scalar(
                        select(SelectTICIDs.id)
                        .where(SelectTICIDs.status == status)
                        .where(
                            Rendered.id  # pylint: disable=singleton-comparison
                            != None
                        )
                        .order_by(SelectTICIDs.id)
                    )
                )

                print(f"Selected TIC ID: {kwargs['displayed_ticid']}")

            context = self._get_context(
                request,
                kwargs["displayed_ticid"],
                kwargs["sort_state"],
                SelectTICIDs,
                Rendered,
                db_session,
            )
        return render(request, "select_ticids/index.html", context)


def toggle_data(request, ticid, selection, review_table, mode):
    """Switch the state (enabled/disabled) for given data for given TIC ID."""

    if selection == "BLS":
        selection = "BLSOOE"
    elif selection not in ["OOE", "SPOC", "QLP"]:
        selection = int(selection)

    if mode in _SAMPLING_PLOT_MODES:
        pending_changes = request.session.get("pending_changes", {})
        tic_pending = pending_changes.setdefault(str(ticid), {})
        pending_toggles = tic_pending.setdefault("toggles", [])
        if selection in pending_toggles:
            pending_toggles.remove(selection)
        else:
            pending_toggles.append(selection)
        request.session["pending_changes"] = pending_changes
        request.session.modified = True
        return redirect(
            f"{review_table}_{mode}_jump",
            sort_state="pending",
            displayed_ticid=ticid,
        )

    excluded = exclude_data[ticid]
    if selection in excluded:
        excluded.remove(selection)
    else:
        excluded.add(selection)

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


def update_bad_sed_threshold(request, ticid, review_table, mode):
    """Stage or apply bad_sed_threshold for the given TIC ID."""

    threshold_str = request.POST.get("bad_sed_threshold", "").strip()
    if mode in _SAMPLING_PLOT_MODES:
        if threshold_str:
            pending_changes = request.session.get("pending_changes", {})
            tic_pending = pending_changes.setdefault(str(ticid), {})
            tic_pending["bad_sed_threshold"] = float(threshold_str)
            request.session["pending_changes"] = pending_changes
            request.session.modified = True
        return redirect(
            f"{review_table}_{mode}_jump",
            sort_state="pending",
            displayed_ticid=ticid,
        )
    if threshold_str:
        with CacheSession.begin() as cache:  # pylint: disable=no-member
            cache.execute(
                update(CachedSED)
                .filter_by(tic_id=ticid)
                .values(bad_sed_threshold=float(threshold_str))
            )
    return redirect(
        f"{review_table}_{mode}_jump",
        sort_state="pending",
        displayed_ticid=ticid,
    )


def apply_likelihood_changes(request, ticid, review_table, mode):
    """Apply all staged likelihood changes for the given TIC ID."""

    pending_changes = request.session.get("pending_changes", {})
    tic_pending = pending_changes.pop(str(ticid), {})
    request.session["pending_changes"] = pending_changes
    request.session.modified = True

    for selection in tic_pending.get("toggles", []):
        excluded = exclude_data[ticid]
        if selection in excluded:
            excluded.remove(selection)
        else:
            excluded.add(selection)

    threshold = tic_pending.get("bad_sed_threshold")
    if threshold is not None:
        with CacheSession.begin() as cache:  # pylint: disable=no-member
            cache.execute(
                update(CachedSED)
                .filter_by(tic_id=ticid)
                .values(bad_sed_threshold=float(threshold))
            )

    return redirect(
        f"{review_table}_{mode}_decision",
        sort_state="continue",
        decision="changed_likelihood",
        displayed_ticid=ticid,
    )
