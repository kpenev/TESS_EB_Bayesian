"""The available views for the app allowing the selection of TIC IDs."""

import sys
from os import path, remove
from base64 import b64encode
from functools import cached_property

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
    show_skipped = False
    states = [(1, "selected"), (2, "discarded")]
    grid = {"columns": "1fr", "rows": "1fr"}

    @cached_property
    def _state_slug_to_ind(self):
        """Map decision/state slugs to their integer index."""

        return {
            slugify(label): ind for ind, label in self.states
        } | {"pending": 0}

    @cached_property
    def _state_ind_to_slug(self):
        """Inverse of ``_state_slug_to_ind``."""

        return {v: k for k, v in self._state_slug_to_ind.items()}

    def _apply_decision(
        self,
        db_session,
        SelectTICIDs,  # pylint: disable=invalid-name
        kwargs,
    ):
        """Apply DB updates for the user's decision on the displayed TIC.

        Reads ``decision`` and ``displayed_ticid`` from ``kwargs``.
        ``skip`` is a pure navigation action — no DB write. The caller
        decides whether to advance to the next TIC based on the
        decision slug.
        """

        decision = kwargs["decision"]
        if decision == "skip":
            return
        if decision == "enable-review":
            updates = {"skip_review": False}
        elif decision == "disable-review":
            updates = {"skip_review": True}
        else:
            new_status = self._state_slug_to_ind[decision]
            updates = {"status": new_status} if new_status else {}
        if updates:
            db_session.execute(
                update(SelectTICIDs)
                .filter_by(id=kwargs["displayed_ticid"])
                .values(**updates)
            )

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

    def _outerjoin_rendered(
        self,
        select_stmt,
        SelectTICIDs,  # pylint: disable=invalid-name
        RenderedTable,  # pylint: disable=invalid-name
    ):
        """Outerjoin select with RenderedTable on the active plot type.

        The plot match restricts the join to the rendered plot type
        appropriate for the current review mode. No filters on
        ``rendered_only`` or ``skip_review`` are applied — use
        ``_filter_reviewable`` for navigation queries that need them.
        """

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

    def _filter_reviewable(
        self,
        select_stmt,
        SelectTICIDs,  # pylint: disable=invalid-name
        RenderedTable,  # pylint: disable=invalid-name
    ):
        """Outerjoin with RenderedTable and apply review filters.

        Skipped TICs are excluded unless ``self.show_skipped`` is true.
        Unrendered TICs are excluded when ``self.rendered_only`` is
        true.
        """

        select_stmt = self._outerjoin_rendered(
            select_stmt, SelectTICIDs, RenderedTable
        )
        if self.rendered_only:
            select_stmt = select_stmt.where(
                # pylint: disable=singleton-comparison
                RenderedTable.id != None
            )
        if not self.show_skipped:
            select_stmt = select_stmt.where(
                # pylint: disable=singleton-comparison
                SelectTICIDs.skip_review == False
            )
        return select_stmt

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

        select_expr = self._outerjoin_rendered(
            select(
                SelectTICIDs.id,
                Rendered.id != None,  # pylint: disable=singleton-comparison
                SelectTICIDs.skip_review,
            ),
            SelectTICIDs,
            Rendered,
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
            "displayed_ticid": displayed_ticid,
        }
        # pylint: enable=no-member

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

        if displayed_ticid is None:
            return context

        displayed_skip_review = db_session.scalar(
            select(SelectTICIDs.skip_review).filter_by(id=displayed_ticid)
        )
        context["decisions"] = context["decisions"] + (
            "enable review" if displayed_skip_review else "disable review",
        )

        pending_changes = {}
        if mode in _SAMPLING_PLOT_MODES:
            pending_changes = request.session.get("pending_changes", {}).get(
                str(displayed_ticid), {}
            )
        pending_toggles = pending_changes.get("toggles", [])

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
                else CachedSED.default_bad_sed_threshold
            )
            db_penalty = (
                cached_sed.bad_sed_penalty
                if cached_sed and cached_sed.bad_sed_penalty is not None
                else CachedSED.default_bad_sed_penalty
            )
        context["bad_sed_threshold"] = (
            pending_changes["bad_sed_threshold"]
            if "bad_sed_threshold" in pending_changes
            else db_threshold
        )
        context["bad_sed_penalty"] = (
            pending_changes["bad_sed_penalty"]
            if "bad_sed_penalty" in pending_changes
            else db_penalty
        )
        context["has_pending_changes"] = bool(
            pending_toggles
            or "bad_sed_threshold" in pending_changes
            or "bad_sed_penalty" in pending_changes
        )
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

        return context

    def get(self, request, **kwargs):
        """Allow user to review LCs from Villanova catalog and select some."""

        print(f"Select TIC ID view with kwargs: {kwargs}")
        if "sort_state" not in kwargs:
            kwargs["sort_state"] = "pending"

        print(f"State slugs: {self._state_ind_to_slug!r}")
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
                sort_state_index = self._state_slug_to_ind[
                    kwargs["sort_state"]
                ]
            else:
                sort_state_index = db_session.scalar(
                    select(SelectTICIDs.status).filter_by(
                        id=kwargs["displayed_ticid"]
                    )
                )
                kwargs["sort_state"] = self._state_ind_to_slug[
                    sort_state_index
                ]

            if "decision" in kwargs:
                assert "displayed_ticid" in kwargs
                self._apply_decision(db_session, SelectTICIDs, kwargs)
            if "displayed_ticid" not in kwargs or (
                "decision" in kwargs
                and kwargs["decision"] != "enable-review"
            ):
                select_ticid = self._filter_reviewable(
                    select(SelectTICIDs.id), SelectTICIDs, Rendered
                ).where(SelectTICIDs.status == sort_state_index)
                if "displayed_ticid" in kwargs:
                    select_ticid = select_ticid.where(
                        SelectTICIDs.id > kwargs["displayed_ticid"]
                    )

                next_ticid = db_session.scalar(
                    select_ticid.order_by(SelectTICIDs.id)
                )
                if next_ticid is None and "decision" in kwargs:
                    kwargs["displayed_ticid"] = None
                else:
                    kwargs["displayed_ticid"] = (
                        next_ticid
                        or db_session.scalar(
                            select(SelectTICIDs.id).where(
                                SelectTICIDs.status == sort_state_index
                            )
                        )
                        or db_session.scalar(
                            self._filter_reviewable(
                                select(SelectTICIDs.id),
                                SelectTICIDs,
                                Rendered,
                            )
                            .where(SelectTICIDs.status == sort_state_index)
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


def _record_bad_sed(ticid, values):
    """Set new bad SED threshold/penalty for the given TIC id."""

    with CacheSession.begin() as cache:  # pylint: disable=no-member
        cached_sed = cache.get(CachedSED, ticid)
        if cached_sed is None:
            cache.add(CachedSED(tic_id=ticid, **values))
        else:
            for key, value in values.items():
                setattr(cached_sed, key, value)


def update_bad_sed_threshold(request, ticid, review_table, mode):
    """Stage or apply bad SED parameters for the given TIC ID."""

    threshold_str = request.POST.get("bad_sed_threshold", "").strip()
    penalty_str = request.POST.get("bad_sed_penalty", "").strip()
    if mode in _SAMPLING_PLOT_MODES:
        pending_changes = request.session.get("pending_changes", {})
        tic_pending = pending_changes.setdefault(str(ticid), {})
        if threshold_str:
            tic_pending["bad_sed_threshold"] = float(threshold_str)
        if penalty_str:
            tic_pending["bad_sed_penalty"] = float(penalty_str)
        if threshold_str or penalty_str:
            request.session["pending_changes"] = pending_changes
            request.session.modified = True
        return redirect(
            f"{review_table}_{mode}_jump",
            sort_state="pending",
            displayed_ticid=ticid,
        )
    values = {}
    if threshold_str:
        values["bad_sed_threshold"] = float(threshold_str)
    if penalty_str:
        values["bad_sed_penalty"] = float(penalty_str)
    if values:
        _record_bad_sed(ticid, values)
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

    sed_values = {}
    if "bad_sed_threshold" in tic_pending:
        sed_values["bad_sed_threshold"] = float(
            tic_pending["bad_sed_threshold"]
        )
    if "bad_sed_penalty" in tic_pending:
        sed_values["bad_sed_penalty"] = float(tic_pending["bad_sed_penalty"])
    if sed_values:
        _record_bad_sed(ticid, sed_values)

    return redirect(
        f"{review_table}_{mode}_decision",
        sort_state="continue",
        decision="changed_likelihood",
        displayed_ticid=ticid,
    )
