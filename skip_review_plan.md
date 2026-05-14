# Plan: `skip_review` flag for the BUI

## Goal

Let the user mark a TIC as "don't show me again in review" without changing
its sampling state. Concretely: a TIC with `status=1` (selected, currently
being sampled by `new_sampling.py` / `continue_sampling.py`) can also carry
`skip_review=True`, which only affects what the review views in
`scripts/bui/select_ticids/` display and navigate to.

## Design choice: orthogonal boolean column

`status` is overloaded — it both records a review decision and acts as the
"include in sampling" gate (`status > 0` in `new_sampling.py:195` and
`continue_sampling.py:130`). Reusing it for a "skip review" state would
either force the TIC out of sampling or duplicate every status with a
"reviewed/not-reviewed" twin.

Instead, add a new boolean column `skip_review` to the dynamic per-table
`SelectTable`. Sampling scripts and plot generation continue to consult
only `status`; the review UI ANDs `skip_review == False` into its TIC
listing queries.

## File-by-file changes

### 1. `scripts/bui/select_ticids/data_model.py`

Add the column inside `get_ticid_select_tables`, on the
dynamically-defined `SelectTable` class:

```python
skip_review: Mapped[bool] = mapped_column(
    default=False,
    server_default="0",
    nullable=False,
    doc="If true, hide this TIC from review listings. Independent of "
        "status / sampling inclusion.",
)
```

`server_default="0"` matters because the
`SelectTable.__table__.create(db_engine)` call inside
`get_ticid_select_tables` only runs for *new* tables. Existing rows in
already-created tables need a backfilled value (see migration step
below).

### 2. Migration for existing tables (one-off script)

`SelectTable` is created dynamically per review table name (e.g.
`sample_prsa`), so there is no Django/Alembic migration framework. For
each pre-existing table, run once against `scripts/bui/TESS_EBs.db`:

```sql
ALTER TABLE sample_prsa ADD COLUMN skip_review BOOLEAN NOT NULL DEFAULT 0;
```

Plan to add a small idempotent helper in `get_ticid_select_tables`,
on the `else` branch of the `has_table(tablename)` check, that does:

```python
existing_cols = {
    c["name"] for c in inspect(db_engine).get_columns(tablename)
}
if "skip_review" not in existing_cols:
    with db_engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE {tablename} ADD COLUMN skip_review BOOLEAN "
            "NOT NULL DEFAULT 0"
        ))
```

This keeps the auto-discovery / on-the-fly creation pattern that the rest
of the module uses, and avoids a separate manual step the next time a new
review table is introduced from a fresh DB versus an old one.

### 3. `scripts/bui/select_ticids/views.py`

Three concerns: filter queries, the decision handler, and a "show
skipped" override.

**a. Filter helper for navigation, plain outerjoin for sidebar:**

Filtering is for *navigation* (which TIC the view advances to or
displays next). The sidebar bucket lists, in contrast, must always
show **every** TIC in their status bucket — the rendered/skip-review
state is encoded as visual styling rather than as a filter, so the
user can still click any TIC. Concretely:

- Rename the old `_join_rendered` → `_filter_reviewable`. It does the
  outerjoin with `RenderedTable` on the correct plot type and applies
  *all* review-time filters:
  - appends `.where(RenderedTable.id != None)` when
    `self.rendered_only` is true (was previously duplicated at the
    call sites), and
  - appends `.where(SelectTICIDs.skip_review == False)` unless
    `self.show_skipped` is true.
- Extract the bare outerjoin (no filters) into a sibling helper, e.g.
  `_outerjoin_rendered`. `_filter_reviewable` calls it internally and
  then adds its `.where(...)` clauses. The sidebar query uses
  `_outerjoin_rendered` directly so the rendered flag is available as
  a column expression but neither filter is applied.
- Add `show_skipped = False` as a class attribute on
  `TICIdSelectorView` next to the other defaults (`reviewing`,
  `tablename`, `plot_dirs`, `rendered_only`, `states`).
- In `_get_context`, the `select_expr` driving the `by_state` sidebar
  lists now selects three columns —
  `(SelectTICIDs.id, Rendered.id != None, SelectTICIDs.skip_review)` —
  via `_outerjoin_rendered`, without any `.where(...)` filtering.
  Sidebar tuples become `(id, rendered_flag, skip_review_flag)`.
- In `get`, the `select_ticid` query for "next TIC" navigation
  continues to call `_filter_reviewable`, so navigation honors
  `rendered_only` and `skip_review`.

Observable consequences:

- Each status bucket in the sidebar shows every TIC with that status,
  regardless of `skip_review` or render state. The template
  distinguishes the four combinations via CSS classes (see section
  5b).
- "Next TIC" navigation in `get` (with `rendered_only=False`,
  `show_skipped=False` in current URL configs) walks to the next
  non-skipped TIC by id, regardless of whether a plot exists. The
  previous hard-coded `.where(Rendered.id != None)` is gone.
- Skipped TICs remain clickable in the sidebar and reachable by
  direct URL (the status lookup
  `db_session.scalar(select(SelectTICIDs.status).filter_by(id=kwargs["displayed_ticid"]))`
  never went through `_filter_reviewable`).

The buggy third fallback in the `db_session.scalar(...) or ...` chain
(originally using `status` instead of `sort_state_index` and
referencing `Rendered.id != None` without a join) is also routed
through `_filter_reviewable` and switched to `sort_state_index`. Note
this leaves the second fallback (the unfiltered
`status == sort_state_index` scalar) as the only path that can
surface a skipped or unrendered TIC during navigation, and only when
the filtered queries before and after it both return nothing.

**b. Decision-pipeline handling for review enable/disable:**

Rather than a separate handler, plug the flag into the existing
decision dispatch in `get`. Two new pseudo-decisions are recognized
alongside `skip` (which already moves on without touching the DB):

- **`disable review`** (slug: `disable-review`). Set
  `skip_review=True` on the displayed TIC and advance to the next TIC
  (same advance flow as `skip` — leave `status` alone). Offered when
  the currently displayed TIC has `skip_review=False`.
- **`enable review`** (slug: `enable-review`). Set
  `skip_review=False` on the displayed TIC and **stay** on it (so the
  user can review it now that it's re-enabled). Offered when the
  currently displayed TIC has `skip_review=True`. This is the only
  decision that does *not* trigger the next-TIC selection.

To keep `get` from ballooning, extract the DB-update logic into a
private method `_apply_decision`. Pylint enforces `max-args=5`, so the
state-slug-to-index map (currently rebuilt in every `get` call) moves
to a `@cached_property` on `TICIdSelectorView`. Its inverse (used to
recover the slug from an index in the direct-URL path) becomes a
sibling property.

```python
@cached_property
def _state_slug_to_ind(self):
    return {slugify(label): ind for ind, label in self.states} | {"pending": 0}

@cached_property
def _state_ind_to_slug(self):
    return {v: k for k, v in self._state_slug_to_ind.items()}

def _apply_decision(
    self,
    db_session,
    SelectTICIDs,  # pylint: disable=invalid-name
    kwargs,
):
    """Apply DB updates for the user's decision on the displayed TIC."""
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
```

`skip` is a pure navigation action and writes nothing — this drops the
old behavior where the original conditional structure issued a
no-op `UPDATE ... SET status = <same value>` and bumped the
`onupdate=func.now()` timestamp as a side effect. With no DB writes
on `skip`, `sort_state_index` is no longer needed as an argument and
the method takes `kwargs` as a single arg (decision + displayed_ticid
both read from it), landing at 4 args. The decision dispatch in `get`
then collapses to:

```python
if "decision" in kwargs:
    assert "displayed_ticid" in kwargs
    self._apply_decision(db_session, SelectTICIDs, kwargs)

if "displayed_ticid" not in kwargs or (
    "decision" in kwargs and kwargs["decision"] != "enable-review"
):
    # existing next-TIC selection logic
    ...
```

The two locally-built `state_slug_to_ind` / `state_slugs` dicts at the
top of `get` are removed in favor of the new properties. The
now-unused `status = 0` initialization at the top of the
`with Session.begin()` block also goes.

Note: when a TIC carries `skip_review=True` and the user picks
`disable review` immediately after enabling it (or vice versa), the
filter in `_filter_reviewable` will hide that TIC from the primary
"id > displayed_ticid" query as soon as the flag is set, but since
the query also requires `id > displayed_ticid`, the same TIC can't be
picked again either way — no special handling needed.

**c. "Show skipped" override:**

The `show_skipped = False` class attribute introduced in (a) already
gates the filter. To actually surface skipped TICs, add a parallel set
of URL patterns that mount `TICIdSelectorView` with `show_skipped=True`
under a distinct prefix (e.g. `sample_prsa/<mode>/skipped/...`). Can be
deferred to a follow-up — the minimum useful feature works without it,
since the attribute already exists and the filter is opt-out.

**d. Context exposure:**

In `_get_context`, look up `skip_review` for the currently displayed
TIC (a scalar query on `SelectTICIDs.skip_review`) and use it to build
the decisions tuple offered in the navbar:

```python
context["decisions"] = (
    tuple(s[1] for s in self.states)
    + ("skip",)
    + (("enable review",) if displayed_skip_review else ("disable review",))
)
```

This keeps the template logic simple — it just iterates `decisions`
and slugifies each label for the decision URL — and ensures the
correct enable/disable button is offered without an explicit
`{% if %}` in the template.

**e. End-of-bucket message:**

When the user makes a decision (skip or otherwise) on the
largest-id TIC currently displayable in the bucket per the active
`rendered_only` / `show_skipped` filters, the previous wrap-around
behavior is replaced with an explicit "all reviewed" state:

- In `get`, when `"decision" in kwargs` and the primary
  `_filter_reviewable` query returns `None`, set
  `kwargs["displayed_ticid"] = None` and skip the fallback chain. The
  fallbacks are preserved for the no-decision (initial-visit) path so
  an empty bucket on first load still finds something to show.
- In `_get_context`, hoist `select_expr`, `by_state`, and
  `render_progress` above the per-TIC work and return early when
  `displayed_ticid is None`. Cache lookups (`CachedBLS`, `CachedSED`),
  sector context, and image collection are skipped — they all key off
  `displayed_ticid` and would otherwise crash or produce noise.

The sidebar `by_state` lists, render-progress bar, and `decisions`
tuple remain in the context so the template can still render bucket
navigation when no TIC is displayed.

### 4. `scripts/bui/select_ticids/urls.py`

No new URL patterns needed for skip_review itself — the
`disable review` / `enable review` actions reuse the existing
`<slug:sort_state>/<int:displayed_ticid>/<slug:decision>/` decision
route, which already accepts arbitrary decision slugs.

If we adopt the "show skipped" override (3c), register a parallel set
of routes via a small variant of `get_review_urls` that sets
`show_skipped=True` and uses a distinct URL prefix
(e.g. `sample_prsa/<mode>/skipped/...`).

### 5. `scripts/bui/select_ticids/templates/`

Three concerns: the new review enable/disable buttons in `index.html`,
the end-of-bucket rendering in `index.html`, and the per-TIC styling
in the sidebar partial `tic_list.html`.

**a. Review enable/disable buttons.** No template change required —
the existing decision-button loop (`{% for choice in decisions %}`)
already renders one link per entry in `context["decisions"]`. Because
`_get_context` appends either `"disable review"` or `"enable review"`
to that tuple depending on the displayed TIC's `skip_review` flag
(see 3d), the correct button appears automatically alongside the
existing `skip` / status decisions.

**b. End-of-bucket rendering.** Gate all the per-TIC navbar elements on
`{% if displayed_ticid %}`:

- the decision-button `<ul>`
- the `.excluded-data` sectors panel
- the entire `.nav-controls` div (replot button, TIC-ID input,
  bad-SED form)

In `.main`, branch the `.img-parent` block:

```django
{% if displayed_ticid %}
<div class="img-parent" style="grid-template-columns: {{ grid.columns }}; grid-template-rows: {{ grid.rows }};">
    {% include "./images.html" with images=images%}
</div>
{% else %}
<div class="img-parent">
    <h2>All {{ sort_state }} TICs reviewed</h2>
</div>
{% endif %}
```

The `#progress-with-label` bar and the `.ticid_lists` sidebar
(per-bucket TIC lists) stay unconditional so the user can switch
buckets or jump to any specific TIC from the end-of-bucket state.

**c. Sidebar TIC styling (`tic_list.html`).** Each sidebar entry now
unpacks the `(id, rendered_flag, skip_review_flag)` tuple produced
by `_get_context` (see 3a) and reflects both flags as CSS classes on
the `<a>`:

```django
{% for ticid in list %}
<li {% if ticid.0 == displayed_ticid %}class="displayed"{% endif %}>
    <a href="{% url review|add:'_jump' sort_state=list_name|slugify displayed_ticid=ticid.0 %}"
       class="{% if ticid.1 %}rendered{% else %}not-rendered{% endif %}{% if ticid.2 %} review-disabled{% endif %}">
        {{ ticid.0 }}
    </a>
</li>
{% endfor %}
```

(Also fixes the existing bug where `not-rendered` was written without
the surrounding `class="..."` and so never applied.)

CSS (in `select_ticids/static/select_ticids/css/tic_lists.css`):
keep the existing `.rendered { font-weight: bold; }` rule and add
`.review-disabled { text-decoration: line-through; }`. This yields the
four combinations the user wants:

| rendered | skip_review | style                          |
|----------|-------------|--------------------------------|
| yes      | no          | bold, normal text              |
| yes      | yes         | bold, crossed out              |
| no       | no          | normal weight, normal text     |
| no       | yes         | normal weight, crossed out     |

Every TIC remains clickable regardless of state.

### 7. Auto-re-enable review on chain convergence

A TIC that was marked `skip_review=True` while its chain was still
churning should have the flag cleared automatically once sampling
has progressed past burn-in for every parameter — at that point
there is new information worth re-reviewing, even if the user
previously flagged the TIC as not interesting.

The convergence predicate (`burnin < num_steps`) is computed inside
`get_convergence_data`, which is data preparation rather than
plotting. Today it lives in `scripts/visualize.py` alongside several
other non-plotting helpers, but a cleaner home is a dedicated
analysis module — see 7b.

**a. Helper in `scripts/bui/select_ticids/data_model.py`:**

Add `clear_skip_review(tic_id)` that clears the flag in every
review table known to the BUI:

```python
def clear_skip_review(tic_id):
    """Clear skip_review on tic_id across every review table."""

    with Session.begin() as db_session:
        review_tables = set(
            db_session.scalars(
                select(JobGroup.select_tic_table)
            ).all()
        )
    for tablename in review_tables:
        SelectTable, _ = get_ticid_select_tables(
            tablename, must_exist=True
        )
        with Session.begin() as db_session:
            db_session.execute(
                update(SelectTable)
                .where(
                    SelectTable.id == tic_id,
                    # pylint: disable=singleton-comparison
                    SelectTable.skip_review == True,
                )
                .values(skip_review=False)
            )
```

Notes:
- Review tables are discovered via the existing
  `JobGroup.select_tic_table` column, so any table actively backing
  a job group is covered without hard-coding names.
- The `skip_review == True` predicate makes the update idempotent —
  subsequent calls after the flag is cleared don't bump the
  `timestamp` column.
- `must_exist=True` keeps `get_ticid_select_tables` from accidentally
  creating an unexpected new table.

**b. New module `scripts/chain_analysis.py`:**

Extract the non-plotting MCMC-chain helpers from `visualize.py` into
a new module. None of these touch matplotlib; they operate on chain
arrays and config objects, and several of them are reusable outside
the plotting flow.

Move from `scripts/visualize.py` to `scripts/chain_analysis.py`:

- `get_pickler` (line ~340) — only caller is `get_convergence_data`,
  and leaving it in `visualize.py` while moving `get_convergence_data`
  would create a circular import (`visualize` imports
  `chain_analysis`, which would have to import back).
- `get_chain_expressions` (line ~367)
- `get_convergence_data` (line ~472)
- `get_lstsq_interpreters` (line ~646)
- `get_initial_positions` (line ~952)
- `get_walker_step_params` (line ~985)
- `get_lstsq` (line ~1033)

Leave in `visualize.py`:

- `get_param_binaries` (line ~940) and `get_model_binaries`
  (line ~1074) — too tightly coupled to plotting flow (build
  `Binary` objects keyed off `config.show_model_with_lc`,
  `config.show_lstsq`, etc.).
- `get_plot_data` (line ~1138) — name aside, its responsibilities
  (selecting samples, applying `sample_condition`, `max_plot_steps`)
  are plot-driven.
- All `create_*` functions, `MovieMaker`, `include_in_axis`,
  `hex_color`, `parse_command_line`, `main`.

`visualize.py` keeps working by importing the moved helpers back at
the top of the file, e.g.:

```python
from chain_analysis import (
    get_pickler,
    get_chain_expressions,
    get_convergence_data,
    get_lstsq_interpreters,
    get_initial_positions,
    get_walker_step_params,
    get_lstsq,
)
```

**c. Hook in `scripts/chain_analysis.py`:**

Inside `get_convergence_data`, right after the existing
`burnin = convergence_data["burnin"].max()` line, add:

```python
burnin = convergence_data["burnin"].max()
if burnin < num_steps:
    clear_skip_review(config.tic_id)
print(f"Burnin is {burnin} out of {num_steps} steps.")
```

Import `clear_skip_review` from
`bui.select_ticids.data_model` at the top of `chain_analysis.py`.
`config.tic_id` is already wired through the CLI/config-file parsing
in `visualize.py` (see `parser.add_argument("tic_id", ...)` around
line 52), so no plumbing changes are needed at the call site. The
docstring of `get_convergence_data` should be updated to note the
side effect (re-enabling review on convergence).

`create_convergence_plot` itself is left untouched — its
convergence-region overlay continues to read
`convergence_data["burnin"]` and `convergence_data["num_steps"]`,
which `get_convergence_data` already produces. The function now
lives in `chain_analysis.py` and is reusable by any future caller
that wants a convergence verdict without importing matplotlib.

**Open questions for 7:**

1. ~~Review-table discovery via `JobGroup.select_tic_table` covers
   every table that has ever backed a job group. If a TIC was
   inserted into a review table not tied to a job group, it would
   be missed.~~ **Resolved:** a TIC that is not in a job group has
   no sampling running for it and therefore cannot accumulate the
   new steps needed to cross burn-in, so it is impossible for the
   convergence path to fire on such a TIC. Discovery via
   `JobGroup.select_tic_table` is sufficient.
2. ~~Pickling: a re-run that hits the cache would skip the
   re-enable step.~~ **Resolved:** a pickled
   `convergence_data` is only reusable when the chain has not
   accumulated any new steps. With no new steps, the convergence
   verdict cannot have changed — either the TIC was already past
   burn-in on the first run (in which case `clear_skip_review` ran
   then) or it still isn't, so a cache-hit short-circuit
   correctly skips the call.

### 6. Things deliberately NOT changed

- **`scripts/new_sampling.py`** (`SelectTICTable.status > 0` at line 195
  and `.status.in_(...)` at line 189): untouched. Sampling inclusion
  remains a pure function of `status`.
- **`scripts/continue_sampling.py`** (lines 130, 136, 196–200): same.
- **`scripts/bui/select_ticids/plots.py`** (`status.in_(...)` at
  line 307): untouched. Plot rendering for review continues regardless of
  `skip_review` so that an "unskip" later still has a plot to show.

## Testing checklist

- Fresh DB: confirm `SelectTable.__table__.create` includes the new
  column.
- ~~Existing DB (`scripts/bui/TESS_EBs.db`): confirm the idempotent ALTER
  runs once, succeeds, and is a no-op on subsequent loads.~~
- Click `disable review` on a `status=1` TIC; rerun `new_sampling.py`
  (or inspect the query) and confirm it is still queued for sampling.
- ~~Click `disable review` on a `status=0` (pending) TIC; confirm the
  navbar advances to the next TIC, that the disabled TIC still
  appears in the pending bucket list but rendered with the
  `review-disabled` strike-through style, that it is skipped in the
  "next TIC" rotation, and that it remains reachable by clicking it
  in the sidebar (or by direct URL).~~
- ~~Navigate to a `skip_review=True` TIC — either by clicking its
  crossed-out entry in the sidebar bucket list or by typing its URL.
  Confirm the navbar shows `enable review` (not `disable review`).
  Click it and confirm the flag is cleared, the view stays on the
  same TIC, and the navbar now shows `disable review`.~~
- ~~"Next TIC" navigation skips over disabled-review TICs: arrange a
  bucket containing TICs A < B < C where B has `skip_review=True`.
  From A, click any decision that advances (`skip`, `selected`,
  `discarded`, or `disable review`); confirm the view jumps to C,
  not B.~~
- ~~`skip` is a no-op write-wise: note the displayed TIC's
  `timestamp` column, click `skip`, and confirm the row is
  unchanged (`timestamp` did not move, no other column updated).~~
- ~~End-of-bucket triggers for any decision, not just `disable
  review`: with the largest-id displayable TIC in a bucket, exercise
  `skip` and a regular status decision (e.g. `selected`)
  separately; in each case confirm the main area shows "All
  {sort_state} TICs reviewed", per-TIC navbar elements are hidden,
  and sidebar headers and TIC links remain clickable.~~
- ~~Verify the four sidebar styles render correctly: pick four TICs
  covering (rendered, not skipped), (rendered, skipped), (not
  rendered, not skipped), (not rendered, skipped); confirm bold vs.
  normal weight tracks `rendered` and strike-through tracks
  `skip_review`, and that all four remain clickable.~~
- Auto-re-enable on convergence: mark a TIC `skip_review=True`,
  then run `visualize.py <tic_id>` (or the normal sampling-driven
  convergence-plot pipeline) once its chain has progressed past
  burn-in for every parameter. Confirm `get_convergence_data` clears
  the flag in every review table containing the TIC and that the
  convergence plot itself is unchanged. Re-run; confirm the
  `timestamp` column is not bumped a second time (the
  `skip_review == True` filter makes the update idempotent).
- Negative case: with a TIC where the chain has *not* yet crossed
  burn-in (`burnin >= num_steps`), confirm `clear_skip_review` is
  not called and the flag stays `True`.

## Open questions for review

1. ~~Should `skip_review` filter only the "pending" list, or all
   status buckets in the sidebar?~~ **Resolved:** sidebar lists are
   not filtered — every TIC in a status bucket is shown, with
   `rendered` and `skip_review` reflected as CSS classes (see 3a and
   5c). Filtering is navigation-only.
2. ~~Is the "show skipped" override (3c) worth doing in the same
   change, or split into a follow-up?~~ **Resolved:** keep the
   `show_skipped` class attribute as a `False` default, but do not
   add a URL variant that sets it to `True`. Sidebar lists and direct
   URLs already reach every TIC, so the only thing a `show_skipped`
   mount would add is walking skipped TICs in the "next TIC"
   rotation — useful enough to keep the hook, not useful enough to
   wire URLs in this change.
3. ~~Anywhere else `skip_review` should propagate — e.g. should
   `plots.py`'s rendering loop honor it as a default in `limit_to_*`
   filters, or stay fully orthogonal?~~ **Resolved:** stay fully
   orthogonal. `plots.py` is not updated; plot rendering continues to
   be driven solely by `status` so that re-enabling review on a
   previously skipped TIC still has a plot to show.
