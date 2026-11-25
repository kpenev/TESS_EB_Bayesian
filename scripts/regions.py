"""
The regions bridge sampling estimates the mass of (step 4 of TEC-142).

The posterior is split into regions, one per large group (after merges)
and one per confirmed small region, and their masses are estimated
separately. Any partition gives the right total, provided one fixed rule
assigns every point to at most one region, whatever its source: draws from
the reference densities are classified by `RegionRule`. Chain samples need
no classification: by construction the rule puts each of them in the
region of its label.

The rule (TEC-142 comments of 2026-11-23 and 2026-11-24):

1.  A point whose nearest pooled sample, in a confirmed small region's own
    whitening, is one of that region's samples is in that region (ties
    between small regions decided in the average of their metrics).

2.  Otherwise the kept 1a splits are replayed in order, giving the point's
    1a leaf. A leaf holding one large region gives that region; one holding
    several gives the region of the nearest of their samples, in the
    average of their own whitenings; one holding none gives the owner of
    the nearest large-group sample, in the whitening of all the large
    groups' positions together.
"""

import logging

import numpy
from scipy.spatial import cKDTree

from walker_distances import whiten

_logger = logging.getLogger(__name__)


class RegionRule:
    """
    Assign points to regions.

    Attributes:
        leaf_regions(dict):    The large regions each 1a leaf holds, sorted.
    """

    def __init__(self, grouping, positions, small_samples):
        """
        Prepare the rule for one target.

        Args:
            grouping(dict):    The grouping of the main chain's positions:
                ``splits``, the record of the 1a splits tried (see
                `postprocess.find_coarse_groups()`); ``coarse_labels``, the
                1a leaf of each position, shape (steps, walkers); ``labels``,
                the group of each position after 1b, discards and merges;
                and ``large_groups``, the labels of the large groups.

            positions(array):    The main chain's positions, shape (steps,
                walkers, parameters).

            small_samples(dict):    The samples of each confirmed small
                region (see `local_chain.resolve_small_groups()`).
        """

        if len(grouping["large_groups"]) == 0:
            raise ValueError(
                "No large group: the main chain has no region it samples "
                "reliably, so there is nothing to assign the rest of the "
                "parameter space to."
            )
        self._splits = [split for split in grouping["splits"] if split["kept"]]
        in_large = numpy.isin(grouping["labels"], grouping["large_groups"])
        pairs = numpy.unique(
            numpy.stack(
                [
                    grouping["coarse_labels"][in_large],
                    grouping["labels"][in_large],
                ],
                axis=1,
            ),
            axis=0,
        )
        self.leaf_regions = {
            int(leaf): sorted(pairs[pairs[:, 0] == leaf, 1].tolist())
            for leaf in numpy.unique(grouping["coarse_labels"])
        }
        # The nearest-sample searches, by key: what each one searches until
        # it is first used, then the built search (see `_get_search()`).
        self._search_specs = {}
        self._searches = {}
        large_positions = positions[in_large]
        large_owners = grouping["labels"][in_large]
        # Each confirmed small region's samples and whitening; searches for
        # ties between small regions are registered when first needed.
        self._small_regions = {
            region: (samples["positions"], whiten(samples["positions"]))
            for region, samples in small_samples.items()
        }
        for region, (samples, transform) in self._small_regions.items():
            self._search_specs["own", region] = (
                samples,
                numpy.full(samples.shape[0], region),
                transform,
            )
            self._search_specs["large", region] = (
                large_positions,
                large_owners,
                transform,
            )
        for leaf, regions in self.leaf_regions.items():
            if len(regions) > 1:
                in_leaf = grouping["coarse_labels"] == leaf
                # Whitening the regions together would shrink the gaps
                # between them; each region's own whitening keeps them.
                self._search_specs["leaf", leaf] = (
                    positions[in_leaf & in_large],
                    grouping["labels"][in_leaf & in_large],
                    self._average_metric(
                        [
                            whiten(
                                positions[
                                    in_leaf & (grouping["labels"] == region)
                                ]
                            )
                            for region in regions
                        ],
                        positions.shape[-1],
                    ),
                )
        self._search_specs["fallback"] = (
            large_positions,
            large_owners,
            whiten(large_positions),
        )

    def get_leaves(self, points):
        """
        Return the 1a leaf of each point, by replaying the kept splits.

        Each kept split, in the order the splits were made, classifies the
        points currently in the group it split with its own
        standardization and ``KMeans.predict()``. For the main chain's
        positions this reproduces the 1a labels.

        Args:
            points(array):    The points, one per row.

        Returns:
            array:    The leaf of each point.
        """

        leaves = numpy.zeros(points.shape[0], dtype=int)
        for split in self._splits:
            selected = numpy.flatnonzero(leaves == split["group"])
            if selected.size == 0:
                continue
            classifier = split["classifier"]
            in_second = (
                classifier["kmeans"].predict(
                    (points[selected] - classifier["mean"])
                    / classifier["scale"]
                )
                == 1
            )
            leaves[selected[in_second]] = split["new_group"]
        return leaves

    def classify(self, points):
        """
        Return the region of each point.

        First the confirmed small regions claim the points nearest their
        samples (`_claim_small()`). The others go through the replayed 1a
        splits (`get_leaves()`): a leaf holding one large region gives that
        region; one holding several gives the owner of the nearest of their
        samples, in the average of their own whitenings (each fitted to the
        region's positions in the leaf); one holding none
        gives the owner of the nearest large-group sample, in the whitening
        of all the large groups' positions together.

        Args:
            points(array):    The points, one per row.

        Returns:
            array:    The region (group label) of each point.
        """

        regions = self._claim_small(points)
        rest = numpy.flatnonzero(regions < 0)
        leaves = self.get_leaves(points[rest])
        for leaf in numpy.unique(leaves):
            here = rest[leaves == leaf]
            large = self.leaf_regions[int(leaf)]
            if len(large) == 1:
                regions[here] = large[0]
            else:
                regions[here] = self._find_nearest(
                    ("leaf", int(leaf)) if large else "fallback", points[here]
                )[1]
        return regions

    def _get_search(self, key):
        """
        Return the search with the given key, building it on first use.

        Only distinct samples are kept: a chain repeats a walker's position
        at every rejected step, repeats share an owner, and dropping them
        changes no nearest distance (at 6% acceptance it shrinks the tree
        about 14-fold).

        Args:
            key:    Names a search recorded in ``self._search_specs``
                (its samples, their owners, and the transform to the
                coordinates distances are measured in).

        Returns:
            (cKDTree, array, callable):
                The tree of the transformed distinct samples, their owners,
                and the transform.
        """

        if key not in self._searches:
            samples, owners, transform = self._search_specs.pop(key)
            distinct, index = numpy.unique(samples, axis=0, return_index=True)
            self._searches[key] = (
                cKDTree(transform(distinct)),
                numpy.asarray(owners)[index],
                transform,
            )
        return self._searches[key]

    def _claim_small(self, points):
        """
        Return the confirmed small region claiming each point, or -1.

        Region k claims a point if, in k's whitening, the point's nearest
        pooled sample (k's samples and the large groups') is one of k's.
        The large groups' samples are first searched approximately (the one
        found is at most twice as far as the nearest): if it is closer than
        k's nearest, k does not claim the point; if it is more than twice as
        far, k does. Only the points in between need an exact test. A point
        claimed by several small regions goes to the owner of the nearest of
        their samples in the average of their metrics.

        Args:
            points(array):    The points, one per row.

        Returns:
            array:    The claiming region of each point, -1 if none.
        """

        claimed = numpy.full(points.shape[0], -1)
        if not self._small_regions:
            return claimed
        regions = numpy.array(sorted(self._small_regions), dtype=int)
        claims = numpy.zeros((regions.size, points.shape[0]), dtype=bool)
        for row, region in enumerate(regions):
            own = self._find_nearest(("own", region), points)[0]
            large = self._find_nearest(("large", region), points, eps=1.0)[0]
            claims[row] = own < large / 2
            unsure = numpy.flatnonzero((own >= large / 2) & (own < large))
            claims[row, unsure] = ~self._has_closer(
                ("large", region), points[unsure], own[unsure]
            )
        num_claims = claims.sum(axis=0)
        single = num_claims == 1
        claimed[single] = regions[claims[:, single].argmax(axis=0)]
        for claimants in numpy.unique(claims[:, num_claims > 1], axis=1).T:
            tied = numpy.flatnonzero(
                (num_claims > 1) & (claims == claimants[:, None]).all(axis=0)
            )
            claimed[tied] = self._find_nearest(
                self._get_tie_key(tuple(regions[claimants].tolist())),
                points[tied],
            )[1]
        return claimed

    def _get_tie_key(self, regions):
        """
        Return the key of the search deciding ties between small regions.

        The search is registered the first time these regions tie: their
        samples, in the average of their metrics (`_average_metric()`).

        Args:
            regions(tuple):    The small regions that claim a point, sorted.

        Returns:
            tuple:    The search's key (see `_get_search()`).
        """

        key = ("tie",) + regions
        if key not in self._search_specs and key not in self._searches:
            samples = [self._small_regions[region][0] for region in regions]
            self._search_specs[key] = (
                numpy.concatenate(samples),
                numpy.repeat(regions, [len(part) for part in samples]),
                self._average_metric(
                    [self._small_regions[region][1] for region in regions],
                    samples[0].shape[1],
                ),
            )
        return key

    @staticmethod
    def _average_metric(transforms, num_parameters):
        """
        Return a transform whose squared distances average those of others.

        Each transform is linear plus an offset, so its squared distances
        are a quadratic form, with matrix L L^T for the transform's matrix
        L. The average of these forms is positive definite, and x -> x C for
        its Cholesky factor C (C C^T = the average) measures distances in
        the averaged metric with no more coordinates than parameters.

        Args:
            transforms([callable]):    Each maps positions (one per row) to
                its coordinates (e.g. from `whiten()`).

            num_parameters(int):    The number of parameters (columns) of a
                position.

        Returns:
            callable:    The transform for the averaged metric.
        """

        matrices = [
            transform(numpy.eye(num_parameters))
            - transform(numpy.zeros((1, num_parameters)))
            for transform in transforms
        ]
        factor = numpy.linalg.cholesky(
            numpy.mean([matrix @ matrix.T for matrix in matrices], axis=0)
        )
        return lambda values: values @ factor

    def _find_nearest(self, key, points, eps=0.0):
        """
        Return each point's distance to its nearest sample, and its owner.

        Args:
            key:    The search (see `_get_search()`).

            points(array):    The points, one per row.

            eps(float):    If positive, the sample found may be up to
                1 + eps times farther than the nearest (``cKDTree.query()``),
                which is much faster in many dimensions.

        Returns:
            (array, array):    The distance to the sample found and its
                owner.
        """

        tree, owners, transform = self._get_search(key)
        distance, index = tree.query(transform(points), eps=eps, workers=-1)
        return distance, owners[index]

    def _has_closer(self, key, points, radii):
        """
        Return whether any sample is within the given distance of each point.

        Args:
            key:    The search (see `_get_search()`).

            points(array):    The points, one per row.

            radii(array):    The distance for each point.

        Returns:
            array:    For each point, whether a sample is at most its radius
                away.
        """

        tree, _, transform = self._get_search(key)
        return (
            tree.query_ball_point(
                transform(points), radii, return_length=True, workers=-1
            )
            > 0
        )


def build_region_rule(main_chain, grouping, verdicts, small_samples, config):
    """
    Return the rule assigning points to bridge-sampling regions (step 4).

    No mass can be estimated while a small group that is not discarded has
    an unresolved local chain: its region may be real, with unknown mass.
    Such a target fails here. Rerunning resumes its local chains, or a
    local chain can be declared converged by hand.

    Args:
        main_chain(MainChain):    The main chain.

        grouping(dict):    ``splits`` and ``coarse_labels`` from
            `postprocess.find_coarse_groups()`, ``labels`` (the group of each
            position after 1b and merges, see
            `local_chain.resolve_small_groups()`) and ``discarded`` (the
            discarded groups).

        verdicts(dict):    The verdict of each small group's local chain.

        small_samples(dict):    The samples of each confirmed small region.

        config:    The command line configuration (``tic_id`` and
            ``group_min_large_walkers``).

    Returns:
        RegionRule:    The rule.
    """

    unresolved = sorted(
        group for group, verdict in verdicts.items() if verdict == "unresolved"
    )
    if unresolved:
        raise RuntimeError(
            f"TIC {config.tic_id}: the local chains of groups {unresolved} "
            "are unresolved, so no masses can be estimated; rerun to resume "
            "them, or declare them converged."
        )
    labels = grouping["labels"]
    large_groups = [
        int(group)
        for group in numpy.unique(labels)
        if group not in grouping["discarded"]
        and (labels == group).any(axis=0).sum()
        >= config.group_min_large_walkers
    ]
    rule = RegionRule(
        {
            "splits": grouping["splits"],
            "coarse_labels": grouping["coarse_labels"],
            "labels": labels,
            "large_groups": large_groups,
        },
        main_chain.positions,
        small_samples,
    )
    for region in large_groups:
        here = labels == region
        _logger.info(
            "Region %d (large): %.3f of positions, %d walkers ever, in 1a "
            "leaves %s.",
            region,
            here.mean(),
            here.any(axis=0).sum(),
            [
                leaf
                for leaf, regions in rule.leaf_regions.items()
                if region in regions
            ],
        )
    for region, samples in small_samples.items():
        _logger.info(
            "Region %d (small): %d local-chain samples from %d walkers.",
            region,
            samples["positions"].shape[0],
            numpy.unique(samples["walkers"]).size,
        )
    return rule
