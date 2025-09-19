"""Interface for accessing the Villanova TESS Eclipsing Binary Catalog.

This module provides a clean Python interface to query eclipsing binary
parameters from the Villanova TESS-EBs catalog. It supports looking up
periods, eclipse properties, and other parameters needed for BLS analysis.
"""

import logging
import os
from pathlib import Path

import pandas

CATALOG_PATH = Path(__file__).parent.parent / "tess_ebs_catalog.csv"

logger = logging.getLogger(__name__)


class TESSEBsCatalog:
    """Interface to the Villanova TESS Eclipsing Binary Catalog.

    This class provides methods to load and query the TESS-EBs catalog,
    extracting eclipsing binary parameters like orbital periods, eclipse
    depths, and timing information needed for period analysis.
    """

    def __init__(self, catalog_path=None):
        """Initialize the catalog interface.

        Args:
            catalog_path (str, optional): Path to the catalog CSV file.
                Defaults to tess_ebs_catalog.csv in parent directory.
        """
        if catalog_path is None:
            catalog_path = CATALOG_PATH

        self.catalog_path = catalog_path
        self.catalog_data = None
        self.load_catalog()

    def load_catalog(self):
        """Load the TESS-EBs catalog from CSV file.

        Attempts to load the catalog data into a pandas DataFrame.
        Sets catalog_data to None if loading fails.
        """
        if not os.path.exists(self.catalog_path):
            logger.warning("Catalog file not found at %s", self.catalog_path)
            self.catalog_data = None
            return

        try:
            self.catalog_data = pandas.read_csv(self.catalog_path)
            logger.info(
                "Loaded TESS-EBs catalog with %d entries",
                len(self.catalog_data),
            )
        except (
            FileNotFoundError,
            pandas.errors.EmptyDataError,
            pandas.errors.ParserError,
        ) as exc:
            logger.error("Failed to load catalog: %s", exc)
            self.catalog_data = None

    def is_available(self):
        """Check if catalog data is loaded and available.

        Returns:
            bool: True if catalog data is loaded, False otherwise.
        """
        return self.catalog_data is not None

    def get_eb_info(self, tic_id):
        """Get eclipsing binary information for a given TIC ID.

        Args:
            tic_id (int): TESS Input Catalog identifier.

        Returns:
            dict or None: Dictionary containing EB parameters if found,
                None if TIC ID not in catalog or catalog unavailable.

        Dictionary contains:
            - tic_id: TESS Input Catalog ID
            - period: Orbital period in days
            - period_uncert: Period uncertainty in days
            - bjd0: Reference epoch (BJD)
            - bjd0_uncert: Epoch uncertainty
            - prim_depth_pf: Primary eclipse depth (polynomial fit)
            - sec_depth_pf: Secondary eclipse depth (polynomial fit)
            - ra, dec: Coordinates
            - tmag: TESS magnitude
            - Additional eclipse parameters if available
        """
        if not self.is_available():
            return None

        matches = self.catalog_data[self.catalog_data["tess_id"] == tic_id]
        if len(matches) == 0:
            return None

        eb_data = matches.iloc[0]

        result = {
            "tic_id": int(eb_data["tess_id"]),
            "period": float(eb_data["period"]),
            "period_uncert": (
                float(eb_data["period_uncert"])
                if pandas.notna(eb_data["period_uncert"])
                else None
            ),
            "bjd0": float(eb_data["bjd0"]),
            "bjd0_uncert": (
                float(eb_data["bjd0_uncert"])
                if pandas.notna(eb_data["bjd0_uncert"])
                else None
            ),
            "morph_coeff": (
                float(eb_data["morph_coeff"])
                if pandas.notna(eb_data["morph_coeff"])
                else None
            ),
            "ra": float(eb_data["ra"]),
            "dec": float(eb_data["dec"]),
            "tmag": (
                float(eb_data["Tmag"])
                if pandas.notna(eb_data["Tmag"])
                else None
            ),
        }

        if pandas.notna(eb_data["prim_depth_pf"]):
            result["prim_depth_pf"] = float(eb_data["prim_depth_pf"])
        if pandas.notna(eb_data["prim_width_pf"]):
            result["prim_width_pf"] = float(eb_data["prim_width_pf"])
        if pandas.notna(eb_data["prim_pos_pf"]):
            result["prim_pos_pf"] = float(eb_data["prim_pos_pf"])

        if pandas.notna(eb_data["sec_depth_pf"]):
            result["sec_depth_pf"] = float(eb_data["sec_depth_pf"])
        if pandas.notna(eb_data["sec_width_pf"]):
            result["sec_width_pf"] = float(eb_data["sec_width_pf"])
        if pandas.notna(eb_data["sec_pos_pf"]):
            result["sec_pos_pf"] = float(eb_data["sec_pos_pf"])

        for param in [
            "prim_depth_2g",
            "prim_width_2g",
            "prim_pos_2g",
            "sec_depth_2g",
            "sec_width_2g",
            "sec_pos_2g",
        ]:
            if pandas.notna(eb_data[param]):
                result[param] = float(eb_data[param])

        if pandas.notna(eb_data["sectors"]):
            result["sectors"] = eb_data["sectors"]

        return result

    def get_period_range(self, tic_id, factor=0.1):
        """Get period search range around catalog value.

        Args:
            tic_id (int): TESS Input Catalog identifier.
            factor (float): Fractional range around catalog period.
                Default 0.1 means ±10% of catalog period.

        Returns:
            tuple or None: (min_period, max_period) in days if found,
                None if TIC ID not in catalog.
        """
        eb_info = self.get_eb_info(tic_id)
        if eb_info is None:
            return None

        period = eb_info["period"]
        period_uncert = eb_info.get("period_uncert", None)

        if period_uncert is not None and period_uncert > 0:
            range_size = max(3 * period_uncert, factor * period)
        else:
            range_size = factor * period

        min_p = max(0.1, period - range_size)
        max_p = min(100.0, period + range_size)

        return (min_p, max_p)

    def get_all_tic_ids(self):
        """Get list of all TIC IDs in the catalog.

        Returns:
            list: List of TIC IDs as integers, empty list if catalog
            unavailable.
        """
        if not self.is_available():
            return []
        return self.catalog_data["tess_id"].tolist()

    def search_by_period(self, min_period, max_period):
        """Find all TIC IDs with periods in specified range.

        Args:
            min_period (float): Minimum period in days.
            max_period (float): Maximum period in days.

        Returns:
            list: List of TIC IDs with periods in range,
                empty list if catalog unavailable.
        """
        if not self.is_available():
            return []

        mask = (self.catalog_data["period"] >= min_period) & (
            self.catalog_data["period"] <= max_period
        )
        return self.catalog_data[mask]["tess_id"].tolist()


def get_catalog():
    """Get the singleton catalog instance.

    Returns:
        TESSEBsCatalog: Shared catalog instance, created on first call.
    """
    if getattr(get_catalog, 'catalog_instance', None) is None:
        get_catalog.catalog_instance = TESSEBsCatalog()
    return get_catalog.catalog_instance


def get_eb_info(tic_id):
    """Convenience function to get EB info for a TIC ID.

    Args:
        tic_id (int): TESS Input Catalog identifier.

    Returns:
        dict or None: EB parameters dict or None if not found.
    """
    return get_catalog().get_eb_info(tic_id)


def get_catalog_period_range(tic_id, factor=0.1):
    """Convenience function to get period search range.

    Args:
        tic_id (int): TESS Input Catalog identifier.
        factor (float): Fractional range around catalog period.

    Returns:
        tuple or None: (min_period, max_period) or None if not found.
    """
    return get_catalog().get_period_range(tic_id, factor)


def is_in_catalog(tic_id):
    """Check if a TIC ID exists in the catalog.

    Args:
        tic_id (int): TESS Input Catalog identifier.

    Returns:
        bool: True if TIC ID is in catalog, False otherwise.
    """
    return get_eb_info(tic_id) is not None
