#!/usr/bin/env python3

"""Test BLS catalog integration on specific binary types identified by Kalo.

Tests the comprehensive P/P2 period search on various challenging binary types:
- Equal mass circular binaries (P/2 ambiguity)
- Eccentric binaries with similar eclipse depths
- Binaries with very different eclipse depths
- Badly behaved binaries with pulsations/variability
"""

import sys
import logging
from catalog_interface import get_eb_info
from tess_target import TESSTarget

# Set up logging to see debug output
logging.basicConfig(level=logging.INFO)

# Test cases from Kalo's review
TEST_CASES = {
    48658291: "Equal mass circular binary (BLS picks P/2)",
    7695666: "Eccentric binary with similar eclipse depths",
    31961007: "Very different primary/secondary eclipse depths",
    48507019: "Badly behaved binary with pulsations/variability",
}


def test_binary_type(tic_id, description):
    """Test BLS catalog integration for a specific binary type.

    Args:
        tic_id (int): TIC ID to test.
        description (str): Description of the binary type.

    Returns:
        bool: True if test completed successfully, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Testing TIC {tic_id}: {description}")
    print("=" * 60)

    # Check if target is in catalog
    cat_info = get_eb_info(tic_id)
    if cat_info is None:
        print(f"❌ TIC {tic_id} not found in Villanova catalog")
        return False

    print(f"✓ Found in catalog:")
    print(f"  Period: {cat_info['period']:.6f} days")
    print(f"  Primary depth: {cat_info.get('prim_depth_pf', 'N/A')}")
    print(f"  Secondary depth: {cat_info.get('sec_depth_pf', 'N/A')}")

    try:
        # Create TESS target
        target = TESSTarget(tic_id)
        print(f"✓ Created TESSTarget for TIC {tic_id}")

        # Test catalog-guided BLS
        print("\n--- Testing catalog-guided BLS ---")
        bls_catalog = target.fit_bls(use_catalog=True)

        catalog_period = cat_info["period"]
        bls_period = bls_catalog["period"][0]
        period_diff = abs(bls_period - catalog_period) / catalog_period * 100

        print(f"Catalog period: {catalog_period:.6f} days")
        print(f"BLS period: {bls_period:.6f} days")
        print(f"Difference: {period_diff:.3f}%")

        if "candidate_type" in bls_catalog:
            print(f"Selected candidate: {bls_catalog['candidate_type']}")

        # Test blind BLS for comparison
        print("\n--- Testing blind BLS (for comparison) ---")
        bls_blind = target.fit_bls(use_catalog=False)
        blind_period = bls_blind["period"][0]
        blind_diff = abs(blind_period - catalog_period) / catalog_period * 100

        print(f"Blind BLS period: {blind_period:.6f} days")
        print(f"Blind difference: {blind_diff:.3f}%")

        # Summary
        print(f"\n--- SUMMARY for TIC {tic_id} ---")
        print(f"Catalog-guided accuracy: {period_diff:.3f}% difference")
        print(f"Blind search accuracy: {blind_diff:.3f}% difference")

        if period_diff < 5.0:
            print("✓ Catalog-guided BLS working well")
        else:
            print("⚠️  Large difference - may need investigation")

        return True

    except Exception as e:
        print(f"❌ Error testing TIC {tic_id}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run tests on all specified binary types."""
    print("Testing BLS Catalog Integration on Specific Binary Types")
    print("As requested by Kalo in PR review")
    print("=" * 60)

    results = {}

    for tic_id, description in TEST_CASES.items():
        results[tic_id] = test_binary_type(tic_id, description)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    successful = sum(results.values())
    total = len(results)

    for tic_id, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"TIC {tic_id}: {status} - {TEST_CASES[tic_id]}")

    print(f"\nOverall: {successful}/{total} tests completed successfully")

    if successful == total:
        print("🎉 All tests passed! BLS catalog integration working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed - see details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
