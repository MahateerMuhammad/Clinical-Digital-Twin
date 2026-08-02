
# ── repo-root bootstrap ──────────────────────────────────────────────────────
# These scripts live two levels below the project root. Python puts the *script's*
# directory on sys.path, not the working directory, so `import src...` would fail
# from here; and many of them address data with root-relative paths such as
# "models/" or "reports/tables/". Both are fixed by putting the root on the path
# and running from it, which makes execution identical from any directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────
import sys
import traceback
from tests.fixtures.synthetic_patients import create_synthetic_tables
from tests.test_pipeline_regressions import (
    test_no_row_cap,
    test_chartevents_labevents_itemid_filter,
    test_no_false_duplicates_event_tables,
    test_radiology_field_ordinal_key,
    test_icu_merge_no_fanout,
    test_n_icu_stays_is_a_true_count,
    test_lab_missing_ratio_explicit,
    test_vitals_missing_ratio_behavior,
    test_lab_variant_features_separate,
    test_timestamp_order_flagged_not_dropped,
    test_vectorized_vs_original_lab_features,
)

test_funcs = [
    test_no_row_cap,
    test_chartevents_labevents_itemid_filter,
    test_no_false_duplicates_event_tables,
    test_radiology_field_ordinal_key,
    test_icu_merge_no_fanout,
    test_n_icu_stays_is_a_true_count,
    test_lab_missing_ratio_explicit,
    test_vitals_missing_ratio_behavior,
    test_lab_variant_features_separate,
    test_timestamp_order_flagged_not_dropped,
    test_vectorized_vs_original_lab_features,
]

def main():
    tables = create_synthetic_tables()
    passed = 0
    failed = 0
    print("=== STEP 2: REGRESSION TEST SUITE RESULTS ===\n")
    for fn in test_funcs:
        name = fn.__name__
        try:
            fn(tables)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n=== PHASE 0 UNIT TEST SUITE RESULTS ===\n")
    import unittest
    from tests.test_phase0 import TestPhase0
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase0)
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    if res.wasSuccessful():
        passed += res.testsRun
        print(f"  [PASS] All {res.testsRun} Phase 0 unit tests passed.")
    else:
        failed += len(res.failures) + len(res.errors)

    print(f"\nSummary: {passed} passed ({failed} failed).")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
