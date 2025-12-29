"""
Quick verification script to test that both Task 2a and Task 2b scripts are syntactically correct
and can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*70)
print("VERIFICATION: Testing Task 2 Scripts")
print("="*70)

# Test Task 2a imports
print("\n[1/3] Testing Task 2a baseline model script imports...")
try:
    from task2a_baseline_model import (
        prepare_features_target,
        stratified_train_test_split,
        train_baseline_logistic_regression,
        evaluate_and_print_metrics
    )
    print("  ✓ All Task 2a functions imported successfully")
except Exception as e:
    print(f"  ✗ Error importing Task 2a: {e}")
    sys.exit(1)

# Test Task 2b imports
print("\n[2/3] Testing Task 2b ensemble model script imports...")
try:
    from task2b_ensemble_model import (
        calculate_scale_pos_weight,
        train_xgboost_ensemble,
        perform_stratified_cv,
        hyperparameter_tuning_light,
        compare_models
    )
    print("  ✓ All Task 2b functions imported successfully")
except Exception as e:
    print(f"  ✗ Error importing Task 2b: {e}")
    sys.exit(1)

# Test that models directory exists
print("\n[3/3] Verifying repository structure...")
models_dir = Path('models')
tests_dir = Path('tests')
reports_dir = Path('reports')

if models_dir.exists():
    print(f"  ✓ models/ directory exists")
else:
    print(f"  ✗ models/ directory missing")

if tests_dir.exists():
    print(f"  ✓ tests/ directory exists")
else:
    print(f"  ✗ tests/ directory missing")

if (reports_dir / 'task2_report.md').exists():
    print(f"  ✓ reports/task2_report.md exists")
else:
    print(f"  ✗ reports/task2_report.md missing")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nAll Task 2 scripts are syntactically correct and ready to run.")
print("\nTo execute the full pipeline:")
print("  1. python src/task2a_baseline_model.py")
print("  2. python src/task2b_ensemble_model.py")
print("\nNote: Task 2b includes hyperparameter tuning and may take several minutes.")
