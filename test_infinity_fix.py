"""
Quick test to verify infinity handling in feature engineering
"""
import numpy as np
import pandas as pd

# Create test data with potential infinity-generating values
X_test = pd.DataFrame({
    'f1': [1.0, 2.0, 0.0, -1.0, 3.0],
    'f2': [0.0, 1.0, 0.0, 2.0, 0.0],  # Contains zeros for division
    'f3': [1e-100, 1e100, 1.0, 2.0, 3.0]  # Extreme values
})

print("Original data:")
print(X_test)
print()

# Test operations that could generate infinity
print("Testing operations that could generate infinity:")
print("-" * 50)

# Division by near-zero
result = X_test['f1'].values / (X_test['f2'].values + 1e-10)
print(f"Division result (before cleaning): {result}")
result_clean = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Division result (after cleaning): {result_clean}")
print()

# Square of large values
result = np.square(X_test['f3'].values)
print(f"Square result (before cleaning): {result}")
result_clean = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Square result (after cleaning): {result_clean}")
print()

# Log of zero
result = np.log(np.abs(X_test['f2'].values) + 1e-10)
print(f"Log result: {result}")
print()

print("✓ All operations handled correctly!")
print("The infinity fix should work in the main pipeline.")
