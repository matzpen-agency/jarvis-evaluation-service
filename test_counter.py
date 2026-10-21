from collections import Counter

expected_rows = [(100.0,), (200.0,), (300.0,)]
generated_rows = [(100.0,), (200.0,), (300.0,)]
print("expected == generated:", expected_rows == generated_rows)
print("Counter(expected) == Counter(generated):", Counter(expected_rows) == Counter(generated_rows))
print("expected_rows:", expected_rows)
print("generated_rows:", generated_rows)
