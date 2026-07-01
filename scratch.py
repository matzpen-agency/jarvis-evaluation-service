from collections import Counter
a = [(1.0,), (2.0,), (3.0,)]
b = [(3.0,), (1.0,), (2.0,)]
print("a == b?", Counter(a) == Counter(b))
