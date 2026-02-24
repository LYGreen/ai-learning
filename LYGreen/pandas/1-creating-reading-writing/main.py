import pandas as pd
import numpy as np

# ====== DataFrame ======
# Create from Dictionary
df = pd.DataFrame({
    "name": ["A", "B", "C"],
    "age": [20, 21, 22]
})
print(df)

# Create from List
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df)

# ====== Series ======
df = pd.Series([1, 2, 3], index=['A', 'B', 'C'], name='numbers')
print(df)

# ====== numpy ======
arr = np.random.randn(3, 3)
df = pd.DataFrame(arr, columns=["A", "B", "C"])
print(df)
