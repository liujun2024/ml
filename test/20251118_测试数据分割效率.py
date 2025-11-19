import pandas as pd
import numpy as np
import time
from ml import _utils as utils


df = pd.DataFrame(np.random.randn(1000000, 10))
n_chunks = 10

# 方案A: iloc 切片（最快）⭐⭐⭐
start = time.time()
# chunks = [df.iloc[i::n_chunks] for i in range(n_chunks)]
chuncks = utils.split_df(data=df, num=n_chunks)
print(f"iloc 耗时: {time.time() - start:.4f}s")
print(chuncks[0])

# 方案B: 使用 numpy.array_split
start = time.time()
chunks = np.array_split(df, n_chunks)
print(f"array_split 耗时: {time.time() - start:.4f}s")
print(chunks[0])
