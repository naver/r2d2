import numpy as np

cache_file= "/home/pfvaldez/Development/r2d2/d2-net/hpatches_sequences/cache/superpoint.npy"
data = np.load(cache_file, allow_pickle=True)
print(data)
# f = io.open(cache_file, mode="r", encoding="utf-8")
# file_list = []
# for line in f.readlines():
#     file_list.append(line.strip()) # strips newline character at the end of the line
# for i in sorted(file_list):
#     print(i)
# f.close()
