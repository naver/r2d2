##Load npz file .r2d2
import numpy as np

target_file= "/home/pfvaldez/Development/r2d2/d2-net/hpatches_sequences/hpatches-sequences-release/i_ajuntament/1.ppm.r2d2"
npz= np.load(target_file, mmap_mode='r')
# npz= np.load(cache_file)
# print(npz)

d = dict(zip(("data1{}".format(k) for k in npz), (npz[k] for k in npz)))
print(d)