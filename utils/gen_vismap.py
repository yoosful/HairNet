import os
import numpy as np


# read strandsXXXXX_YYYYY_AAAAA_mBB.vismap
# v = numpy.load(filename)
# Dimension: 100*32*32
# v[i, n, m] is the visibility of the ith point on the [n,m]th strand. 1 means visible, 0 means invisible.  The visibility is computed from the view of the image.
def gen_vis_weight(path, weight_max=10.0, weight_min=0.1):
    vismap = np.load(path)
    weight = vismap
    for i in range(0, 32):
        for j in range(0, 32):
            for k in range(0, 100):
                if vismap[k, i, j] == 1.0:
                    weight[k, i, j] = weight_max
                elif vismap[k, i, j] == 0.0:
                    weight[k, i, j] = weight_min
                else:
                    print("There is something wrong!")
    return weight


def gen_vis_npy():
    files = os.listdir("data")
    for file in files:
        if ".vismap" in file:
            np.save("vismap/" + file.split(".")[0], gen_vis_weight("data/" + file))
