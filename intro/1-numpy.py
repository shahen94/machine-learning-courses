import numpy as np
from math import sqrt

X = [
    [0, 2]
]

Y = [
    [6, 10]
]

np_x = np.arange(9) - 2
np_y = np_x.reshape((3, 3))

print "np_x: {}".format(np_x)
print "np_y: {}".format(np_y)



def euclidean_distance(dx, dy):
    sum_ = 0
    for idx, value in enumerate(dx):
        sum_ += (abs(value - dy[idx]))**2
    return sqrt(sum_)

def euclidean_distance_np(dx):
    return np.linalg.norm(dx)


if __name__ == '__main__':
    print "Custom fn euclidean distance: {}".format(euclidean_distance(X[0], Y[0]))
    print "numpy linalg fn: {}".format(euclidean_distance_np(np_x))
    