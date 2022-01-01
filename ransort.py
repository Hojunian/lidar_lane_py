import numpy as np
import random

# rand_quicksort module
def partition(points, p, r, coord):
    # for special(wrong) input
    if p > r:
        return 'wrong index'
    elif p == r:
        return p

    x = points[r,coord]
    i = p - 1

    for j in range(p, r):
        if points[j, coord] <= x:
            i += 1
            points[[i, j]] = points[[j, i]]

    points[[i+1, r]] = points[[r, i+1]]

    return i + 1  # return the index of pivot


def randomized_quicksort(points, a, b, i):
    n = b-a
    # for corner case
    if n == 0:
        return 'empty'
    elif n == 1:
        return points

    r = random.randrange(a, b)
    points[[b-1, r]] = points[[r, b-1]]
    q = partition(points, a, b-1,i)
    randomized_quicksort(points, a, q, i)
    randomized_quicksort(points, q, b, i)