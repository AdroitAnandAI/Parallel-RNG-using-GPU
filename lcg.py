# The stub code obtained from here and modified accordingly.
# https://stackoverflow.com/questions/19140589/linear-congruential-generator-in-python

import random
import array
import sys


def lcg(x, a, c, m):
    while True:
        x = (a * x + c) % m
        yield x


def random_uniform_sample(n, interval, seed=0):
    a, c, m = 1103515245, 12345, 2 ** 31
    bsdrand = lcg(seed, a, c, m)

    lower, upper = interval[0], interval[1]
    sample = []

    for i in range(n):
        observation = (upper - lower) * ((float)(next(bsdrand)) / (2 ** 31 - 1)) + lower
        sample.append(round(observation))

    return sample

# 30 numbers between 0 and 100
rus = random_uniform_sample(30, [0, 100])
print(rus)