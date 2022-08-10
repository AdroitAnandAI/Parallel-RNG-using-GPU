
import random
import array
import sys
import os
import datetime

NUM2GEN = 3000000

j = 127
k = 30
modval = 1<<32
print("Mode Value = ", modval)

# val = ""
file = open('lfgout.txt', 'w')


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
        sample.append(int(round(observation)))

    return sample


random.seed(5)

def conv():
	arr = []
	for i in range(j):
		arr.append(random.randint(0, modval));	
	return arr


# s = [1804289384, 846930887, 1681692778, 1714636916, 1957747794, 424238336, 719885387, 1649760493, 596516650, 1189641422, 1025202363, 1350490028, 783368691, 1102520060, 2044897764, 1967513927, 1365180541, 1540383427, 304089173, 1303455737, 35005212, 521595369, 294702568, 1726956430, 336465783, 861021531, 278722863, 233665124, 2145174068, 468703136, 1101513930, 1801979803, 1315634023, 635723059, 1369133070, 1125898168, 1059961394, 2089018457, 628175012, 1656478043, 1131176230, 1653377374, 859484422, 1914544920, 608413785, 756898538, 1734575199, 1973594325, 149798316, 2038664371, 1129566414, 184803527, 412776092, 1424268981, 1911759957]

s = random_uniform_sample(j, [0, modval])

print "j=",j," k=",k
# print "Seed:\t",val
if (len(s)<k):
	print "Value needs to be larger than 7"
	exit()


# skip = 15

timer = 0.0;

for n in xrange(NUM2GEN):

    t0 = datetime.datetime.now()

    out = (s[0] + s[k]) % modval
    for i in xrange(len(s)-1):
			s[i] = s[i+1] 
    s[len(s)-1] = out

    t1 = datetime.datetime.now()

    timer += (t1-t0).microseconds

    # if (n % skip == 0):
    file.write(str(out)+"\n")

print('Time taken to generate %d RNs: %f (ms). Gen. Speed = %f MRS\n\n' % (NUM2GEN, timer/1000, NUM2GEN/timer));


file.close()
