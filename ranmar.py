import random
import array
import sys


j = 33
k = 97
modval = 1<<24
modvalC = 1<<24 - 3

d = 7654321

# val = ""
file = open('ranmarout.txt', 'w')


random.seed(5)

def genSeedLFG():
	arr = []
	for i in range(k):
		arr.append(random.randint(0, modval));	
	return arr

def genSeedSeq():
	arr = []
	for i in range(k):
		arr.append(random.randint(0, modvalC));	
	return arr

# def conv(val):
# 	arr = []
# 	for i in xrange(len(val)):
# 		arr.append(int(val[i]))
# 	return arr

def showvals(val,j,k):
	for i in xrange(len(val)):
		if (i==j-1):  
			print "[%3d]"%val[i],
		elif (i==k-1):  
			print "[%3d]"%val[i],
		else:
			print "%3d"%val[i],

s=genSeedLFG()
sc = []
sc.append(1<<16) #random seed

print "j=",j," k=",k
# print "Seed:\t",val
if (len(s)<k):
	print "Value needs to be larger than 7"
	exit()
# showvals(s,j,k)
for n in xrange(100000):
    lfg = (s[j-1] - s[k-1]) % modval
    for i in xrange(len(s)-1):
			s[i] = s[i+1] 
    s[len(s)-1] = lfg
    
    if (n > 0):
    	sc.append((sc[n - 1] - d) % modvalC)

    out = lfg - sc[n]
    if (out < 0):
    	out += modval;

    # print "-->",out
    file.write(str(out)+"\n")
    # showvals(s,j,k)
# print "-->",out

file.close()