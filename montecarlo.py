# https://asecuritysite.com/encryption/mc
import sys
import math


file1 = open("test","r+")
# print(file1.read())

values = []

for i in range(25000000):
	rnd = file1.readline()
	values.append(float(rnd))

# print(values[0])
# values = [5,125,10,1,32, 101,33, 54,200,230,215,93,50,100,3,6,43]


if (len(sys.argv)>1):
	values=eval(sys.argv[1])

max=1<<32 #255.0
inval=0.0
outval=0.0
ptr=0

print(max)

print ("First Five Pairs:")
print ("\tX\tY\tZ:")


for i in range(0,len(values)//2):
	# x = (values[ptr]-max/2)/(max/2)
	# y = (values[ptr+1]-max/2)/(max/2)

	x = (values[ptr])/(max)
	y = (values[ptr+1])/(max)

	z = math.sqrt(x*x+y*y)
	# print(z)
	if (i<5):
		print ("\t",round(x,3),"\t",round(y,3),"\t",round(z,3))
	if (z<1):
		inval=inval+2
	else:
		outval=outval+2
	ptr=ptr+2

print ("\nInval:",inval," Outval:",outval)

print ("\nResult: ",4.0*inval/(inval+outval))

# print ("\n----------------\nValues Used:",values)