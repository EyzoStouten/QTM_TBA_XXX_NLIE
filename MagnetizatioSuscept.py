import numpy as np
import math,cmath
import matplotlib.pyplot as pl
import matplotlib.axes as ax
from Magnetization_Fn import Magnetization

#Number of intervals. Scales exponentially! linspace only picks a few of
#these to calculate and draw a line.
points = 100
T_max = 1
#as of now Free_Energy quits after beta>900 for zero magnetic fields take care!
# It does not like increasing magnetic field at all. So keep beta<100 there for h>1
beta_max = 100.
T = np.linspace(1./beta_max,T_max,points)
beta = 1/T
h=0
J=1/2.
L=100
FE1 = np.zeros(points,float)
MAG = np.zeros(points,float)
SUC = np.zeros(points,float)

for i in range(points):
    #Free_Energy : vars : Free_Energy(bet = 100.,h = 0,J = 2.,L = 100.,N = 2**14):
    #Last N is Trotter number!

    out = Magnetization(beta[i],h,J,L)
    FE1[i] = out[0]
    MAG[i] = out[1]
    SUC[i] = out[2]

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return ','.join([i, (d+'0'*n)[:n]])

filename='Su2hAllh%sJ%sL%sT%s_%sN%i.txt' %(truncate(h,3),truncate(J,2),truncate(2L,3),truncate(1/beta_max,2),truncate(T_max,2),points)
print filename
with open('%s'%filename,'w') as output:
    dat = np.vstack([T,FE1,MAG,SUC])
    # output.write(np.array2string(dat))
    np.savetxt(output,np.transpose(dat))
    output.close()

pl.figure()
pl.xlabel('T')
pl.title('f(T)')
pl.scatter(T,FE1,s=1,marker="o")
pl.show()

pl.figure()
pl.xlabel('T')
pl.title('M(T)')
pl.scatter(T,MAG,s=1,marker="o")
pl.show()

pl.figure()
pl.xlabel('T')
pl.title('Xi(T)')
pl.scatter(T,SUC,s=1,marker="o")
pl.show()
