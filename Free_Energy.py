import numpy as np
import math,cmath
import matplotlib.pyplot as pl
import matplotlib.axes as ax
from Free_EnergyFn import Free_Energy

#Number of intervals. Scales exponentially! linspace only picks a few of
#these to calculate and draw a line.
points = 20
T_max = 1.
#as of now Free_Energy quits after beta>900 for zero magnetic fields take care!
# It does not like increasing magnetic field at all. So keep beta<100 there for h>1
beta_max = 1000.
T = np.linspace(1/beta_max,T_max,points)
beta = 1/T
h=0.
J=2.
L=50
FE1 = np.zeros(points,float)

#Calculate free energy for several beta at points
#Calls Free_EnergyFn file
for i in range(points):
    #Free_Energy : vars : Free_Energy(bet = 100.,h = 0,J = 2.,L = 100.,N = 2**14):
    #Last N is Trotter number!
    FE1[i] = Free_Energy(beta[i],h,J,L)

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return ','.join([i, (d+'0'*n)[:n]])

filename='Su2Feh%sJ%sL%sT%s_%sN%i.txt' %(truncate(h,3),truncate(J,2),truncate(2L,3),truncate(1/beta_max,2),truncate(T_max,2),points)
print filename
with open('%s'%filename,'w') as output:
    dat =np.vstack([T,FE1])
    # output.write(np.array2string(dat))
    np.savetxt(output,dat)
    output.close()

pl.figure()
pl.xlabel('T')
pl.title('f(T)')
pl.scatter(T,FE1,s=1,marker="o")
pl.show()
