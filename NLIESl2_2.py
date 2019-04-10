import numpy as np
import math,cmath
import matplotlib.pyplot as pl
import matplotlib.axes as ax
# from scipy.fftpack import fft, np.fft.ifft
#from matplotlib2tikz import save as tikz_save #Saves to a .tex figure

################################################################
#initiating variables
N = 2**15 # number of sites or trotter number
J = 1
L = 50.
bet = 1000
h = 0# J*np.pi+1
trot = -(J*bet)/float(N) # Trotter inhomogenieties
#x-axis
dx = 2.*L/N
x = np.zeros(N,float)
for i in range(N):
    x[i] = dx*(i-N/2)
#k-axis
K = np.pi*N/(2.*L)
dk = 2.*K/float(N)
k = np.zeros(N,float)
for i in range(N):
    k[i] = dk*(i-N/2)
#initializing alternating matrix for convoution
neg = np.empty((N,),int)
neg[::2] = 1
neg[1::2] = -1
print neg
################################################################
#Initiating functions

#Driving term
def D(x,n):
    return (-(J*bet)*np.pi)/(np.cosh(np.pi*x))+(((-1)**(n+1))*np.full(N,bet*h/2))

#Kernels
def K0(k):
    return np.exp(-np.abs(k)/2)/(2*np.cosh(k/2))
def K1(k):
    return -np.exp(-k-np.abs(k)/2)/(2*np.cosh(k/2))
def K2(k):
    return -np.exp(k-np.abs(k)/2)/(2*np.cosh(k/2,dtype=np.complex_))
# def log(1+a)=log A
def latolA(la):
    if la.real <= 0:
        return np.log(1+np.exp(la),dtype=np.complex)
    else:
        return la+np.log(1+np.exp(-la),dtype=np.complex)
#intital value
la1=D(x,1)
la2=D(x,2)

lA1 = np.zeros(N,dtype=np.complex_)
lA2 = np.zeros(N,dtype=np.complex_)

for i in range(0,len(la1)):
    lA1[i]=latolA(la1[i])
    lA2[i]=latolA(la2[i])

klA1=np.fft.fft(neg*lA1)
klA2=np.fft.fft(neg*lA2)

la1= D(x,1)+neg*(np.fft.ifft(K0(k)*klA1)+np.fft.ifft(K1(k)*klA2))
la2= D(x,2)+neg*(np.fft.ifft(K2(k)*klA1)+np.fft.ifft(K0(k)*klA2))


###############################################################
#iterative step
print 'Computing:\n Sl(2) Uimin-Sutherland equations\n at'
print 'beta= %f' %bet
print 'J= %f' %J
print 'h= %f' %h
print 'N= %i' %N
print 'L= %i' %L

err1=err2=0.
conv1=conv2=True
step = 0
while conv1 and conv2 and step<200:
    olda1=la1
    olda2=la2

    for i in range(0,len(la1)):
        lA1[i]=latolA(la1[i])
        lA2[i]=latolA(la2[i])

    klA1=np.fft.fft(neg*lA1)
    klA2=np.fft.fft(neg*lA2)

    la1= D(x,1)+neg*(np.fft.ifft(K0(k)*klA1)+np.fft.ifft(K1(k)*klA2))
    la2= D(x,2)+neg*(np.fft.ifft(K2(k)*klA1)+np.fft.ifft(K0(k)*klA2))

    err1 =abs(abs(olda1)-abs(la1))
    err2 =abs(abs(olda2)-abs(la2))

    conv1=bool(max(err1)>10e-10)
    conv2=bool(max(err2)>10e-10)

    step+=1

print 'After %i steps' %step
print 'converged to within %s' %str(max(err1))

print lA1
##################################################################
# Eigen value

for i in range(0,len(la1)):
    lA1[i]=latolA(la1[i])
    lA2[i]=latolA(la2[i])

T11 =J*(1-2*np.log(2))+0.*h-(1/bet)*(neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*lA1))+ neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*lA2)))

pl.figure()
pl.xlabel('k')
pl.ylabel('K0')
pl.title('K0(k) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
pl.scatter(k,K0(k).real,s=0.7,marker="o",label='Re a2')
pl.scatter(k,K0(k).imag,s=0.7,marker="o",label='Im a2')
pl.legend(loc='center right')
pl.show()

pl.figure()
pl.xlabel('k')
pl.ylabel('K1')
pl.title('K1(k) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
pl.scatter(k,K1(k).real,s=0.7,marker="o",label='Re a2')
pl.scatter(k,K1(k).imag,s=0.7,marker="o",label='Im a2')
pl.legend(loc='center right')
pl.show()

pl.figure()
pl.xlabel('k')
pl.ylabel('K2')
pl.title('K2(k) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
pl.scatter(k,K2(k).real,s=0.7,marker="o",label='Re a2')
pl.scatter(k,K2(k).imag,s=0.7,marker="o",label='Im a2')
pl.legend(loc='center right')
pl.show()
# pl.figure()
# pl.xlabel('2L')
# pl.ylabel('la1')
# pl.title('la1 \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
# pl.scatter(x,la1.real,s=1,marker="o",label='Re a1')
# pl.scatter(x,la1.imag,s=1,marker="o",label='Im a1')
# pl.axhline(y=bet*h,linestyle='dashed',linewidth=0.75,color='black',label='Exp(beta h)')
# pl.axvline(x=(-2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.axvline(x=(2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.legend(loc='lower left')
# pl.show()
# pl.figure()
# pl.xlabel('2L')
# pl.ylabel('la2')
# pl.title('la2 \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
# pl.scatter(x,la2.real,s=0.7,marker="o",label='Re a2')
# pl.scatter(x,la2.imag,s=0.7,marker="o",label='Im a2')
# pl.axhline(y=-bet*h,xmin=-100,xmax=100,linestyle='dashed',linewidth=0.75,color='black',label='Exp(-beta h)')
# pl.axvline(x=(-2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.axvline(x=(2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.legend(loc='center right')
# pl.show()
#
# pl.figure()
# pl.xlabel('2L')
# pl.ylabel('lA1')
# pl.title('lA1 \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
# pl.scatter(x,(lA1).real,s=0.7,marker="o",label='Re a2')
# pl.scatter(x,(lA1).imag,s=0.7,marker="o",label='Im a2')
# pl.axvline(x=(-2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.axvline(x=(2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.legend(loc='center right')
# pl.show()
#
# pl.figure()
# pl.xlabel('2L')
# pl.ylabel('lA2')
# pl.title('lA2 \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
# pl.scatter(x,(lA2).real,s=0.7,marker="o",label='Re a2')
# pl.scatter(x,(lA2).imag,s=0.7,marker="o",label='Im a2')
# pl.axvline(x=(-2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.axvline(x=(2/np.pi)*np.log(bet),linestyle='dashed',linewidth=0.75,color='black')
# pl.legend(loc='center right')
# pl.show()

pl.figure()
pl.xlabel('2L')
pl.title('error')
pl.ylim(0, max(1.7*err1))
pl.scatter(x,err1,s=1,marker="o")
pl.scatter(x,err2,s=1,marker="o")
pl.show()
