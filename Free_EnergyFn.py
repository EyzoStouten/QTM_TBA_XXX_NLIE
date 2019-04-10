#Calculates the free energy of the sl(2) Uimin-Sutherland model for a given temperature and magnetic field. This function is meant to be used to calculate thermodynamical quantites.

import numpy as np
import math,cmath

def Free_Energy(bet = 100.,h = 0,J = 2.,L = 100.,N = 2**14):
    ################################################################
    #initiating variables
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
    #initializing
    neg = np.empty((N,),int)
    neg[::2] = 1
    neg[1::2] = -1

    a1 = np.zeros(N,dtype=np.complex_)
    a2 = np.zeros(N,dtype=np.complex_)
    lA1 = np.zeros(N,dtype=np.complex_)
    lA2 = np.zeros(N,dtype=np.complex_)
    T11 = np.zeros(N,dtype=np.complex_)
    ################################################################
    #Initiating functions

    #Driving term
    def D(x,n):
        return (-(J*bet)*np.pi)/(np.cosh(np.pi*x))+(((-1)**(n+1))*np.full(N,bet*h/2))
    # magentic field
    def mag():
        return np.full(N,bet*h)
    #Kernels
    def K0(k):
        return np.exp(-np.abs(k)/2)/(2*np.cosh(k/2))
    def K1(k):
        return -np.exp(-k-np.abs(k)/2)/(2*np.cosh(k/2))
    def K2(k):
        return -np.exp(k-np.abs(k)/2)/(2*np.cosh(k/2,dtype=np.complex_))

    def latolA(la):
        if la.real <= 0:
            return np.log(1+np.exp(la),dtype=np.complex)
        else:
            return la+np.log(1+np.exp(-la),dtype=np.complex)

    la1=D(x,1)
    la2=D(x,2)
    for i in range(0,len(la1)):
        lA1[i]=latolA(la1[i])
        lA2[i]=latolA(la2[i])

    klA1=np.fft.fft(neg*lA1)
    klA2=np.fft.fft(neg*lA2)

    klA1=np.fft.fft(neg*lA1)
    klA2=np.fft.fft(neg*lA2)

    la1= D(x,1)+neg*(np.fft.ifft(K0(k)*klA1)+np.fft.ifft(K1(k)*klA2))
    la2= D(x,2)+neg*(np.fft.ifft(K2(k)*klA1)+np.fft.ifft(K0(k)*klA2))
    ###############################################################
    #iterative step

    print 'Computing:\n Sl(2) Uimin-Sutherland free energy\n at'
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


##################################################################
# Eigen value
    for i in range(0,len(la1)):
        lA1[i]=latolA(la1[i])
        lA2[i]=latolA(la2[i])

# Returns free energy using converged lA1 and lA2
    return J*(1-2*np.log(2))+0.*h-(1/bet)*(neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*lA1))+ neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*lA2)))[N/2]
