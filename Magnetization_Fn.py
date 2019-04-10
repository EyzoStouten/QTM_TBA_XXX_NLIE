#Calculates the free energy of the sl(2) Uimin-Sutherland model for a given temperature and magnetic field. This function is meant to be used to calculate thermodynamical quantites.

import numpy as np
import matplotlib.pyplot as pl

import math,cmath

def Magnetization(bet = 100.,h = 0,J = 1.,L = 100.,N = 2**15):
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

    def latolA(la):
        if la.real <= 0:
            return np.log(1+np.exp(la),dtype=np.complex)
        else:
            return la+np.log(1+np.exp(-la),dtype=np.complex)

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

    print 'Computing:\n Sl(2) Uimin-Sutherland auxiliary functions'
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

        conv1=bool(max(err1)>10e-11)
        conv2=bool(max(err2)>10e-11)

        step+=1

    print 'After %i steps' %step
    print 'converged to within %s' %str(max(err1))

    print 'Computing:\n Sl(2) Uimin-Sutherland auxiliary function derivatives wrt h'

    hla1 = -np.full(N,bet/2)
    hla2 = np.full(N,bet/2)

    hklA1=np.fft.fft(neg*np.exp(la1-lA1)* hla1)
    hklA2=np.fft.fft(neg*np.exp(la2-lA2)* hla2)


    hla1 = -bet/2 - neg*(np.fft.ifft(K0(k)*hklA1)+np.fft.ifft(K1(k)*hklA2))
    hla2 =  bet/2 - neg*(np.fft.ifft(K2(k)*hklA1)+np.fft.ifft(K0(k)*hklA2))

    # hklA1=np.fft.fft(neg*np.exp(la1-lA1)* hla1)
    # hklA2=np.fft.fft(neg*np.exp(la1-lA1)* hla1)
    #
    # hla1=np.vstack([hla1, -bet/2 - neg*(np.fft.ifft(K0(k)*hklA1)+np.fft.ifft(K1(k)*hklA2))])
    # hla2=np.vstack([hla2, bet/2 - neg*(np.fft.ifft(K2(k)*hklA1)+np.fft.ifft(K0(k)*hklA2))])
    #
    # hla1[0]=np.sum(hla1,axis=0)/2
    # hla2[0]=np.sum(hla2,axis=0)/2

    err1=err2=0.
    conv1=conv2=True
    step = 0

    # pl.figure()
    # pl.xlabel('k')
    # pl.ylabel('K2*hlA2_0')
    # pl.title('K2*hlA2_0(k) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
    # pl.scatter(k,neg*(np.fft.ifft(K2(k)*hklA1)).real,s=0.7,marker="o",label='Re hlA2')
    # pl.scatter(k,neg*(np.fft.ifft(K2(k)*hklA1)).imag,s=0.7,marker="o",label='Im hlA2')
    # pl.legend(loc='center right')
    # pl.show()
    #
    # pl.figure()
    # pl.xlabel('2L')
    # pl.ylabel('hlA1_0')
    # pl.title('hlA1_0(x) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
    # pl.scatter(x,(np.exp(la1-lA1)*(-np.full(N,bet/2))).real,s=0.7,marker="o",label='Re hlA1')
    # pl.scatter(x,(np.exp(la1-lA1)*(-np.full(N,bet/2))).imag,s=0.7,marker="o",label='Im hlA1')
    # pl.legend(loc='center right')
    # pl.show()
    #
    # pl.figure()
    # pl.xlabel('2L')
    # pl.ylabel('hlA2_0')
    # pl.title('hlA2_0(x) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
    # pl.scatter(x,(np.exp(la2-lA2)* (np.full(N,bet/2))).real,s=0.7,marker="o",label='Re hlA2')
    # pl.scatter(x,(np.exp(la2-lA2)* (np.full(N,bet/2))).imag,s=0.7,marker="o",label='Im hlA2')
    # pl.legend(loc='center right')
    # pl.show()
    while conv1 and conv2 and step<200:
        olda1=hla1
        olda2=hla2

        hklA1=np.fft.fft(neg*np.exp(la1-lA1)* hla1)
        hklA2=np.fft.fft(neg*np.exp(la2-lA2)* hla2)

        hla1= -bet/2 + neg * (np.fft.ifft(K0(k)*hklA1)+np.fft.ifft(K1(k)*hklA2))
        hla2= bet/2 + neg * (np.fft.ifft(K2(k)*hklA1)+np.fft.ifft(K0(k)*hklA2))

        err1 =abs(abs(olda1)-abs(hla1))
        err2 =abs(abs(olda2)-abs(hla2))
        # olda1=hla1[0]
        # olda2=hla2[0]
        #
        # hlA1 = np.exp(la1-lA1)* hla1[0]
        # hlA2 = np.exp(la2-lA2)* hla2[0]
        #
        # hklA1=np.fft.fft(neg*hlA1)
        # hklA2=np.fft.fft(neg*hlA2)
        #
        # hla1[1]= -bet/2 + neg * (np.fft.ifft(K0(k)*hklA1)+np.fft.ifft(K1(k)*hklA2))
        # hla2[1]= bet/2 + neg * (np.fft.ifft(K2(k)*hklA1)+np.fft.ifft(K0(k)*hklA2))
        #
        # hla1[0]=np.sum(hla1,axis=0)/2
        # hla2[0]=np.sum(hla2,axis=0)/2
        #
        # err1 =abs(abs(olda1)-abs(hla1[0]))
        # err2 =abs(abs(olda2)-abs(hla2[0]))

        conv1=bool(max(err1)>10e-15)
        conv2=bool(max(err2)>10e-15)

        step+=1

    # pl.figure()
    # pl.xlabel('x')
    # pl.title('hla1(x) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
    # pl.scatter(x,hla1.real,s=1,marker="o",label='re(hla1)')
    # pl.scatter(x,hla1.imag,s=1,marker="o",label='im(hla1)')
    # pl.show()
    #
    # pl.figure()
    # pl.xlabel('x')
    # pl.title('hla2(x) \n after %i steps \n beta=%f, h=%f, J=%f ,N=%i' %(step,bet,h,J,N))
    # pl.scatter(x,hla2.real,s=1,marker="o",label='re(hla2)')
    # pl.scatter(x,hla2.imag,s=1,marker="o",label='im(hla2)')
    # pl.show()

    print 'After %i steps' %step
    print 'converged to within %s' %str(max(err1))

    print 'Computing:\n Sl(2) Uimin-Sutherland auxiliary function second derivatives wrt h'

    hhla1 = hhla2 = np.zeros(N)

    hhlA1 = np.exp(la1-2*lA1) * (hla1**2) + np.exp(la1-lA1) * hhla1
    hhlA2 = np.exp(la2-2*lA2) * (hla2**2) + np.exp(la2-lA2) * hhla1

    hhklA1=np.fft.fft(neg*hhlA1)
    hhklA2=np.fft.fft(neg*hhlA2)

    hhla1 = - neg*(np.fft.ifft(K0(k)*hhklA1) + np.fft.ifft(K1(k)*hhklA2))
    hhla2 = - neg*(np.fft.ifft(K2(k)*hhklA1) + np.fft.ifft(K0(k)*hhklA2))

    err1=err2=0.
    conv1=conv2=True
    step = 0

    while conv1 and conv2 and step<200:
        olda1=hhla1
        olda2=hhla2

        hhlA1 = np.exp(la1-2*lA1) * (hla1**2) + np.exp(la1-lA1) * hhla1
        hhlA2 = np.exp(la2-2*lA2) * (hla2**2) + np.exp(la2-lA2) * hhla1
        # print max(np.abs(np.exp(la1-lA1)))
        hhklA1=np.fft.fft(neg*hhlA1)
        hhklA2=np.fft.fft(neg*hhlA2)

        hhla1 = - neg*(np.fft.ifft(K0(k)*hhklA1)+np.fft.ifft(K1(k)*hhklA2))
        hhla2 = - neg*(np.fft.ifft(K2(k)*hhklA1)+np.fft.ifft(K0(k)*hhklA2))

        err1 =abs(abs(olda1)-abs(hhla1))
        err2 =abs(abs(olda2)-abs(hhla2))

        conv1=bool(max(err1)>10e-11)
        conv2=bool(max(err2)>10e-11)

        step+=1

    print 'After %i steps' %step
    print 'converged to within %s' %str(max(err1))

##################################################################
# Eigen value
    for i in range(0,len(la1)):
        lA1[i]=latolA(la1[i])
        lA2[i]=latolA(la2[i])

    hlA1 = np.exp(la1-lA1)* hla1
    hlA2 = np.exp(la2-lA2)* hla2

    hhlA1 = np.exp(la1-2*lA1) * (hla1**2) + np.exp(la1-lA1) * hhla1
    hhlA2 = np.exp(la2-2*lA2) * (hla2**2) + np.exp(la1-lA1) * hhla1

    f = J*(1-2*np.log(2))+0.*h-(1/bet)*(neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*lA1))+ neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*lA2)))[N/2]
    M = 1/bet *(neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*hlA1))+ neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*hlA2)))[N/2]
    X = 1/bet *(neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*hhlA1))+ neg*np.fft.ifft((1/(2*np.cosh(k/2)))*np.fft.fft(neg*hhlA2)))[N/2]

    return (f,M,X)
