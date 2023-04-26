#
# Solutions of Burgers' equations
#
# Copyright (c) 2023
# Somrath Kanoksirirath <somrath.kan@ncr.nstda.or.th>
# All rights reserved under BSD 3-clause license.
#

from math import erf, erfc, sqrt, exp
import scipy.special as sp
import numpy as np


def analyticSol_IC1_Burger_DimLess(x, t, nu, L=1., R=-1.):
    '''
    Solution of Burgers' equation to step initial condition
    '''

    erf_R = lambda x,t : erfc(-(x-R*t)/2/sqrt(nu*t))
    erf_L = lambda x,t : erfc( (x-L*t)/2/sqrt(nu*t))

    phi = np.zeros_like(x)
    mid = 0.5*(L-R)*t
    for i in range(len(phi)):
        ### Uncomment below to run at very small t and nu
        #if x[i] < mid - 50 :
        #    phi[i] = L
        #elif x[i] > mid + 50 :
        #    phi[i] = R
        #else:
        phi[i] = (R*erf_R(x[i],t)*exp(0.5*(L-R)*(x[i]-0.5*(R+L)*t)/nu) + L*erf_L(x[i],t))/(erf_R(x[i],t)*exp(0.5*(L-R)*(x[i]-0.5*(R+L)*t)/nu) + erf_L(x[i],t))

    return phi


def analyticSol_IC2_Burger_DimLess(x, t, nu):
    '''
    Decaying kink solution of Burgers' equation ((x/t)/(1 + sqrt(t)*exp(x**2/4/t)))
    '''
    t0 = np.exp(1/8/nu)
    return (x/t)/(1 + sqrt(nu*t/t0)*np.exp(x**2/4/nu/t))


def gaussian_quadrature_7p(x0, lx, func, param):
    # Using 7-point gaussian quadrature + Single step [Current]
    result = 0.
    result += 0.129484966168869693*( func(x0+lx*0.0254460438286207375, param) + func(x0+lx*0.9745539561713792625, param) )
    result += 0.279705391489276668*( func(x0+lx*0.12923440720030278  , param) + func(x0+lx*0.87076559279969722  , param) )
    result += 0.381830050505118945*( func(x0+lx*0.2970774243113014165, param) + func(x0+lx*0.7029225756886985835, param) )
    result += 0.417959183673469388*func(x0+lx*0.5, param)

    return result*0.5*lx


def analyticSol_IC3_Burger_DimLess(x, t, nu, N=10):
    '''
    Solution of Burgers' equation with sin(np.pi*x)/nu (dimensionless) as initial condition with Periodic BC.
    '''

    def CN(n, nu):
        func = lambda x, p : np.exp((np.cos(np.pi*x) - 1.)/(2*np.pi*p[1]))*np.cos(p[0]*np.pi*x)
        param = [n, nu]

        result = 0.
        m = 500     # Should be large when nu is small ???
        xl = 0.
        l = 1./m
        for i in range(m):
            result += gaussian_quadrature_7p(xl, l, func, param)
            xl += l

        return result

    numeri = np.zeros_like(x)
    denomi = np.zeros_like(x) + CN(0,nu)
    for n in range(1,N):
        a = CN(n, nu)
        numeri += n*a*exp(-n**2*np.pi**2*nu*t)*np.sin(n*np.pi*x)
        denomi += 2*a*exp(-n**2*np.pi**2*nu*t)*np.cos(n*np.pi*x)

    return 4*nu*np.pi*numeri/denomi


def burger_testcase(num, sx=2, st=5):
    # [K, t0, dt, nt], [n, xL, xR], XXX_BC
    nx = sx+1
    nt = st
    if num== 100 :
        xL = -7.5
        xR = 22.5
        dt = 0.04/st
        t0 = 0.001
        nu = 1.0
        BC = 'InfiniteBC'
        Ana_func = lambda x,t,nu : analyticSol_IC1_Burger_DimLess(x, t, nu, L=1, R=0)
        IC_func = Ana_func
    elif num==102 :
        xL = -10.
        xR = 70.
        dt = 0.1/st
        t0 = 0.5
        nu = 1.0
        BC = 'InfiniteBC'
        Ana_func = lambda x,t,nu : analyticSol_IC1_Burger_DimLess(x, t, nu, L=1, R=0)
        IC_func = Ana_func
    elif num==103 :
        xL = -15.
        xR = 420.
        dt = 0.8/st
        t0 = 10.
        nu = 1.0
        BC = 'InfiniteBC'
        Ana_func = lambda x,t,nu : analyticSol_IC1_Burger_DimLess(x,t, nu, L=1, R=0)
        IC_func = Ana_func
    elif num==200 :
        xL = 0.
        xR = 1.2
        dt = 0.7/st
        t0 = 1.0
        nu = 0.005
        BC = 'ZeroPointBC'
        Ana_func = lambda x,t,nu : analyticSol_IC2_Burger_DimLess(x, t, nu)
        IC_func = Ana_func
    elif num==300 :
        xL = 0.
        xR = 2.0
        dt = 0.1/st
        t0 = 0.0
        nu = 0.01
        BC = 'PeriodicBC'
        Ana_func = lambda x,t,nu : analyticSol_IC3_Burger_DimLess(x, t, nu, N=50)
        IC_func = lambda x,t,nu : np.sin(np.pi*x)
    elif num==400 :
        xL = -7.5
        xR = 22.5
        dt = 0.04/st
        t0 = 0.0
        nu = 1.0
        BC = 'InfiniteBC'
        Ana_func = lambda x,t,nu : 0.5*(1.0 - np.tanh(x - 0.5*t)) # For the eq. with additional source term and with nu = 1.0 !!!
        IC_func = Ana_func

    elif num== -100 :
        xL = -7.5
        xR = 22.5
        dt = 0.5/st
        t0 = 0.24
        nu = 1.0
        BC = 'InfiniteBC'
        Ana_func = lambda x,t,nu : analyticSol_IC1_Burger_DimLess(x, t, nu, L=1, R=0)
        IC_func = Ana_func
    elif num== -200 :
        xL = 0.
        xR = 1.2
        dt = 0.7/st
        t0 = 1.0
        nu = 0.005
        BC = 'ZeroPointBC'
        Ana_func = lambda x,t,nu : analyticSol_IC2_Burger_DimLess(x, t, nu=nu)
        IC_func = Ana_func
    elif num== -300 :
        xL = 0.
        xR = 2.0
        dt = 0.1/st
        t0 = 0.0
        nu = 0.1
        BC = 'PeriodicBC'
        Ana_func = lambda x,t,nu : analyticSol_IC3_Burger_DimLess(x, t, nu=nu, N=50)
        IC_func = Ana_func


    return nx, xL, xR, nt, dt, t0, nu, BC, Ana_func, IC_func



