#
# Class: QuinticGrid_1D
# = An integral-like numerical approach using quintic interpolation
#
# Copyright (c) 2023
# Somrath Kanoksirirath <somrath.kan@ncr.nstda.or.th>
# All rights reserved under BSD 3-clause license.
#

import numpy as np
import math


class QuinticGrid_1D :
    '''
    Class: QuinticGrid_1D
    = An integral-like numerical approach using quintic interpolation
       + __init__(x0, phi0, grad_phi0=None, curv_phi0=None)
       + x()
       + x_of_index(i)
       + find_segment(pos)
       + coeff(n, is_left_origin=True)
       + values_at(pos)
       + phi_at(pos)
       + grad_phi_at(pos)
       + diffuse(n, l, d, Ddt, left_origin=True)
       + diffuse_without_commonfactors(n, l, d, Ddt, left_origin=True)
       + integral_of_phi()
       + diffuse_in_HopfColeSpace_DimLess(int_phi, n, l, d, Ddt, nTerms_TaylorSeries=5, left_origin=True, Mid_TaylorExpan=True)
       + diffuse_in_HopfColeSpace_DimLess_without_commonfactor(int_phi, n, l, d, Ddt, nTerms_TaylorSeries=5, left_origin=True, Mid_TaylorExpan=True)

       + x0
       + dx
       + phi
       + grad_phi
       + curv_phi
    '''
    def __init__(self, x0, dx, phi0, grad_phi0=None, curv_phi0=None):
        '''
        1D Quintic Grid :
        An integral-like numerical approach using quintic interpolation
        x0        = the first grid point position
        dx        = grid spacing
        phi0      = initial phi
        (Optional) grad_phi0 = initial grad_phi = d phi/dx (If None --> Numerical derivative of phi0)
        (Optional) curv_phi0 = initial curv_phi = d grad_phi/dx (If None --> Numerical derivative of grad_phi0)
        '''

        self.x0 = float(x0)
        self.dx = float(abs(dx))

        if np.any(grad_phi0) == None :
            #print('QuinticGrid_1D.__init__ :: grad_phi0 is not assigned <-- using Finite Difference (FD) of phi0')
            grad_phi0 = np.zeros_like(phi0)
            for i in range(1, len(grad_phi0)-1):
                grad_phi0[i] = (phi0[i+1] - phi0[i-1])/(2*self.dx)
            grad_phi0[0]  = ( phi0[1] - phi0[0] )/self.dx
            grad_phi0[-1] = (phi0[-1] - phi0[-2])/self.dx

        if np.any(curv_phi0) == None :
            #print('QuinticGrid_1D.__init__ :: curv_phi0 is not assigned <-- using Finite Difference (FD) of grad_phi0')
            curv_phi0 = np.zeros_like(grad_phi0)
            for i in range(1, len(curv_phi0)-1):
                curv_phi0[i] = (grad_phi0[i+1] - grad_phi0[i-1])/(2*self.dx)
            curv_phi0[0]  = ( grad_phi0[1] - grad_phi0[0] )/self.dx
            curv_phi0[-1] = (grad_phi0[-1] - grad_phi0[-2])/self.dx

        if len(phi0) != len(grad_phi0) :
            print('QuinticGrid_1D.__init__ :: INPUT ERROR!! len(phi0) and len(grad_phi0) are not equal')
            return
        if len(grad_phi0) != len(curv_phi0) :
            print('QuinticGrid_1D.__init__ :: INPUT ERROR!! len(grad_phi0) and len(curv_phi0) are not equal')
            return

        self.phi = phi0.astype(float)
        self.grad_phi = grad_phi0.astype(float)
        self.curv_phi = curv_phi0.astype(float)


    def copy(self, inverse=False):
        if inverse :
            return QuinticGrid_1D(self.x0, self.dx, -self.phi, -self.grad_phi, -self.curv_phi)
        else:
            return QuinticGrid_1D(self.x0, self.dx, self.phi, self.grad_phi, self.curv_phi)

    ### Basic function ### *******************************************

    def x(self):
        '''
        Return the array of position, x[:]
        '''
        return self.x0 + np.arange(len(self.phi), dtype=float)*self.dx


    def x_of_index(self, i):
        '''
        Return position x of index i
        '''
        if i >= len(self.phi) or i < 0 :
            raise ValueError('QuinticGrid:: Invalid index ==> Out of bound')

        return self.x0 + i*self.dx


    def find_segment(self, pos):
        '''
        Return index iL, that satisfies x[iL] <= pos < x[iL+1]
        '''
        iL = 0
        iR = len(self.phi)-1
        while iL+1 != iR :
            m = math.floor((iL+iR)/2)
            if pos < self.x0 + m*self.dx :
                iR = m
            else:
                iL = m

        return iL


    def coeff(self, n, is_left_origin=True):
        '''
        Return quintic coefficients of phi(x) where x[n] < x < x[n+1]
        *** Locally, from [0, dx] ***
        That is, phi(x) = phi(x[n]+y) = a*y**5 + b*y**4 + c*y**3 + d*y**2 + e*y + f

        If is_left_origin = False, return that of x[n+1] > x > x[n]
        '''

        if is_left_origin :
            L = self.phi[n]
            R = self.phi[n+1]
            grad_L = self.grad_phi[n]
            grad_R = self.grad_phi[n+1]
            curv_L = self.curv_phi[n]
            curv_R = self.curv_phi[n+1]
        else:
            L = self.phi[n+1]
            R = self.phi[n]
            grad_L = self.grad_phi[n+1]
            grad_R = self.grad_phi[n]
            curv_L = self.curv_phi[n+1]
            curv_R = self.curv_phi[n]

        return self._quintic_coeff(L, R, grad_L, grad_R, curv_L, curv_R, self.dx)


    def _quintic_coeff(self, L, R, grad_L, grad_R, curv_L, curv_R, dx):
        '''
        Return quintic coefficients of phi(x) where x[n] < x < x[n+1]
        *** Locally, from [0, dx] ***
        That is, phi(x) = phi(x[n]+y) = a*y**5 + b*y**4 + c*y**3 + d*y**2 + e*y + f
        '''

        f = L
        e = grad_L
        d = curv_L/2
        a = (  6/dx**5)*( R-(d*dx**2+e*dx+f) ) + (-3/dx**4)*( grad_R-(curv_L*dx+e) ) + (0.5/dx**3)*( curv_R-curv_L )
        b = (-15/dx**4)*( R-(d*dx**2+e*dx+f) ) + ( 7/dx**3)*( grad_R-(curv_L*dx+e) ) + ( -1/dx**2)*( curv_R-curv_L )
        c = ( 10/dx**3)*( R-(d*dx**2+e*dx+f) ) + (-4/dx**2)*( grad_R-(curv_L*dx+e) ) + (0.5/dx   )*( curv_R-curv_L )
        # f(x) = a*x**5 + b*x**4 + c*x**4 + d*x**2 + e*x + f

        return a, b, c, d, e, f


    ### Advection ### *******************************************

    def values_at(self, pos):
        '''
        Return phi, grad_phi, curv_phi at pos
        '''

        iL = self.find_segment(pos)
        c5, c4, c3, c2, c1, c0 = self.coeff(iL)

        lx = pos - (self.x0+iL*self.dx)

        phi = c5*lx**5 + c4*lx**4 + c3*lx**3 + c2*lx**2 + c1*lx + c0
        grad_phi = 5*c5*lx**4 + 4*c4*lx**3 + 3*c3*lx**2 + 2*c2*lx + c1
        curv_phi = 20*c5*lx**3 + 12*c4*lx**2 + 6*c3*lx + 2*c2

        return phi, grad_phi, curv_phi


    def phi_at(self, pos):
        '''
        Return phi at pos
        '''
        iL = self.find_segment(pos)
        c5, c4, c3, c2, c1, c0 = self.coeff(iL)

        lx = pos - (self.x0+iL*self.dx)

        return c5*lx**5 + c4*lx**4 + c3*lx**3 + c2*lx**2 + c1*lx + c0


    def grad_phi_at(self, pos):
        '''
        Return grad_phi at pos
        '''
        iL = self.find_segment(pos)
        c5, c4, c3, c2, c1, c0 = self.coeff(iL)

        lx = pos - (self.x0+iL*self.dx)

        return 5*c5*lx**4 + 4*c4*lx**3 + 3*c3*lx**2 + 2*c2*lx + c1


    def curv_phi_at(self, pos):
        '''
        Return curv_phi at pos
        '''
        iL = self.find_segment(pos)
        c5, c4, c3, c2, c1, c0 = self.coeff(iL)

        lx = pos - (self.x0+iL*self.dx)

        return 20*c5*lx**3 + 12*c4*lx**2 + 6*c3*lx + 2*c2


    ### Diffusion ### *******************************************

    def diffuse(self, n, l, d, Ddt, left_origin=True):
        '''
        Return the amount of phi, grad_phi and curv_phi that diffuse from segment x[n] < x < x[n]+l
        to a position at d distance away for Ddt = D*dt
        where d is diffusion coefficient and dt is time step size
        '''
        phi, grad_phi, curv_phi = self.diffuse_without_commonfactor(n, l, d, Ddt, left_origin=left_origin)

        return 0.5*phi/np.sqrt(np.pi*Ddt), 0.25*grad_phi/Ddt/np.sqrt(np.pi*Ddt), 0.125*curv_phi/Ddt/Ddt/np.sqrt(np.pi*Ddt)


    def diffuse_without_commonfactor(self, n, l, d, Ddt, left_origin=True):
        '''
        As CubicGrid_1D.diffuse()
        BUT without
        Sol_phi /= np.sqrt(4*np.pi*Ddt)
        Sol_grad_phi *= 0.5/Ddt /np.sqrt(4*np.pi*Ddt)
        Sol_curv_phi *= 0.25/Ddt/Ddt /np.sqrt(4*np.pi*Ddt)
        '''

        #if l > self.dx :
        #    print('QuinticGrid_1D.diffuse :: Something wrong !! l cannot be greater than x[n+1]-x[n]')
        #    return

        c5, c4, c3, c2, c1, c0 = self.coeff(n, left_origin)
        # For Phi
        cN = np.array([c0, c1, c2, c3, c4, c5])
        Sol_phi = self._integrate_PolydegNGaussian_LeftOrigin([c0, c1, c2, c3, c4, c5], d, l, 4*Ddt)
        #Sol_phi /= np.sqrt(4*np.pi*Ddt)

        # For grad_phi
        cN_x = np.zeros(len(cN)+1, dtype=float)
        cN_x[:-1] += d*cN
        cN_x[1:]  += cN
        Sol_grad_phi = self._integrate_PolydegNGaussian_LeftOrigin(cN_x, d, l, 4*Ddt)
        #Sol_grad_phi *= 0.5/Ddt /np.sqrt(4*np.pi*Ddt)

        cN_xx = np.zeros(len(cN)+2, dtype=float)
        cN_xx[:-2]  += (d**2 - 2*Ddt)*cN
        cN_xx[1:-1] += 2*d*cN
        cN_xx[2:]   += cN
        Sol_curv_phi = self._integrate_PolydegNGaussian_LeftOrigin(cN_xx, d, l, 4*Ddt)
        #Sol_curv_phi *= 0.25/Ddt/Ddt /np.sqrt(4*np.pi*Ddt)

        return Sol_phi, Sol_grad_phi, Sol_curv_phi


    def _integrate_PolydegNGaussian_LeftOrigin(self, cN, x0, l, b):
        '''
        Return the integration of polynomial (N terms) x gaussian function
        from xi to xi+l (local y from 0 to l) [Left]

        SHOULD be imporved to reduce round-off error ****
        - Try decimal, float128, ...
        - More precise function: erf, ...

        Note: cN[0] + cN[1]*y + cN[2]*y**2 + ... + cN[N-1]*y**(N-1)
        '''

        N = len(cN)
        sol = np.zeros(N)

        # 1 --- Integration of c0*exp(-(y+x0)^2/b)
        if N >= 1 :
            sol[0] = 0.5*np.sqrt(np.pi*b) * (math.erf( (x0+l)/np.sqrt(b) ) - math.erf( x0/np.sqrt(b) ))

        # y --- Integration of c1*y*exp(-(y+x0)^2/b)
        if N >= 2 :
            sol[1]  = -0.5*np.sqrt(np.pi*b)*x0 * (math.erf( (x0+l)/np.sqrt(b) ) - math.erf( x0/np.sqrt(b) ))
            sol[1] += -0.5*b * (np.exp( -(x0+l)**2/b ) - np.exp( -x0**2/b ))
        else:
            return cN[0]*sol[0]

        # y**k where k>1 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = 0 to l ONLY ***)
        for k in range(2,N):
            sol[k]  = -0.5*b*l**(k-1) * np.exp( -(x0+l)**2/b )
            sol[k] += -x0*sol[k-1]
            sol[k] +=  0.5*b*(k-1)*sol[k-2]

        return np.sum(np.multiply(cN, sol))


    ### Integration ### *******************************************

    def integral_phi(self, nL, nR):
        '''
        Return the integral of phi
        '''
        int_phi = np.zeros(len(self.phi), dtype=float)

        for i in range(nL+1, nR):
            # Sum up the next segment
            c5, c4, c3, c2, c1, c0 = self.coeff(i-1)
            int_phi[i] = (c5/6)*self.dx**6 + (c4/5)*self.dx**5  + (c3/4)*self.dx**4 + (c2/3)*self.dx**3 + (c1/2)*self.dx**2 + c0*self.dx

        # Careful !!
        # Round-off error == sum many small values
        # Parallel version should work better (see reduction) !!!
        int_phi[nL:nR] = np.cumsum(int_phi[nL:nR])

        return int_phi


    ### Nonlinear Advection + Diffusion = Burgers equation ### **************
    # MUST use at least float64 ***

    def diffuse_in_HopfColeSpace_DimLess(self, int_phi, n, l, d, Ddt, nTerms_TaylorSeries=5, left_origin=True, Mid_TaylorExpan=True, autoDiscardSmallTaylorExpan=True):
        '''
        Return the amount of P, Px, Pxx and Pxxx that diffuse from segment x[n] < x < x[n]+l
        "in the Hopf-Cole space" of phi to a position at d distance away for dt,
        with diffusion coefficient D_coef, to solve the Burger equation

        Step 1/2 = Hopf-Cole transformation of int_phi, phi, grad_phi, curv_phi
        - int_phi  = Input <-- to vary its range using +C
        - phi      = Internal of QuinticGrid_1D
        - grad_phi = Internal of QuinticGrid_1D
        - curv_phi = Internal of QuinticGrid_1D

        Step 2/2 = Diffusion using septic spline interpolation and Taylor expansion of exp(...) at x[n]+l/2
        and return the contribution of phi(x) where x[n] < x < x[n]+l --> P, Px, Pxx and Pxxx

        Note: For Burger equation,
        --> phi_n+1 = -2*D_coef* Px/P,
        --> grad_phi_n+1 = -2*D_coef* (P*Pxx - Px*Px)/P/P
        --> curv_phi_n+1 = -2*D_coef* (P**2 * Pxxx + 2*Px**3 - 3*P*Px*Pxx)/P/P/P
        Note 2: When autoDiscardSmallTaylorExpan = True, nTerms_TaylorSeries --> maximum limit
        '''

        P, Px, Pxx, Pxxx = self.diffuse_in_HopfColeSpace_DimLess_without_commonfactor(int_phi, n, l, d, Ddt, nTerms_TaylorSeries, left_origin, Mid_TaylorExpan, autoDiscardSmallTaylorExpan)
        P *= 0.5/np.sqrt(np.pi*Ddt)
        Px   *= 0.25/Ddt/np.sqrt(np.pi*Ddt)
        Pxx  *= 0.125/Ddt/Ddt/np.sqrt(np.pi*Ddt)
        Pxxx *= 0.0625/Ddt**3/np.sqrt(np.pi*Ddt)

        return P, Px, Pxx


    def diffuse_in_HopfColeSpace_DimLess_without_commonfactor(self, int_phi, n, l, d, Ddt, nTerms_TaylorSeries=5, left_origin=True, Mid_TaylorExpan=True, isAuto=True):
        '''
        As CubicGrid_1D.diffuse_in_HopfColeSpace_DimLess()
        BUT without
        P    /= np.sqrt(4*np.pi*D_coef*dt)
        Px   *= 1/(2*D_coef*dt)    /np.sqrt(4*np.pi*D_coef*dt)
        Pxx  *= 1/(2*D_coef*dt)**2 /np.sqrt(4*np.pi*D_coef*dt)
        Pxxx *= 1/(2*D_coef*dt)**3 /np.sqrt(4*np.pi*D_coef*dt)

        Note: D_coef is removed by rescale phi, x and t coordinate
        '''

        #if l > self.dx :
        #    print('QuinticGrid_1D.diffuse_in_HopfColeSpace :: Something wrong !! l cannot be greater than x[n+1]-x[n]')
        #    return

        nTerms_TaylorSeries = math.ceil(nTerms_TaylorSeries)
        if nTerms_TaylorSeries < 1 :
            print('QuinticGrid_1D.diffuse_in_HopfColeSpace :: nTerms_TaylorSeries must be >= 1 --> will use 1')
            nTerms_TaylorSeries = 1
        if nTerms_TaylorSeries > 16 :
            print('QuinticGrid_1D.diffuse_in_HopfColeSpace :: I am sorry, the current implementation cannot have nTerms_TaylorSeries > 16')
            return

        # Quintic spline interpolation of int_phi
        if left_origin :
            L = int_phi[n]
            R = int_phi[n+1]
            grad_L = self.phi[n]
            grad_R = self.phi[n+1]
            curv_L = self.grad_phi[n]
            curv_R = self.grad_phi[n+1]
            d_curv_L = self.curv_phi[n]
            d_curv_R = self.curv_phi[n+1]
        else:
            L = int_phi[n+1]
            R = int_phi[n]
            grad_L = self.phi[n+1]
            grad_R = self.phi[n]
            curv_L = self.grad_phi[n+1]
            curv_R = self.grad_phi[n]
            d_curv_L = self.curv_phi[n+1]
            d_curv_R = self.curv_phi[n]
        c7, c6, c5, c4, c3, c2, c1, c0 = self._septic_coeff(L, R, grad_L, grad_R, curv_L, curv_R, d_curv_L, d_curv_R, self.dx)

        # -----------------------------------------------------

        # If Mid_TaylorExpan :
        # Taylor expansion of exp(-0.5*int_phi/D_coef) at local y_old=l/2 (x=x[n])
        #                                              == local y_new=0 (x=x[n]+l/2)
        #
        # Calculate new coefficients for y_new where the transformation is y_new = y_old - l/2
        # where int_phi = c7*y_old**7 + c6*y_old**6 + c5*y_old**5 + c4*y_old**4 + c3*y_old**3 + c2*y_old**2 + c1*y_old + c0 = f_old(y_old)
        # f_old(y_old) = f_new(y_new) && f_old(y_old) = f_old(y_new+l/2)  --> f_old(y_new+l/2) = f_new(y_new)
        #
        if Mid_TaylorExpan :
            c_int_phi = [c0 +   c1*(l/2) +    c2*(l/2)**2 +    c3*(l/2)**3 +    c4*(l/2)**4 +    c5*(l/2)**5 +   c6*(l/2)**6 + c7*(l/2)**7,
                         c1 + 2*c2*(l/2) +  3*c3*(l/2)**2 +  4*c4*(l/2)**3 +  5*c5*(l/2)**4 +  6*c6*(l/2)**5 + 7*c7*(l/2)**6,
                         c2 + 3*c3*(l/2) +  6*c4*(l/2)**2 + 10*c5*(l/2)**3 + 15*c6*(l/2)**4 + 21*c7*(l/2)**5,
                         c3 + 4*c4*(l/2) + 10*c5*(l/2)**2 + 20*c6*(l/2)**3 + 35*c7*(l/2)**4,
                         c4 + 5*c5*(l/2) + 15*c6*(l/2)**2 + 35*c7*(l/2)**3,
                         c5 + 6*c6*(l/2) + 21*c7*(l/2)**2,
                         c6 + 7*c7*(l/2),
                         c7 ]
        else:
            c_int_phi = [c0, c1, c2, c3, c4, c5, c6, c7]

        # Define Z = c5*y**5 + c4*y**4 + c3*y**3 + c2*y**2 + c1*y where y = y_new = -l/2 to l/2
        # Therefore, exp(-0.5*int_phi/D_coef) = Const * exp(-0.5*Z/D_coef)
        #
        # Needed, as each contribution will get sum up, Const will not be cancelled out
        if Mid_TaylorExpan :
            Const = math.exp( -0.5*(c0 + c1*(l/2) + c2*(l/2)**2 + c3*(l/2)**3 + c4*(l/2)**4 + c5*(l/2)**5 + c6*(l/2)**6 + c7*(l/2)**7) )
        else:
            Const = math.exp( -0.5*c_int_phi[0] )
        # --> Multiply at the last step --> Less numerical error

        # Taylor expansion of exp(-0.5*Z/D_coef ) where Z is a polynomial degree 5
        c_int_phi = c_int_phi[1:]
        cN = np.zeros(nTerms_TaylorSeries, dtype=float)
        n_cN = len(cN)
        # x**0
        cN[0] = 1.0
        if nTerms_TaylorSeries > 1 : # x**1
            cN[1] += -0.5*c_int_phi[0]
            # == cN[0] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,0,0,0], 1, c_int_phi)
        if nTerms_TaylorSeries > 2 : # x**2
            cN[2] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,0,0,0], 2, c_int_phi)
            cN[2] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,0,0,0], 1, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 2, cN[2], l)
        if nTerms_TaylorSeries > 3 and n_cN >= 3 : # x**3
            cN[3] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,0,0,0], 3, c_int_phi)
            cN[3] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,0,0,0], 2, c_int_phi)
            cN[3] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,0,0,0], 1, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 3, cN[3], l)
        if nTerms_TaylorSeries > 4 and n_cN >= 4 : # x**4
            cN[4] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,0,0,0,0], 4, c_int_phi)
            cN[4] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,0,0,0,0], 3, c_int_phi)
            cN[4] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,0,0,0,0], 2, c_int_phi)
            cN[4] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,0,0,0,0], 2, c_int_phi)
            cN[4] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,1,0,0,0], 1, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 4, cN[4], l)
        if nTerms_TaylorSeries > 5 and n_cN >= 5 : # x**5
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,0,0,0,0], 5, c_int_phi)
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,0,0,0,0], 4, c_int_phi)
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,0,0,0,0], 3, c_int_phi)
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,0,0,0,0], 3, c_int_phi)
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,0,0,0,0], 2, c_int_phi)
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,1,0,0,0], 2, c_int_phi)
            cN[5] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,1,0,0], 1, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 5, cN[5], l)
        if nTerms_TaylorSeries > 6 and n_cN >= 6 : # x**6
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,0,0,0,0], 6, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,0,0,0,0], 5, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,0,0,0,0], 4, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,0,0,0,0], 3, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,0,0,0,0], 4, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,0,0,0,0], 3, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,0,0,0,0], 2, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,1,0,0,0], 3, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,1,0,0,0], 2, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,1,0,0], 2, c_int_phi)
            cN[6] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,0,1,0], 1, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 6, cN[6], l)
        if nTerms_TaylorSeries > 7 and n_cN >= 7 : # x**7
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([7,0,0,0,0,0,0], 7, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([5,1,0,0,0,0,0], 6, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([3,2,0,0,0,0,0], 5, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([1,3,0,0,0,0,0], 4, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([4,0,1,0,0,0,0], 5, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([2,1,1,0,0,0,0], 4, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([0,2,1,0,0,0,0], 3, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([1,0,2,0,0,0,0], 3, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,1,0,0,0], 4, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,1,0,0,0], 3, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,1,0,0,0], 2, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,1,0,0], 3, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,1,0,0], 2, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,0,1,0], 2, c_int_phi)
            cN[7] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,0,0,1], 1, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 7, cN[7], l)
        if nTerms_TaylorSeries > 8 and n_cN >= 8 : # x**8
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([8,0,0,0,0,0,0], 8, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([6,1,0,0,0,0,0], 7, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([4,2,0,0,0,0,0], 6, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([2,3,0,0,0,0,0], 5, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([0,4,0,0,0,0,0], 4, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([5,0,1,0,0,0,0], 6, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([3,1,1,0,0,0,0], 5, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([1,2,1,0,0,0,0], 4, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([2,0,2,0,0,0,0], 4, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([0,1,2,0,0,0,0], 3, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,1,0,0,0], 5, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,1,0,0,0], 4, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,1,0,0,0], 3, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,1,0,0,0], 3, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,2,0,0,0], 2, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,1,0,0], 4, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,1,0,0], 3, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,1,0,0], 2, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,0,1,0], 3, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,0,1,0], 2, c_int_phi)
            cN[8] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,0,0,1], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 8, cN[8], l)
        if nTerms_TaylorSeries > 9 and n_cN >= 9 : # x**9
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([9,0,0,0,0,0,0], 9, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([7,1,0,0,0,0,0], 8, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([5,2,0,0,0,0,0], 7, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([3,3,0,0,0,0,0], 6, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([1,4,0,0,0,0,0], 5, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([6,0,1,0,0,0,0], 7, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([4,1,1,0,0,0,0], 6, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([2,2,1,0,0,0,0], 5, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,3,1,0,0,0,0], 4, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([3,0,2,0,0,0,0], 5, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([1,1,2,0,0,0,0], 4, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,0,3,0,0,0,0], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,1,0,0,0], 6, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,1,0,0,0], 5, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,1,0,0,0], 4, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,1,0,0,0], 4, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,1,0,0,0], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,2,0,0,0], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,0,1,0,0], 5, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,0,1,0,0], 4, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,0,1,0,0], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,0,1,0,0], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,1,1,0,0], 2, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,0,1,0], 4, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,0,1,0], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,0,1,0], 2, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,0,0,1], 3, c_int_phi)
            cN[9] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,0,0,1], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 9, cN[9], l)
        if nTerms_TaylorSeries > 10 and n_cN >= 10 : # x**10
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([10,0,0,0,0,0,0], 10, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([8,1,0,0,0,0,0], 9, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([6,2,0,0,0,0,0], 8, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([4,3,0,0,0,0,0], 7, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([2,4,0,0,0,0,0], 6, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,5,0,0,0,0,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([7,0,1,0,0,0,0], 8, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([5,1,1,0,0,0,0], 7, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([3,2,1,0,0,0,0], 6, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,3,1,0,0,0,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([4,0,2,0,0,0,0], 6, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([2,1,2,0,0,0,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,2,2,0,0,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,0,3,0,0,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,1,0,0,0], 7, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,1,0,0,0], 6, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,1,0,0,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,1,0,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,1,0,0,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,1,0,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,1,0,0,0], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,2,0,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,2,0,0,0], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,0,1,0,0], 6, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,0,1,0,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,0,1,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,0,1,0,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,0,1,0,0], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,1,1,0,0], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,2,0,0], 2, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,0,0,1,0], 5, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,0,0,1,0], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,0,0,1,0], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,0,0,1,0], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,1,0,1,0], 2, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,0,0,1], 4, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,0,0,1], 3, c_int_phi)
            cN[10] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,0,0,1], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 10, cN[10], l)
        if nTerms_TaylorSeries > 11 and n_cN >= 11 : # x**11
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([11,0,0,0,0,0,0], 11, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([9,1,0,0,0,0,0], 10, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([7,2,0,0,0,0,0], 9, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([5,3,0,0,0,0,0], 8, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([3,4,0,0,0,0,0], 7, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,5,0,0,0,0,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([8,0,1,0,0,0,0], 9, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([6,1,1,0,0,0,0], 8, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([4,2,1,0,0,0,0], 7, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,3,1,0,0,0,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,4,1,0,0,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([5,0,2,0,0,0,0], 7, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([3,1,2,0,0,0,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,2,2,0,0,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,0,3,0,0,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,1,3,0,0,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([7,0,0,1,0,0,0], 8, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([5,1,0,1,0,0,0], 7, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([3,2,0,1,0,0,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,3,0,1,0,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([4,0,1,1,0,0,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,1,1,1,0,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,2,1,1,0,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,0,2,1,0,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,2,0,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,2,0,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,2,0,0,0], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,0,1,0,0], 7, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,0,1,0,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,0,1,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,0,1,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,0,1,0,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,0,1,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,0,1,0,0], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,1,1,0,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,1,1,0,0], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,2,0,0], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,0,0,1,0], 6, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,0,0,1,0], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,0,0,1,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,0,0,1,0], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,0,0,1,0], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,1,0,1,0], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,1,1,0], 2, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,0,0,0,1], 5, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,0,0,0,1], 4, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,0,0,0,1], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,0,0,0,1], 3, c_int_phi)
            cN[11] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,1,0,0,1], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 11, cN[11], l)
        if nTerms_TaylorSeries > 12 and n_cN >= 12 : # x**12
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([12,0,0,0,0,0,0], 12, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([10,1,0,0,0,0,0], 11, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([8,2,0,0,0,0,0], 10, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([6,3,0,0,0,0,0], 9, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([4,4,0,0,0,0,0], 8, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,5,0,0,0,0,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,6,0,0,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([9,0,1,0,0,0,0], 10, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([7,1,1,0,0,0,0], 9, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([5,2,1,0,0,0,0], 8, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,3,1,0,0,0,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,4,1,0,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([6,0,2,0,0,0,0], 8, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([4,1,2,0,0,0,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,2,2,0,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,3,2,0,0,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,0,3,0,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,1,3,0,0,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,0,4,0,0,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([8,0,0,1,0,0,0], 9, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([6,1,0,1,0,0,0], 8, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([4,2,0,1,0,0,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,3,0,1,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,4,0,1,0,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([5,0,1,1,0,0,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,1,1,1,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,2,1,1,0,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,0,2,1,0,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,1,2,1,0,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,2,0,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,2,0,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,2,0,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,2,0,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,3,0,0,0], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([7,0,0,0,1,0,0], 8, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([5,1,0,0,1,0,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,2,0,0,1,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,3,0,0,1,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([4,0,1,0,1,0,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,1,1,0,1,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,2,1,0,1,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,0,2,0,1,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,1,1,0,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,1,1,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,1,1,0,0], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,2,0,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,2,0,0], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,0,0,1,0], 7, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,0,0,1,0], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,0,0,1,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,0,0,1,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,0,0,1,0], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,0,0,1,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,0,0,1,0], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,1,0,1,0], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,1,0,1,0], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,1,1,0], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,0,2,0], 2, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,0,0,0,1], 6, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,0,0,0,1], 5, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,0,0,0,1], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,0,0,0,1], 4, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,0,0,0,1], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,1,0,0,1], 3, c_int_phi)
            cN[12] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,1,0,1], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 12, cN[12], l)
        if nTerms_TaylorSeries > 13 and n_cN >= 13 : # x**13
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([13,0,0,0,0,0,0], 13, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([11,1,0,0,0,0,0], 12, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([9,2,0,0,0,0,0], 11, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([7,3,0,0,0,0,0], 10, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([5,4,0,0,0,0,0], 9, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,5,0,0,0,0,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,6,0,0,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([10,0,1,0,0,0,0], 11, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([8,1,1,0,0,0,0], 10, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([6,2,1,0,0,0,0], 9, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,3,1,0,0,0,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,4,1,0,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,5,1,0,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([7,0,2,0,0,0,0], 9, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([5,1,2,0,0,0,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,2,2,0,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,3,2,0,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,0,3,0,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,1,3,0,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,2,3,0,0,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,0,4,0,0,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([9,0,0,1,0,0,0], 10, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([7,1,0,1,0,0,0], 9, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([5,2,0,1,0,0,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,3,0,1,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,4,0,1,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([6,0,1,1,0,0,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,1,1,1,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,2,1,1,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,3,1,1,0,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,0,2,1,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,1,2,1,0,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,0,3,1,0,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,2,0,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,2,0,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,2,0,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,2,0,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,2,0,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,3,0,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([8,0,0,0,1,0,0], 9, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([6,1,0,0,1,0,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,2,0,0,1,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,3,0,0,1,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,4,0,0,1,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([5,0,1,0,1,0,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,1,1,0,1,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,2,1,0,1,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,0,2,0,1,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,1,2,0,1,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,1,1,0,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,1,1,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,1,1,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,1,1,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,2,1,0,0], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,2,0,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,2,0,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,2,0,0], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([7,0,0,0,0,1,0], 8, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([5,1,0,0,0,1,0], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,2,0,0,0,1,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,3,0,0,0,1,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,0,1,0,0,1,0], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,1,1,0,0,1,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,2,1,0,0,1,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,0,2,0,0,1,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,1,0,1,0], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,1,0,1,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,1,0,1,0], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,1,1,0], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,1,1,0], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,0,2,0], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,0,0,0,1], 7, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,0,0,0,1], 6, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,0,0,0,1], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,0,0,0,1], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,0,0,0,1], 5, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,0,0,0,1], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,0,0,0,1], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,1,0,0,1], 4, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,1,0,0,1], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,1,0,1], 3, c_int_phi)
            cN[13] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,0,1,1], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 13, cN[13], l)
        if nTerms_TaylorSeries > 14 and n_cN >= 14 : # x**14
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([14,0,0,0,0,0,0], 14, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([12,1,0,0,0,0,0], 13, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([10,2,0,0,0,0,0], 12, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([8,3,0,0,0,0,0], 11, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([6,4,0,0,0,0,0], 10, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,5,0,0,0,0,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,6,0,0,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,7,0,0,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([11,0,1,0,0,0,0], 12, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([9,1,1,0,0,0,0], 11, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([7,2,1,0,0,0,0], 10, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,3,1,0,0,0,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,4,1,0,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,5,1,0,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([8,0,2,0,0,0,0], 10, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([6,1,2,0,0,0,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,2,2,0,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,3,2,0,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,4,2,0,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,0,3,0,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,1,3,0,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,2,3,0,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,0,4,0,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,1,4,0,0,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([10,0,0,1,0,0,0], 11, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([8,1,0,1,0,0,0], 10, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([6,2,0,1,0,0,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,3,0,1,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,4,0,1,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,5,0,1,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([7,0,1,1,0,0,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,1,1,1,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,2,1,1,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,3,1,1,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,0,2,1,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,1,2,1,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,2,2,1,0,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,0,3,1,0,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,2,0,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,2,0,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,2,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,2,0,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,2,0,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,2,0,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,2,0,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,3,0,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,3,0,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([9,0,0,0,1,0,0], 10, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([7,1,0,0,1,0,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,2,0,0,1,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,3,0,0,1,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,4,0,0,1,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([6,0,1,0,1,0,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,1,1,0,1,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,2,1,0,1,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,3,1,0,1,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,0,2,0,1,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,1,2,0,1,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,3,0,1,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,1,1,0,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,1,1,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,1,1,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,1,1,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,1,1,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,2,1,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,0,2,0,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,0,2,0,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,0,2,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,0,2,0,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,1,2,0,0], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([8,0,0,0,0,1,0], 9, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([6,1,0,0,0,1,0], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,2,0,0,0,1,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,3,0,0,0,1,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,4,0,0,0,1,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,0,1,0,0,1,0], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,1,1,0,0,1,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,2,1,0,0,1,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,0,2,0,0,1,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,1,2,0,0,1,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,1,0,1,0], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,1,0,1,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,1,0,1,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,1,0,1,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,2,0,1,0], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,1,1,0], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,1,1,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,1,1,0], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,0,2,0], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,0,2,0], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([7,0,0,0,0,0,1], 8, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([5,1,0,0,0,0,1], 7, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,2,0,0,0,0,1], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,3,0,0,0,0,1], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([4,0,1,0,0,0,1], 6, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,1,1,0,0,0,1], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,2,1,0,0,0,1], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,0,2,0,0,0,1], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,1,0,0,1], 5, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,1,0,0,1], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,1,0,0,1], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,1,0,1], 4, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,1,0,1], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,0,1,1], 3, c_int_phi)
            cN[14] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,0,0,2], 2, c_int_phi)
            n_cN = self.is_num_cN_enough(isAuto, 14, cN[14], l)
        if nTerms_TaylorSeries > 15 and n_cN >= 15 : # x**15
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([15,0,0,0,0,0,0], 15, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([13,1,0,0,0,0,0], 14, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([11,2,0,0,0,0,0], 13, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([9,3,0,0,0,0,0], 12, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([7,4,0,0,0,0,0], 11, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,5,0,0,0,0,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,6,0,0,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,7,0,0,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([12,0,1,0,0,0,0], 13, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([10,1,1,0,0,0,0], 12, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([8,2,1,0,0,0,0], 11, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,3,1,0,0,0,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,4,1,0,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,5,1,0,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,6,1,0,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([9,0,2,0,0,0,0], 11, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([7,1,2,0,0,0,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,2,2,0,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,3,2,0,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,4,2,0,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,0,3,0,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,1,3,0,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,2,3,0,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,3,3,0,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,0,4,0,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,1,4,0,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,5,0,0,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([11,0,0,1,0,0,0], 12, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([9,1,0,1,0,0,0], 11, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([7,2,0,1,0,0,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,3,0,1,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,4,0,1,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,5,0,1,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([8,0,1,1,0,0,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,1,1,1,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,2,1,1,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,3,1,1,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,4,1,1,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,0,2,1,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,1,2,1,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,2,2,1,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,0,3,1,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,1,3,1,0,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([7,0,0,2,0,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,1,0,2,0,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,2,0,2,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,3,0,2,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,0,1,2,0,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,1,1,2,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,2,1,2,0,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,2,2,0,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,3,0,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,3,0,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,3,0,0,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([10,0,0,0,1,0,0], 11, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([8,1,0,0,1,0,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,2,0,0,1,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,3,0,0,1,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,4,0,0,1,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,5,0,0,1,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([7,0,1,0,1,0,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,1,1,0,1,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,2,1,0,1,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,3,1,0,1,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,0,2,0,1,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,1,2,0,1,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,2,2,0,1,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,3,0,1,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,0,0,1,1,0,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,1,0,1,1,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,2,0,1,1,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,3,0,1,1,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,0,1,1,1,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,1,1,1,1,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,2,1,1,0,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,2,1,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,2,1,0,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,0,2,0,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,0,2,0,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,0,2,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,0,2,0,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,0,2,0,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,1,2,0,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,0,3,0,0], 3, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([9,0,0,0,0,1,0], 10, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([7,1,0,0,0,1,0], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,2,0,0,0,1,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,3,0,0,0,1,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,4,0,0,0,1,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,0,1,0,0,1,0], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,1,1,0,0,1,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,2,1,0,0,1,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,3,1,0,0,1,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,0,2,0,0,1,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,1,2,0,0,1,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,3,0,0,1,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,0,0,1,0,1,0], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,1,0,1,0,1,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,2,0,1,0,1,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,0,1,1,0,1,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,1,1,1,0,1,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,2,0,1,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,0,1,1,0], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,0,1,1,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,0,1,1,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,0,1,1,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,1,1,1,0], 3, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,0,2,0], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,0,2,0], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,0,2,0], 3, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([8,0,0,0,0,0,1], 9, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([6,1,0,0,0,0,1], 8, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,2,0,0,0,0,1], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,3,0,0,0,0,1], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,4,0,0,0,0,1], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([5,0,1,0,0,0,1], 7, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,1,1,0,0,0,1], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,2,1,0,0,0,1], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,0,2,0,0,0,1], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,1,2,0,0,0,1], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([4,0,0,1,0,0,1], 6, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,1,0,1,0,0,1], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,2,0,1,0,0,1], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,1,1,0,0,1], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,0,2,0,0,1], 3, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([3,0,0,0,1,0,1], 5, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,1,0,0,1,0,1], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,0,1,0,1,0,1], 3, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([2,0,0,0,0,1,1], 4, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([0,1,0,0,0,1,1], 3, c_int_phi)
            cN[15] += self.__cal_TaylorExpCoef_HopfCole([1,0,0,0,0,0,2], 3, c_int_phi)
        # from the find_combination_to_N(N) function at the bottom of this file

        cN = cN[:n_cN+1]
        # -----------------------------------------------------

        # Round-off error in _integrate_PolydegNGaussian_MidOrigin
        # due to exp((x0+l)/b) - exp((x0-l)/b) when b is large and l is small.
        # --> no significant improvement compare to LeftOrigin

        cN_Px = np.zeros(len(cN)+1, dtype=float)
        cN_Pxx = np.zeros(len(cN)+2, dtype=float)
        cN_Pxxx = np.zeros(len(cN)+3, dtype=float)
        if Mid_TaylorExpan :
            # P
            P = self._integrate_PolydegNGaussian_MidOrigin(cN, d+l/2, l/2, 4*Ddt)
            # Px
            cN_Px[:-1] += (d+l/2)*cN
            cN_Px[1:]  += cN
            Px = self._integrate_PolydegNGaussian_MidOrigin(cN_Px, d+l/2, l/2, 4*Ddt)
            # Pxx
            cN_Pxx[:-2]  += ((d+l/2)**2 - 2*Ddt)*cN
            cN_Pxx[1:-1] += (2*d+l)*cN
            cN_Pxx[2:]   += cN
            Pxx = self._integrate_PolydegNGaussian_MidOrigin(cN_Pxx, d+l/2, l/2, 4*Ddt)
            # Pxxx
            cN_Pxxx[:-3]  += (d+l/2)*((d+l/2)**2 - 6*Ddt)*cN
            cN_Pxxx[1:-2] += (3*(d+l/2)**2 - 6*Ddt)*cN
            cN_Pxxx[2:-1] += 3*(d+l/2)*cN
            cN_Pxxx[3:]   += cN
            Pxxx = self._integrate_PolydegNGaussian_MidOrigin(cN_Pxxx, d+l/2, l/2, 4*Ddt)
        else:
            # P
            P = self._integrate_PolydegNGaussian_LeftOrigin(cN, d, l, 4*Ddt)
            # Px
            cN_Px[:-1] += d*cN
            cN_Px[1:]  += cN
            Px = self._integrate_PolydegNGaussian_LeftOrigin(cN_Px, d, l, 4*Ddt)
            # Pxx
            cN_Pxx[:-2]  += (d**2 - 2*Ddt)*cN
            cN_Pxx[1:-1] += 2*d*cN
            cN_Pxx[2:]   += cN
            Pxx = self._integrate_PolydegNGaussian_LeftOrigin(cN_Pxx, d, l, 4*Ddt)
            # Pxxx
            cN_Pxxx[:-3]  += d*(d**2 - 6*Ddt)*cN
            cN_Pxxx[1:-2] += (3*d**2 - 6*Ddt)*cN
            cN_Pxxx[2:-1] += 3*d*cN
            cN_Pxxx[3:]   += cN
            Pxxx = self._integrate_PolydegNGaussian_LeftOrigin(cN_Pxxx, d, l, 4*Ddt)

        #P /= # np.sqrt(4*np.pi*D_coef*dt)
        #Px   *= 1/(2*D_coef*dt)    #/np.sqrt(4*np.pi*D_coef*dt)
        #Pxx  *= 1/(2*D_coef*dt)**2 #/np.sqrt(4*np.pi*D_coef*dt)
        #Pxxx *= 1/(2*D_coef*dt)**3 #/np.sqrt(4*np.pi*D_coef*dt)
        # --> cancelled out --> handle outside --> improve speed and reduce numerical error


        # After using this function to transform the entire domain and BC --->
        #      phi_n+1 = -2*D_coef/P (dP/dx) = -2*v*P'/P = -2*D_coef* Px/P
        # grad_phi_n+1 = -2*D_coef (P*P'' - (P')^2)/(P^2) = -2*D_coef* (P*Pxx - Px*Px)/P/P
        # curv_phi_n+1 = -2*D_coef* (P**2 * Pxxx + 2*Px**3 - 3*P*Px*Pxx)/P/P/P
        #
        # Const is not cancelled out as it will be sum up first
        # However, the constant np.pi, D_coef, dt could get cancelled
        return Const*P, Const*Px, Const*Pxx, Const*Pxxx


    def _septic_coeff(self, L, R, grad_L, grad_R, curv_L, curv_R, d_curv_L, d_curv_R, dx):
        '''
        Return Septic coefficients of phi(x) where x[n] < x < x[n+1]
        *** Locally, from [0, dx] ***
        That is, phi(x) = phi(x[n]+y) = a*y**7 + b*y**6 + c*y**5 + d*y**4 + e*y**3 + f*y**2 + g*y + h
        '''

        h = L
        g = grad_L
        f = curv_L/2.
        e = d_curv_L/6.
        a = (-20./dx**7)*( R-(e*dx**3+f*dx**2+g*dx+h) ) + ( 10./dx**6)*( grad_R-(3*e*dx**2+curv_L*dx+g) ) + (-2.0/dx**5)*( curv_R-(6*e*dx+curv_L) ) + ( 1./6./dx**4)*( d_curv_R-d_curv_L )
        b = ( 70./dx**6)*( R-(e*dx**3+f*dx**2+g*dx+h) ) + (-34./dx**5)*( grad_R-(3*e*dx**2+curv_L*dx+g) ) + ( 6.5/dx**4)*( curv_R-(6*e*dx+curv_L) ) + (-0.5  /dx**3)*( d_curv_R-d_curv_L )
        c = (-84./dx**5)*( R-(e*dx**3+f*dx**2+g*dx+h) ) + ( 39./dx**4)*( grad_R-(3*e*dx**2+curv_L*dx+g) ) + (-7.0/dx**3)*( curv_R-(6*e*dx+curv_L) ) + ( 0.5  /dx**2)*( d_curv_R-d_curv_L )
        d = ( 35./dx**4)*( R-(e*dx**3+f*dx**2+g*dx+h) ) + (-15./dx**3)*( grad_R-(3*e*dx**2+curv_L*dx+g) ) + ( 2.5/dx**2)*( curv_R-(6*e*dx+curv_L) ) + (-1./6./dx   )*( d_curv_R-d_curv_L )
        # f(x) = a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h

        return a, b, c, d, e, f, g, h


    def is_num_cN_enough(self, isAuto, n, cN, l):
        '''
        Decide if the next cN is needed
        '''
        if isAuto and abs(cN*math.pow(l,n)) < 0.0001*math.factorial(n) :
            return n
        else:
            return n+1


    def __cal_TaylorExpCoef_HopfCole(self, deg_of_var, from_pow, c):
        '''
        Return (-0.5/D)**n/(n)! PMF(d1,d2,d3,d4,d5,d6,d7) c1**d1 c2**d2 c3**d3 c4**d4 c5**d5 c6**d6 c7**d7
        where PMF= Probability mass function
        '''
        return (-0.5)**from_pow \
               /math.factorial(deg_of_var[0]) \
               /math.factorial(deg_of_var[1]) \
               /math.factorial(deg_of_var[2]) \
               /math.factorial(deg_of_var[3]) \
               /math.factorial(deg_of_var[4]) \
               /math.factorial(deg_of_var[5]) \
               /math.factorial(deg_of_var[6]) \
                * c[0]**deg_of_var[0] \
                * c[1]**deg_of_var[1] \
                * c[2]**deg_of_var[2] \
                * c[3]**deg_of_var[3] \
                * c[4]**deg_of_var[4] \
                * c[5]**deg_of_var[5] \
                * c[6]**deg_of_var[6]


    def _integrate_PolydegNGaussian_MidOrigin(self, cN, x0, l, b):
        '''
        Return the integration of polynomial (N terms) x gaussian function
        from xi-l to xi+l (local y from -l to l) [Middle]

        --> Warning !! round-off error when exp((x+l)/b) - exp((x-l)/b)

        SHOULD be imporved to reduce round-off error ****
        - Try decimal, float128, ...
        - More precise function: erf, exp, sqrt ...

        Note: cN[0] + cN[1]*y + cN[2]*y**2 + ... + cN[N-1]*y**(N-1)
        '''

        N = len(cN)
        sol = np.zeros(N)

        # 1 --- Integration of c0*exp(-(y+x0)^2/b)
        if N > 0 :
            sol[0] = 0.5*np.sqrt(np.pi*b) * (math.erf( (x0+l)/np.sqrt(b) ) - math.erf( (x0-l)/np.sqrt(b) ))

        # y --- Integration of c1*y*exp(-(y+x0)^2/b)
        if N > 1 :
            sol[1]  = -0.5*np.sqrt(np.pi*b)*x0 * (math.erf( (x0+l)/np.sqrt(b) ) - math.erf( (x0-l)/np.sqrt(b) ))
            sol[1] += -0.5*b * (np.exp( -(x0+l)**2/b ) - np.exp( -(x0-l)**2/b ))
        else:
            return cN[0]*sol[0]

        # y**k where k>1 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
        for k in range(2,N):
            sol[k]  = -0.5*b*( l**(k-1)*np.exp( -(x0+l)**2/b ) - (-l)**(k-1)*np.exp( -(x0-l)**2/b ))
            sol[k] += -x0*sol[k-1]
            sol[k] +=  0.5*b*(k-1)*sol[k-2]

        return np.sum(np.multiply(cN, sol))


"""
def find_combination_to_N(N):
    '''
    find [n1, n2, n3, n4, n5, n6, n7] that give N = n1 + 2*n2 + 3*n3 + 4*n4 + 5*n5 + 6*n6 + 7*n7
    where ni is a positive integer.
    '''
    n7 = 0
    n6 = 0
    n5 = 0
    n4 = 0
    n3 = 0
    n2 = 0
    n1 = 0
    while( n7 <= N/7 ):
        while( n6 <= N/6 ):
            while( n5 <= N/5 ):
                while( n4 <= N/4 ):
                    while( n3 <= N/3 ):
                        while( n2 <= N/2 ):
                            while( n1 <= N ):
                                if N == n1 + 2*n2 + 3*n3 + 4*n4 + 5*n5 + 6*n6 + 7*n7 :
                                    #print('Found = [',n1,',',n2,',',n3,',',n4,',',n5,']')
                                    print('cN['+str(N)+'] += self.__cal_TaylorExpCoef_HopfCole(['+str(n1)+','+str(n2)+','+str(n3)+','+str(n4)+','+str(n5)+','+str(n6)+','+str(n7)+'],', str(n1+n2+n3+n4+n5+n6+n7)+', c_int_phi)')
                                n1 += 1
                            n1 = 0
                            n2 += 1
                        n1 = 0
                        n2 = 0
                        n3 += 1
                    n1 = 0
                    n2 = 0
                    n3 = 0
                    n4 += 1
                n1 = 0
                n2 = 0
                n3 = 0
                n4 = 0
                n5 += 1
            n1 = 0
            n2 = 0
            n3 = 0
            n4 = 0
            n5 = 0
            n6 += 1
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        n5 = 0
        n6 = 0
        n7 += 1

    return
"""



