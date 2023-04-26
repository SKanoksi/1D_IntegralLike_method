#
# Class: work_burger
# = for solving Burgers' equation using the integral-like approach
# = Serial run --> to Parallel run
#
# Copyright (c) 2023
# Somrath Kanoksirirath <somrath.kan@ncr.nstda.or.th>
# All rights reserved under BSD 3-clause license.
#

import numpy as np

from Grid.LinearGrid_1D import LinearGrid_1D as linear_grid
from Grid.CubicGrid_1D import CubicGrid_1D as cubic_grid
from Grid.QuinticGrid_1D import QuinticGrid_1D as quintic_grid


class abstract_work_burger_dimless :
    def __init__(self, nPhi, K, dt, nl_sigma):
        self.Kdt = K*dt
        self.dt = dt
        self.nl_sigma = nl_sigma
        self.nL = self.nl_sigma
        self.nR = nPhi - self.nl_sigma

    def shared_left_BC(self, var):
        return var[self.nL:self.nL+self.nl_sigma]

    def shared_right_BC(self, var):
        return var[self.nR-self.nl_sigma:self.nR]

    def apply_left_PeriodicBC(self, varL, varR, replace=False):
        nR = len(varR) - self.nl_sigma
        if replace :
            varL[:self.nL+1] = varR[nR-self.nl_sigma-1:nR]
        else:
            varL[:self.nL] = varR[nR-self.nl_sigma-1:nR-1]

    def apply_right_PeriodicBC(self, varR, varL, replace=False):
        if replace :
            varR[len(varR)-self.nl_sigma-1:] = varL[self.nL:self.nL+self.nl_sigma+1]
        else:
            varR[len(varR)-self.nl_sigma:] = varL[self.nL+1:self.nL+self.nl_sigma+1]

    def apply_left_MirrorBC(self, var, inverted=False):
        if inverted :
            var[self.nL-1::-1] = - var[self.nL+1:self.nL+self.nl_sigma+1]
        else:
            var[self.nL-1::-1] = var[self.nL+1:self.nL+self.nl_sigma+1]

    def apply_right_MirrorBC(self, var, inverted=False):
        if inverted :
            var[self.nR:] = - var[self.nR-2:self.nR-self.nl_sigma-2:-1]
        else:
            var[self.nR:] = var[self.nR-2:self.nR-self.nl_sigma-2:-1]

    def _derive_phi(self, P, Px):
        return -Px/np.abs(P)/self.Kdt

    def _derive_phi_x(self, P, Px, Pxx):
        return (0.5/self.Kdt**2)*(- Pxx + np.sign(P)*Px**2/np.abs(P))/np.abs(P)

    def _derive_phi_xx(self, P, Px, Pxx, Pxxx):
        return (-0.25/self.Kdt**3)*(Pxxx + (-3*Px*Pxx + 2*Px**3/np.abs(P))/np.abs(P))/np.abs(P)

    def forward(self, int_phi, shifted=True):
        pass


# ------------------------------------


class linear_work_burger_dimless(abstract_work_burger_dimless):
    def __init__(self, x0, dx, phi0, K, dt, nl_sigma):
        super().__init__(len(phi0), K, dt, nl_sigma)
        self.g = linear_grid(x0, dx, phi0)
        self.tmp_phi = np.zeros(self.nR-self.nL, dtype=float)

    def forward(self, int_phi, shifted=True):
        # *** 1. Shift integratal of phi
        if shifted :
            maxValue = np.max(int_phi)
            minValue = np.min(int_phi)
            int_phi -= (maxValue + minValue)/2

        # *** 2. Hopf-Cole transformation + Diffusion in Hopf-Cole space
        # OMP PARALLEL
        for j in range(self.nL, self.nR):
            # OMP PRIVATE
            Ps = np.zeros(2, dtype=float)
            for i in range(j-self.nl_sigma, j+self.nl_sigma):
                Ps[:] += self.g.diffuse_in_HopfColeSpace_DimLess_without_commonfactor(int_phi, i, self.g.dx, (i-j)*self.g.dx, self.Kdt, nTerms_TaylorSeries=16, Mid_TaylorExpan=True, isAuto=True)

            # *** 3. Inverse Hopf-Cole transform
            # OMP REDUCTION
            # SERIAL OPERATION
            self.tmp_phi[j-self.nL] = self._derive_phi(Ps[0], Ps[1])

        # *** 4. Update variables
        self.g.phi[self.nL:self.nR] = self.tmp_phi


# ------------------------------------


class cubic_work_burger_dimless(abstract_work_burger_dimless) :
    def __init__(self, x0, dx, phi0, K, dt, nl_sigma):
        super().__init__(len(phi0), K, dt, nl_sigma)
        self.g = cubic_grid(x0, dx, phi0)
        nx = self.nR - self.nL
        self.tmp_phi      = np.zeros(nx, dtype=float)
        self.tmp_grad_phi = np.zeros(nx, dtype=float)

    def forward(self, int_phi, shifted=True):
        # *** 1. Shift integratal of phi
        if shifted :
            maxValue = np.max(int_phi)
            minValue = np.min(int_phi)
            int_phi -= (maxValue + minValue)/2

        # *** 2. Hopf-Cole transformation + Diffusion in Hopf-Cole space
        # OMP PARALLEL
        for j in range(self.nL, self.nR):
            # OMP PRIVATE
            Ps = np.zeros(3, dtype=float)
            for i in range(j-self.nl_sigma, j+self.nl_sigma):
                Ps[:] += self.g.diffuse_in_HopfColeSpace_DimLess_without_commonfactor(int_phi, i, self.g.dx, (i-j)*self.g.dx, self.Kdt, nTerms_TaylorSeries=16, Mid_TaylorExpan=True, isAuto=True)

            # *** 3. Inverse Hopf-Cole transform
            # OMP REDUCTION
            # SERIAL OPERATION
            self.tmp_phi[j-self.nL] = self._derive_phi(Ps[0], Ps[1])
            self.tmp_grad_phi[j-self.nL] = self._derive_phi_x(Ps[0], Ps[1], Ps[2])

        # *** 4. Update variables
        self.g.phi[self.nL:self.nR] = self.tmp_phi
        self.g.grad_phi[self.nL:self.nR] = self.tmp_grad_phi


# ------------------------------------


class quintic_work_burger_dimless(abstract_work_burger_dimless) :
    # Task <--> MPI rank
    # Proc <--> OpenMP thread (/Serial loop)
    def __init__(self, x0, dx, phi0, K, dt, nl_sigma):
        super().__init__(len(phi0), K, dt, nl_sigma)
        self.g = quintic_grid(x0, dx, phi0)
        nx = self.nR - self.nL
        self.tmp_phi      = np.zeros(nx, dtype=float)
        self.tmp_grad_phi = np.zeros(nx, dtype=float)
        self.tmp_curv_phi = np.zeros(nx, dtype=float)

    def forward(self, int_phi, shifted=True):
        # *** 1. Shift integratal of phi
        if shifted :
            maxValue = np.max(int_phi)
            minValue = np.min(int_phi)
            int_phi -= (maxValue + minValue)/2

        # *** 2. Hopf-Cole transformation + Diffusion in Hopf-Cole space
        # OMP PARALLEL
        for j in range(self.nL, self.nR):
            # OMP PRIVATE
            Ps = np.zeros(4, dtype=float)
            for i in range(j-self.nl_sigma, j+self.nl_sigma):
                Ps[:] += self.g.diffuse_in_HopfColeSpace_DimLess_without_commonfactor(int_phi, i, self.g.dx, (i-j)*self.g.dx, self.Kdt, nTerms_TaylorSeries=16, Mid_TaylorExpan=True, isAuto=True)

            # *** 3. Inverse Hopf-Cole transform
            # OMP REDUCTION
            # SERIAL OPERATION
            self.tmp_phi[j-self.nL] = self._derive_phi(Ps[0], Ps[1])
            self.tmp_grad_phi[j-self.nL] = self._derive_phi_x(Ps[0], Ps[1], Ps[2])
            self.tmp_curv_phi[j-self.nL] = self._derive_phi_xx(Ps[0], Ps[1], Ps[2], Ps[3])

        # *** 4. Update variables
        self.g.phi[self.nL:self.nR] = self.tmp_phi
        self.g.grad_phi[self.nL:self.nR] = self.tmp_grad_phi
        self.g.curv_phi[self.nL:self.nR] = self.tmp_curv_phi
