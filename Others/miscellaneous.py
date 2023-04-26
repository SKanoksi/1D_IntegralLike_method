#
# Various other functions
#
# = L1ErrorNorm(phi, phiExact)
# = L2ErrorNorm(phi, phiExact)
# = LinfErrorNorm(phi, phiExact)
# = TotalMass(phi, dx=1)
#
# Copyright (c) 2023
# Somrath Kanoksirirath <somrath.kan@ncr.nstda.or.th>
# All rights reserved under BSD 3-clause license.
#

import numpy as np
import scipy.special as sp

def L1ErrorNorm(phi, phiExact):
    """
    Calculates the L1 error norm (RAS error) of phi in comparison to
    phiExact --> dx = 1
    """
    return np.sum(np.abs(phi - phiExact))/np.sum(np.abs(phiExact))


def L2ErrorNorm(phi, phiExact):
    """
    Calculates the L2 error norm (RMS error) of phi in comparison to
    phiExact --> dx = 1
    """
    return np.sqrt(np.sum( (phi - phiExact)**2 )/np.sum(phiExact**2))


def LinfErrorNorm(phi, phiExact):
    """
    Calculates the L2 error norm (RMS error) of phi in comparison to
    phiExact
    """
    return np.max(np.abs(phi - phiExact))/np.max(np.abs(phiExact))


def TotalMass(phi, x):
    """
    Calculates the total mass of phi in the domain using trapezoidal rule
    """

    value = 0.5*(phi[1:] + phi[:-1])
    spacing = x[1:] - x[:-1]

    return np.sum(np.multiply(np.array(value),np.array(spacing)))


# ---------------------------------------------------


def find_phi_at(pos, grid):
    value = np.zeros_like(pos)
    for i in range(len(pos)) :
        value[i] = grid.phi_at(pos[i])
    return value


def find_phi_in_array(pos, x, value):
    out = np.zeros_like(pos)
    for i in range(len(pos)) :
        # Find index
        iR = 0
        while iR < len(x):
            if pos[i] < x[iR]:
                iL = iR - 1
                break
            iR += 1
        # Find value
        d = pos[i] - x[iL]
        out[i] = value[iL] + d*(value[iR]-value[iL])/(x[iR]-x[iL])

    return out
