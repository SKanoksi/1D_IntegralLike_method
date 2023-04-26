
import scipy.special as sp
import numpy as np
import warnings
warnings.filterwarnings("error")

from Parallel_Burger_Block import *
from Others.BurgerCases import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

root = size-1  # Don't change


# ------------ BEGIN MAIN ------------

# Run with 1 MPI process *** (IMPORTANT) ***
# --> mpi4py is probably not compatible with standard python try-except
# --> Otherwise, remove K = 0.25


# Infinite BC:: Traveling wave
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(103, 400, 1)
xL *= 4
xR *= 4
t0 = 1560
nt = 100

# Vary nx <--> dx  -- while K, dt, nt are fixed
K_list = np.array([0.5, 0.75, 1, 2, 4, 8, 16, 32, 64])
# + Causing exception at K = 0.25 !!!
# + Initial phi of large K could disagree with BC. Careful !!

if rank == root :
    calError = L2ErrorNorm
    lg_error = np.ones(len(K_list), dtype=np.float64)
    cg_error = np.ones(len(K_list), dtype=np.float64)
    qg_error = np.ones(len(K_list), dtype=np.float64)

for i in range(len(K_list)):

    try:
        lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, nt, dt, t0, K_list[i], BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            phi_ana = Ana_func(lg_x, t0 + nt*dt, K_list[i])
            lg_error[i] = calError(lg_phi, phi_ana)
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            lg_error[i] = 10**3

    try:
        cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, nt, dt, t0, K_list[i], BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            phi_ana = Ana_func(cg_x, t0 + nt*dt, K_list[i])
            cg_error[i] = calError(cg_phi, phi_ana)
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            cg_error[i] = 10**3

    try:
        qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, nt, dt, t0, K_list[i], BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            phi_ana = Ana_func(qg_x, t0 + nt*dt, K_list[i])
            qg_error[i] = calError(qg_phi, phi_ana)
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            qg_error[i] = 10**3

    #if rank == root :
    #    plt.figure(1, figsize=(9.6, 3.6))
    #    plt.plot(lg_x, phi_ana, 'k', label='Ana')
    #    plt.plot(lg_x, lg_phi , 'g-.', label='LG')
    #    plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
    #    plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')

if rank == root :
    size = 12
    plt.rcParams.update({'font.size': size})
    plt.figure(0, figsize=(9.6, 3.6))

    dx = (xR-xL)/(nx-1)
    D_list = K_list*dt/dx/dx
    plt.loglog(D_list, lg_error, 'g-.^', label='LG')
    plt.loglog(D_list, cg_error, 'b:x', markersize=6, label='CG')
    plt.loglog(D_list, qg_error, 'r--o', markersize=4, label='QG')

    y_limit = [3*10**(-6), 10]
    plt.plot([1.0]*2   , y_limit, 'k:')
    plt.plot([0.02]*2, y_limit, 'k-.')
    plt.text(1.0, 3, 'd = 1', fontsize=10, horizontalalignment='right', verticalalignment='center', rotation=+90)
    plt.text(0.02, 10**(-4), 'd = 0.02', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-90)

    plt.xlabel(r'Diffusivity constant $d = \nu \Delta t/\Delta x^2$')
    plt.ylabel(r'$\ell_2 error norm$')
    plt.ylim(y_limit)
    #plt.xlim([0.35, 70])
    plt.legend()

    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_103_Stability.pdf', dpi=960)
    #plt.show()
    plt.clf()


# ------------ END MAIN ------------

MPI.Finalize()
