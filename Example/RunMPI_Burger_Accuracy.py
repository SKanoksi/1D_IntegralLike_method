
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

# Run each case separately --> See the comments below

def linearFit(x, y, sigma):
    """Function to perform linear regression (fit a line)
    Inputs
      x       Independent variable
      y       Dependent variable
      sigma   Estimated error in y
    Outputs
      fit   Fit parameters; b is intercept, a is slope
      sig   Estimated error in the parameters a()
      yy      Curve fit to the data
      chisqr  Chi squared statistic
    """
    # (From my undergraduate class using MatLab)

    # Check inputs
    if len(x)!=len(y) or len(x)!=len(sigma) :
        raise Exception('Input: x, y and sigma should have the same length.')
    else:
        x = x.astype(float)
        y = y.astype(float)
        sigma = sigma.astype(float)

    # Evaluate various sigma sums
    sigmaTerm = sigma**(-2)
    s = sum(sigmaTerm)
    sx = sum(np.multiply(x, sigmaTerm))
    sy = sum(np.multiply(y, sigmaTerm))
    sxy = sum(np.multiply(np.multiply(x,y), sigmaTerm))
    sxx = sum(np.multiply(np.multiply(x,x), sigmaTerm))
    denom = s*sxx - sx**2

    # Compute intercept b_fit and slope a_fit
    a_fit = (s*sxy - sx*sy)/denom
    b_fit = (sxx*sy - sx*sxy)/denom

    # Compute error bars for intercept and slope
    a_sig = np.sqrt(s/denom)
    b_sig = np.sqrt(sxx/denom)

    # Compute Chi-square
    yy = a_fit*x + b_fit
    chisqr = sum( np.power(np.divide((y-yy), sigma), 2) )

    fit = np.array([a_fit, b_fit])
    sig = np.array([a_sig, b_sig])

    return fit, sig, chisqr


# ----------------------------------------


# No-Flux BC:: Decaying pulse
if rank == root :
    print('\n\n No-Flux BC -- Decaying pulse \n\n')

nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(200, 60, 35)
nt = 100

# Vary nx <--> dx  -- while K, dt, nt are fixed
nx_list = np.array([30, 40, 50, 60, 80, 100, 120, 160, 200, 240]) + 1

#nx_list = np.array([10, 15, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240]) + 1
# Cannot run nx = 10, 15 with MPI
# --> mpi4py may not compatible with standard python try-except

dx_list = (xR-xL)/(nx_list-1)

if rank == root :
    calError = L2ErrorNorm
    lg_error = np.ones(len(nx_list), dtype=np.float64)
    cg_error = np.ones(len(nx_list), dtype=np.float64)
    qg_error = np.ones(len(nx_list), dtype=np.float64)

for i in range(len(nx_list)):

    try:
        lg_phi, lg_x = Parallel_Burger_Block('LG', nx_list[i], xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            lg_error[i] = calError(lg_phi, Ana_func(lg_x, t0+nt*dt, K))
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            lg_error[i] = 10**3

    try:
        cg_phi, cg_x = Parallel_Burger_Block('CG', nx_list[i], xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            cg_error[i] = calError(cg_phi, Ana_func(cg_x, t0+nt*dt, K))
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            cg_error[i] = 10**3

    try:
        qg_phi, qg_x = Parallel_Burger_Block('QG', nx_list[i], xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            qg_error[i] = calError(qg_phi, Ana_func(qg_x, t0+nt*dt, K))
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            qg_error[i] = 10**3

    #if rank == root :
    #    plt.figure(1, figsize=(9.6, 3.6))
    #    plt.plot(cg_x, Ana_func(cg_x, t0+nt*dt, K), 'k', label='Ana')
    #    plt.plot(lg_x, lg_phi , 'g-.', label='LG')
    #    plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
    #    plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')


if rank == root :
    size = 12
    plt.rcParams.update({'font.size': size})
    plt.figure(0, figsize=(9.6, 3.6))

    plt.loglog(dx_list, lg_error, 'g-.^', label='LG')
    plt.loglog(dx_list, cg_error, 'b:x', markersize=6, label='CG')
    plt.loglog(dx_list, qg_error, 'r-o', markersize=3, label='QG')

    coeff = linearFit(np.log(dx_list), np.log(lg_error), np.ones_like(dx_list))[0]
    print('The order of LG scheme is around', coeff[0],'accurate')
    coeff = linearFit(np.log(dx_list), np.log(cg_error), np.ones_like(dx_list))[0]
    print('The order of CG scheme is around', coeff[0],'accurate')
    coeff = linearFit(np.log(dx_list), np.log(qg_error), np.ones_like(dx_list))[0]
    print('The order of QG scheme is around', coeff[0],'accurate')

    y_limit = [2*10**(-5), 1]
    dx_d1   = np.sqrt(K*dt)
    plt.plot([dx_d1]*2, y_limit, 'k:')
    plt.text(dx_d1, 0.3, 'd = 1', fontsize=10, horizontalalignment='right', verticalalignment='center', rotation=+90)
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\ell_2$')
    plt.ylim(y_limit)
    plt.legend()

    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_200_Acc.pdf', dpi=960)
    #plt.show()
    plt.clf()



# ----------------------------------------

# Run this case with MPI !!! (Periodic BC)

# Periodic BC:: Sine
if rank == root :
    print('\n\n Periodic BC -- Sine \n\n')

nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(300, 100, 8)
nt = 100

# Vary nx <--> dx  -- while K, dt, nt are fixed
nx_list = np.array([50, 75, 100, 150, 200, 250, 300, 350, 400]) + 1
dx_list = (xR-xL)/(nx_list-1)

if rank == root :
    calError = L2ErrorNorm
    lg_error = np.ones(len(nx_list), dtype=np.float64)
    cg_error = np.ones(len(nx_list), dtype=np.float64)
    qg_error = np.ones(len(nx_list), dtype=np.float64)

for i in range(len(nx_list)):

    try:
        lg_phi, lg_x = Parallel_Burger_Block('LG', nx_list[i], xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            lg_error[i] = calError(lg_phi, Ana_func(lg_x, t0+nt*dt, K))
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            lg_error[i] = 10**3

    try:
        cg_phi, cg_x = Parallel_Burger_Block('CG', nx_list[i], xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            cg_error[i] = calError(cg_phi, Ana_func(cg_x, t0+nt*dt, K))
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            cg_error[i] = 10**3

    try:
        qg_phi, qg_x = Parallel_Burger_Block('QG', nx_list[i], xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
        if rank == root :
            qg_error[i] = calError(qg_phi, Ana_func(qg_x, t0+nt*dt, K))
    except (RuntimeWarning, OverflowError) :
        if rank == root :
            qg_error[i] = 10**3

    #if rank == root :
    #    plt.figure(1, figsize=(9.6, 3.6))
    #    plt.plot(cg_x, Ana_func(cg_x, t0+nt*dt, K), 'k', label='Ana')
    #    plt.plot(lg_x, lg_phi , 'g-.', label='LG')
    #    plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
    #    plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')


if rank == root :
    size = 12
    plt.rcParams.update({'font.size': size})
    plt.figure(0, figsize=(9.6, 3.6))

    plt.loglog(dx_list, lg_error, 'g-.^', label='LG')
    plt.loglog(dx_list, cg_error, 'b:x', markersize=6, label='CG')
    plt.loglog(dx_list, qg_error, 'r-o', markersize=3, label='QG')

    coeff = linearFit(np.log(dx_list), np.log(lg_error), np.ones_like(dx_list))[0]
    print('The order of LG scheme is around', coeff[0],'accurate')
    coeff = linearFit(np.log(dx_list), np.log(cg_error), np.ones_like(dx_list))[0]
    print('The order of CG scheme is around', coeff[0],'accurate')
    coeff = linearFit(np.log(dx_list), np.log(qg_error), np.ones_like(dx_list))[0]
    print('The order of QG scheme is around', coeff[0],'accurate')

    y_limit = [10**(-4), 1]
    dx_d1   = np.sqrt(K*dt)
    plt.plot([dx_d1]*2, y_limit, 'k:')
    plt.text(dx_d1, 0.3, 'd = 1', fontsize=10, horizontalalignment='right', verticalalignment='center', rotation=+90)

    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\ell_2$')
    plt.ylim(y_limit)
    plt.legend()

    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_300_Acc.pdf', dpi=960)
    #plt.show()
    plt.clf()


# ------------ END MAIN ------------

MPI.Finalize()
