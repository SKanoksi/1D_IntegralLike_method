import numpy as np
import matplotlib.pyplot as plt

from Parallel_Burger_Block import *
from Others.BurgerCases import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

root = size-1  # Don't change


# ------------ BEGIN MAIN ------------

# Infinite BC:: Traveling wave
if rank == root :
    print('\n\nTraveling wave: case 100 \n\n')
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(100, 50, 1)
for sc in [0, 1, 4, 16, 64, 128, 256, 384, 512, 640] :
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    if rank == root :
        size = 12
        plt.rcParams.update({'font.size': size})

        plt.figure(0, figsize=(9.6, 3.6))
        plt.plot(cg_x, Ana_func(cg_x, t0+sc*nt*dt, K), 'k', label='Ana')
        plt.plot(lg_x, lg_phi , 'g-.', label='LG')
        plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
        plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')
        if sc == 0 :
            plt.legend()

if rank == root :
    plt.text(-0.05, 0.5, 'T = 0.001, 0.041, 0.161, 0.641', fontsize=10, horizontalalignment='right', verticalalignment='center')
    plt.text(0.9, 0.5, 'T = 2.561', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-60)
    plt.text(2.5, 0.5, 'T = 5.121', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-45)
    plt.text(5, 0.5, 'T = 10.241', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-45)
    plt.text(7.5, 0.5, 'T = 15.361', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-45)
    plt.text(10, 0.5, 'T = 20.481', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-45)
    plt.text(12.5, 0.5, 'T = 25.601', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-45)
    plt.locator_params(axis='x', nbins=13)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.xlim([-7.5, 22])
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_100-50-1.pdf', dpi=960)
    #plt.show()
    plt.clf()


# Infinite BC:: Traveling wave
if rank == root :
    print('\n\nTraveling wave: case 102 \n\n')
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(102, 200, 1)
for sc in [0, 250, 500, 750, 1000]:
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    if rank == root :
        size = 12
        plt.rcParams.update({'font.size': size})

        plt.figure(0, figsize=(9.6, 3.6))
        plt.plot(cg_x, Ana_func(cg_x, t0+sc*nt*dt, K), 'k', label='Ana')
        plt.plot(lg_x, lg_phi , 'g-.', label='LG')
        plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
        plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')
        if sc == 0 :
            plt.legend()

if rank == root :
    plt.text(-1, 0.5, 'T = 0.5', fontsize=10, horizontalalignment='right', verticalalignment='center')
    plt.text(11.5, 0.5, 'T = 25.5', fontsize=10, horizontalalignment='right', verticalalignment='center')
    plt.text(26, 0.5, 'T = 50.5', fontsize=10, horizontalalignment='left', verticalalignment='center')
    plt.text(38.5, 0.5, 'T = 75.5', fontsize=10, horizontalalignment='left', verticalalignment='center')
    plt.text(51, 0.5, 'T = 100.5', fontsize=10, horizontalalignment='left', verticalalignment='center')
    plt.locator_params(axis='x', nbins=13)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.xlim([-10, 70])
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_102-200-1.pdf', dpi=960)
    #plt.show()
    plt.clf()


# Infinite BC:: Traveling wave
if rank == root :
    print('\n\nTraveling wave: case 103 \n\n')
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(103, 100, 1)
for sc in [0, 250, 500, 750, 1000]:
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    if rank == root :
        size = 12
        plt.rcParams.update({'font.size': size})

        plt.figure(0, figsize=(9.6, 3.6))
        plt.plot(cg_x, Ana_func(cg_x, t0+sc*nt*dt, K), 'k', label='Ana')
        plt.plot(lg_x, lg_phi , 'g-.', label='LG')
        plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
        plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')
        if sc == 0 :
            plt.legend(loc=(0.1,0.1))

if rank == root :
    plt.text(7, 0.5, 'T = 10', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-85)
    plt.text(107, 0.5, 'T = 210', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-85)
    plt.text(207, 0.5, 'T = 410', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-85)
    plt.text(307, 0.5, 'T = 610', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-85)
    plt.text(407, 0.5, 'T = 810', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-85)
    plt.locator_params(axis='x', nbins=13)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.xlim([-15, 420])
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_103-100-1.pdf', dpi=960)
    #plt.show()
    plt.clf()


# ----------------------------------------


# No-Flux BC:: Decaying pulse
if rank == root :
    print('\n\nDecaying pulse: case 200 \n\n')
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(200, 60, 35)
for sc in [0, 1, 2, 3] :
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    if rank == root :
        size = 12
        plt.rcParams.update({'font.size': size})

        plt.figure(0, figsize=(9.6, 3.6))
        plt.plot(cg_x, Ana_func(cg_x, t0+sc*nt*dt, K), 'k', label='Ana')
        plt.plot(lg_x, lg_phi , 'g-.', label='LG')
        plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
        plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')
        if sc == 0 :
            plt.legend()

if rank == root :
    plt.text(0.3, 0.35, 'T = 1.0', horizontalalignment='right', verticalalignment='bottom')
    plt.text(0.65, 0.38, 'T = 1.7', horizontalalignment='center', verticalalignment='bottom')
    plt.text(0.77, 0.32, 'T = 2.4', horizontalalignment='center', verticalalignment='bottom')
    plt.text(0.92, 0.28, 'T = 3.1', horizontalalignment='center', verticalalignment='bottom')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.xlim([0,1.2])
    plt.ylim([-0.01,0.5])
    #plt.locator_params(axis='x', nbins=9)
    #plt.locator_params(axis='y', nbins=9)
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_200-60-35.pdf', dpi=960)
    #plt.show()
    plt.clf()


# ----------------------------------------


# Periodic BC:: Sine
if rank == root :
    print('\n\nSine: case 300 \n\n')
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(300, 100, 20)
for sc in [0, 2, 4, 10] :
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None)
    if rank == root :
        size = 12
        plt.rcParams.update({'font.size': size})

        plt.figure(0, figsize=(9.6, 3.6))
        plt.plot(cg_x, Ana_func(cg_x, t0+sc*nt*dt, K), 'k', label='Ana')
        plt.plot(lg_x, lg_phi , 'g-.', label='LG')
        plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
        plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')
        if sc == 0 :
            plt.legend()

if rank == root :
    plt.text(0.50,  1.05, 'T = 0.0', horizontalalignment='center', verticalalignment='bottom')
    plt.text(0.72, 1.05, 'T = 0.2', horizontalalignment='center', verticalalignment='bottom')
    plt.text(1.0, 1.0, 'T = 0.4', horizontalalignment='center', verticalalignment='bottom')
    plt.text(1.02, 0.3, 'T = 1.0', horizontalalignment='left', verticalalignment='center')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.xlim([0.0, 2.0])
    plt.ylim([-1.1,1.3])
    #plt.locator_params(axis='y', nbins=9)
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_300-100-20.pdf', dpi=960)
    #plt.show()
    plt.clf()


# ----------------------------------------

# Infinite BC:: Traveling wave with additional source term
if rank == root :
    print('\n\nAdditional source term: case 400 \n\n')
nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(400, 50, 1)
for sc in [0, 1, 4, 16, 64, 128, 256, 384, 512, 640] :
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None, Is_Source_Experiment=True)
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None, Is_Source_Experiment=True)
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, savefig=None, Is_Source_Experiment=True)
    if rank == root :
        size = 12
        plt.rcParams.update({'font.size': size})

        plt.figure(0, figsize=(9.6, 3.6))
        plt.plot(cg_x, Ana_func(cg_x, t0+sc*nt*dt, K), 'k', label='Ana')
        plt.plot(lg_x, lg_phi , 'g-.', label='LG')
        plt.plot(cg_x, cg_phi , 'b:x', markersize=6, label='CG')
        plt.plot(qg_x, qg_phi , 'ro', markersize=3, label='QG')
        if sc == 0 :
            plt.legend()

if rank == root :
    plt.text(-0.5, 0.5, 'T = 0, 0.04, 0.16, 0.64', fontsize=10, horizontalalignment='right', verticalalignment='center')
    plt.text(1.2, 0.5, 'T = 2.56', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-75)
    plt.text(2.5, 0.5, 'T = 5.12', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-75)
    plt.text(5.0, 0.5, 'T = 10.24', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-75)
    plt.text(7.7, 0.5, 'T = 15.36', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-75)
    plt.text(10.2, 0.5, 'T = 20.48', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-75)
    plt.text(12.75, 0.5, 'T = 25.60', fontsize=10, horizontalalignment='left', verticalalignment='center', rotation=-75)
    plt.locator_params(axis='x', nbins=13)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.xlim([-7.5, 22])
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1)

    plt.savefig('TestCase_400-50-1.pdf', dpi=960)
    #plt.show()
    plt.clf()


# ------------ END MAIN ------------

MPI.Finalize()
