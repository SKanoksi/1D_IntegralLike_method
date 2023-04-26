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

# Infinite BC: Traveling wave
if rank == root :
    print('\n\nTraveling wave\n\n')
Pos = [-6., -3., 0., 3., 6., 9., 12., 18., 21.]
for scx, sct in zip([50,50,100,100],[5,10,10,20]):
    nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(-100, scx, sct)
    sc = 20
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_TravelingWave_LG_'+str(sc*nt))
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_TravelingWave_CG_'+str(sc*nt))
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_TravelingWave_QG_'+str(sc*nt))


# No-Flux BC:: Decaying pulse
if rank == root :
    print('\n\nDecaying pulse\n\n')
Pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
for scx, sct in zip([60, 60,120,120],[20,35,35,140]):
    nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(-200, scx, sct)
    sc = 2
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_Decay_LG_'+str(sc*nt))
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_Decay_CG_'+str(sc*nt))
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_Decay_QG_'+str(sc*nt))


# Periodic BC:: Sine
if rank == root :
    print('\n\nSine\n\n')
Pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
for scx, sct in zip([100,100,200,200],[5,10,10,40]):
    nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func = burger_testcase(-300, scx, sct)
    sc = 10
    lg_phi, lg_x = Parallel_Burger_Block('LG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_Sine_LG_'+str(sc*nt))
    cg_phi, cg_x = Parallel_Burger_Block('CG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_Sine_CG_'+str(sc*nt))
    qg_phi, qg_x = Parallel_Burger_Block('QG', nx, xL, xR, sc*nt, dt, t0, K, BC, Ana_func, IC_func,
                                         cal_error=True, sampling_pos=Pos,
                                         savefig='CheckOutput_Sine_QG_'+str(sc*nt))

# ------------ END MAIN ------------

MPI.Finalize()
