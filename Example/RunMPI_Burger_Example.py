
import scipy.special as sp
import numpy as np

from Parallel_Burger_Block import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

root = size-1  # Don't change


# ------------ BEGIN MAIN ------------

def Case(x, t, a=1.0):
    z = x/2/np.sqrt(t)
    return (2./np.sqrt(np.pi*t))*np.exp(-z**2)/(a + sp.erfc(z))

Pos = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8]

Parallel_Burger_Block(grid='CG', nx=41, xL=-10, xR=10, nt=400, dt=0.01, t0=0.1, K=1.0,
                      BC='InfiniteBC', Ana_func=Case, IC_func=Case,
                      cal_error=True, sampling_pos=Pos, savefig='Example_1_CG.png')



# ------------ END MAIN ------------

MPI.Finalize()
