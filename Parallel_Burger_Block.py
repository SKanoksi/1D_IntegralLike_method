#
# Parallel_Burger_Block
# = for solving Burgers' equation using the integral-like approach
# = Parallel run
#
# Copyright (c) 2023
# Somrath Kanoksirirath <somrath.kan@ncr.nstda.or.th>
# All rights reserved under BSD 3-clause license.
#

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns as timer

from Work_Burger import *
from Others.miscellaneous import *

from mpi4py import MPI


# MPI function
def Parallel_Burger_Block(grid, nx, xL, xR, nt, dt, t0, K, BC, Ana_func, IC_func, cal_error=False, sampling_pos=None, savefig=None, Is_Source_Experiment=False):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    root = size - 1
    if rank == root :
        if size == 1 :
            print('INFO :: THIS FUNCTION IS SPECIFICALLY WRITTEN FOR >1 MPI PROCESSES.')

    # Tag:
    # 0-16 == root to others
    # >16 = Shared BC
    #    16 + 0-1 == phi
    #    16 + 2-3 == grad_phi
    #    16 + 4-5 == curv_phi
    # >32 = Periodic BC
    #    32 + 0-1 == phi
    #    32 + 2-3 == grad_phi
    #    32 + 4-5 == curv_phi

    dx = (xR-xL)/(nx-1)
    nsigma = 5
    l = nsigma*np.sqrt(2*K*dt)
    nl_sigma = max(int(np.ceil(l/dx)), 1)
    d_coeff = K*dt/dx**2

    # Method initialization
    if grid == 'LG' :
        worker_init = lambda x0, phi0 : linear_work_burger_dimless(x0, dx, phi0, K, dt, nl_sigma)
    elif grid == 'CG' :
        worker_init = lambda x0, phi0 : cubic_work_burger_dimless(x0, dx, phi0, K, dt, nl_sigma)
    elif grid == 'QG' :
        worker_init = lambda x0, phi0 : quintic_work_burger_dimless(x0, dx, phi0, K, dt, nl_sigma)
    else:
        print('INPUT ERROR -- the \'grid\' must be LG, CG or QG.')
        exit(0)

    # MPI Send-Receive == Domain update
    if grid == 'LG' :
        if BC == 'PeriodicBC' :
            def apply_BC():
                nR = len(worker.g.phi) - nl_sigma
                if rank == 0 :
                    comm.Send([worker.g.phi[nl_sigma:2*nl_sigma+1],MPI.DOUBLE], dest=size-1, tag=0+32)
                    comm.Recv([worker.g.phi[:nl_sigma],MPI.DOUBLE], source=size-1, tag=1+32)
                if rank == size -1 :
                    comm.Send([worker.g.phi[nR-nl_sigma-1:nR-1],MPI.DOUBLE], dest=0, tag=1+32)
                    comm.Recv([worker.g.phi[nR-1:],MPI.DOUBLE], source=0, tag=0+32)
        elif BC == 'ZeroPointBC' :
            def apply_BC():
                if rank == 0 :
                    worker.apply_left_MirrorBC(worker.g.phi, inverted=True)
                if rank == size-1 :
                    worker.apply_right_MirrorBC(worker.g.phi, inverted=True)
        else:
            def apply_BC():
                return
        def update_shared_BC():
            # WL :: Shared Right, Receive Right from Left
            if rank != size-1 :
                comm.Send([worker.shared_right_BC(worker.g.phi),MPI.DOUBLE], dest=rank+1, tag=0+16)
                comm.Recv([worker.g.phi[worker.nR:],MPI.DOUBLE], source=rank+1, tag=1+16)
            # WR :: Shared Left, Receive Left from Right
            if rank != 0 :
                comm.Send([worker.shared_left_BC(worker.g.phi),MPI.DOUBLE], dest=rank-1, tag=1+16)
                comm.Recv([worker.g.phi[:worker.nL],MPI.DOUBLE], source=rank-1, tag=0+16)
    elif grid == 'CG' :
        if BC == 'PeriodicBC' :
            def apply_BC():
                nR = len(worker.g.phi) - nl_sigma
                if rank == 0 :
                    comm.Send([worker.g.phi[nl_sigma:2*nl_sigma+1],MPI.DOUBLE], dest=size-1, tag=0+32)
                    comm.Recv([worker.g.phi[:nl_sigma],MPI.DOUBLE], source=size-1, tag=1+32)
                    comm.Send([worker.g.grad_phi[nl_sigma:2*nl_sigma+1],MPI.DOUBLE], dest=size-1, tag=2+32)
                    comm.Recv([worker.g.grad_phi[:nl_sigma],MPI.DOUBLE], source=size-1, tag=3+32)
                if rank == size -1 :
                    comm.Send([worker.g.phi[nR-nl_sigma-1:nR-1],MPI.DOUBLE], dest=0, tag=1+32)
                    comm.Recv([worker.g.phi[nR-1:],MPI.DOUBLE], source=0, tag=0+32)
                    comm.Send([worker.g.grad_phi[nR-nl_sigma-1:nR-1],MPI.DOUBLE], dest=0, tag=3+32)
                    comm.Recv([worker.g.grad_phi[nR-1:],MPI.DOUBLE], source=0, tag=2+32)
        elif BC == 'ZeroPointBC' :
            def apply_BC():
                if rank == 0 :
                    worker.apply_left_MirrorBC(worker.g.phi, inverted=True)
                    worker.apply_left_MirrorBC(worker.g.grad_phi, inverted=False)
                if rank == size-1 :
                    worker.apply_right_MirrorBC(worker.g.phi, inverted=True)
                    worker.apply_right_MirrorBC(worker.g.grad_phi, inverted=False)
        else:
            def apply_BC():
                return
        def update_shared_BC():
            # WL :: Shared Right, Receive Right from Left
            if rank != size-1 :
                comm.Send([worker.shared_right_BC(worker.g.phi),MPI.DOUBLE], dest=rank+1, tag=0+16)
                comm.Recv([worker.g.phi[worker.nR:],MPI.DOUBLE], source=rank+1, tag=1+16)
                comm.Send([worker.shared_right_BC(worker.g.grad_phi),MPI.DOUBLE], dest=rank+1, tag=2+16)
                comm.Recv([worker.g.grad_phi[worker.nR:],MPI.DOUBLE], source=rank+1, tag=3+16)
            # WR :: Shared Left, Receive Left from Right
            if rank != 0 :
                comm.Send([worker.shared_left_BC(worker.g.phi),MPI.DOUBLE], dest=rank-1, tag=1+16)
                comm.Recv([worker.g.phi[:worker.nL],MPI.DOUBLE], source=rank-1, tag=0+16)
                comm.Send([worker.shared_left_BC(worker.g.grad_phi),MPI.DOUBLE], dest=rank-1, tag=3+16)
                comm.Recv([worker.g.grad_phi[:worker.nL],MPI.DOUBLE], source=rank-1, tag=2+16)
    else:
        if BC == 'PeriodicBC' :
            def apply_BC():
                nR = len(worker.g.phi) - nl_sigma
                if rank == 0 :
                    comm.Send([worker.g.phi[nl_sigma:2*nl_sigma+1],MPI.DOUBLE], dest=size-1, tag=0+32)
                    comm.Recv([worker.g.phi[:nl_sigma],MPI.DOUBLE], source=size-1, tag=1+32)
                    comm.Send([worker.g.grad_phi[nl_sigma:2*nl_sigma+1],MPI.DOUBLE], dest=size-1, tag=2+32)
                    comm.Recv([worker.g.grad_phi[:nl_sigma],MPI.DOUBLE], source=size-1, tag=3+32)
                    comm.Send([worker.g.curv_phi[nl_sigma:2*nl_sigma+1],MPI.DOUBLE], dest=size-1, tag=4+32)
                    comm.Recv([worker.g.curv_phi[:nl_sigma],MPI.DOUBLE], source=size-1, tag=5+32)
                if rank == size -1 :
                    comm.Send([worker.g.phi[nR-nl_sigma-1:nR-1],MPI.DOUBLE], dest=0, tag=1+32)
                    comm.Recv([worker.g.phi[nR-1:],MPI.DOUBLE], source=0, tag=0+32)
                    comm.Send([worker.g.grad_phi[nR-nl_sigma-1:nR-1],MPI.DOUBLE], dest=0, tag=3+32)
                    comm.Recv([worker.g.grad_phi[nR-1:],MPI.DOUBLE], source=0, tag=2+32)
                    comm.Send([worker.g.curv_phi[nR-nl_sigma-1:nR-1],MPI.DOUBLE], dest=0, tag=5+32)
                    comm.Recv([worker.g.curv_phi[nR-1:],MPI.DOUBLE], source=0, tag=4+32)
        elif BC == 'ZeroPointBC' :
            def apply_BC():
                if rank == 0 :
                    worker.apply_left_MirrorBC(worker.g.phi, inverted=True)
                    worker.apply_left_MirrorBC(worker.g.grad_phi, inverted=False)
                    worker.apply_left_MirrorBC(worker.g.curv_phi, inverted=True)
                if rank == size-1 :
                    worker.apply_right_MirrorBC(worker.g.phi, inverted=True)
                    worker.apply_right_MirrorBC(worker.g.grad_phi, inverted=False)
                    worker.apply_right_MirrorBC(worker.g.curv_phi, inverted=True)
        else:
            def apply_BC():
                return
        def update_shared_BC():
            # WL :: Shared Right, Receive Right from Left
            if rank != size-1 :
                comm.Send([worker.shared_right_BC(worker.g.phi),MPI.DOUBLE], dest=rank+1, tag=0+16)
                comm.Recv([worker.g.phi[worker.nR:],MPI.DOUBLE], source=rank+1, tag=1+16)
                comm.Send([worker.shared_right_BC(worker.g.grad_phi),MPI.DOUBLE], dest=rank+1, tag=2+16)
                comm.Recv([worker.g.grad_phi[worker.nR:],MPI.DOUBLE], source=rank+1, tag=3+16)
                comm.Send([worker.shared_right_BC(worker.g.curv_phi),MPI.DOUBLE], dest=rank+1, tag=4+16)
                comm.Recv([worker.g.curv_phi[worker.nR:],MPI.DOUBLE], source=rank+1, tag=5+16)
            # WR :: Shared Left, Receive Left from Right
            if rank != 0 :
                comm.Send([worker.shared_left_BC(worker.g.phi),MPI.DOUBLE], dest=rank-1, tag=1+16)
                comm.Recv([worker.g.phi[:worker.nL],MPI.DOUBLE], source=rank-1, tag=0+16)
                comm.Send([worker.shared_left_BC(worker.g.grad_phi),MPI.DOUBLE], dest=rank-1, tag=3+16)
                comm.Recv([worker.g.grad_phi[:worker.nL],MPI.DOUBLE], source=rank-1, tag=2+16)
                comm.Send([worker.shared_left_BC(worker.g.curv_phi),MPI.DOUBLE], dest=rank-1, tag=5+16)
                comm.Recv([worker.g.curv_phi[:worker.nL],MPI.DOUBLE], source=rank-1, tag=4+16)

    def forward_source(u_old, dt):
        u_new = np.empty_like(u_old)
        A = 0.25*(1. - (2*u_old - 1.)**(-2) )
        for i in range(len(u_old)):
            if u_old[i] > 0.5 :
                u_new[i] = 0.5*(1. + np.sqrt(np.power(1. - 4*A[i]*np.exp(-3*dt), -1)))
            else:
                u_new[i] = 0.5*(1. - np.sqrt(np.power(1. - 4*A[i]*np.exp(-3*dt), -1)))
        return u_new


    if not Is_Source_Experiment :
        def run_worker():
            # u_t + u u_x = u_xx
            worker.forward(worker.g.integral_phi(0, len(worker.g.phi)), shifted=True)
    else:
        def run_worker():
            # u_t + u u_x = u_xx
            worker.forward(worker.g.integral_phi(0, len(worker.g.phi)), shifted=True)

            # u_t = -3 u (1-u) (1-2u)
            worker.g.phi = forward_source(worker.g.phi, dt) # ***** FOR THE ADDITONAL TERM *****

    # * * * * *

    if rank == root :

        # Initialization
        x, dx = np.linspace(xL-dx*nl_sigma, xR+dx*nl_sigma, nx+2*nl_sigma, retstep=True)
        phi_init = IC_func(x, t0, K)

        # ------------------------

        print("\n------------- Start -------------\n")

        print('Number of grid point =', nx)
        print('Range of influence (l) =', l)
        print('Grid spacing (dx) =', dx)
        print('Non-dimensional diffusion coefficient =', d_coeff)
        print('Append', nl_sigma,'cells to both ends')
        if 2*nl_sigma > nx :
            print('Warning !! resulting nl_sigma seem to be too large compared to n')
        print('Start running',grid,'for', nt,'iteration with time step size =',dt)
        print('-----')

    # -----------------------------------

    if rank == root :
        print('Start MPI initial data transfer --- ')
        starttime_init = timer()

        stride = int(np.floor(nx/size))
        workload = np.empty(size+1, dtype=int)
        workload[0] = 0
        for r in range(size):
            if r < nx - stride*size :
                workload[r+1] = stride + 1
            else:
                workload[r+1] = stride
        workload = np.cumsum(workload)
        if workload[-1] != nx :
            print('Error found in the domain decomposition algorithm, i.e., workload[-1] != nx')

    # * * * * *

    # MPI Send-Receive == Initialization == 1 work per rank
        for i in range(size):
            x0   = x[workload[i]] # (workload[i]+nl_sigma)-nl_sigma
            phi0 = phi_init[workload[i]:workload[i+1]+2*nl_sigma]
            if i != root :
                comm.send(x0, dest=i, tag=0)
                comm.Send([phi0, MPI.DOUBLE], dest=i, tag=1)

    if rank != root :
        x0 = comm.recv(source=root, tag=0)
        stride = int(np.floor(nx/size))
        if rank < nx - stride*size :
            stride = stride + 1
        phi0 = np.empty(stride+2*nl_sigma, dtype=float)
        comm.Recv([phi0,MPI.DOUBLE], source=root, tag=1)

    # to Non-dimensional space
    phi0 /= K
    Kdt = K*dt
    # Initialize phi0, grad_phi0, curv_phi0
    worker = worker_init(x0, phi0)
    if rank == 0 or rank == size-1 :
        apply_BC()

    # * * * * *

    if rank == root :
        endtime_init = timer()
        print("Finish MPI initial data transfer with elapsed time (sec) =", (endtime_init-starttime_init)/1E9)
        print('>>> Total elapsed time (sec) =', (endtime_init-starttime_init)/1E9,'\n')

    # -----------------------------------

    if rank == root :
        print('Start running',grid,'scheme')
        starttime = timer()

    # * * * * *

    # MAIN == MPI run
    for it in range(nt):
        run_worker()

        comm.Barrier()
        update_shared_BC()
        if rank == 0 or rank == size-1 :
            apply_BC()

    comm.Barrier()

    # * * * * *

    if rank == root :
        endtime = timer()
        print('Finish running',grid,'with elapsed time (sec) =', (endtime-starttime)/1E9)
        print('>>> Total elapsed time (sec) =', (endtime-starttime_init)/1E9,'\n')

    # -----------------------------------

    if rank == root :
        print('Start MPI final data transfer --- ')
        starttime_final = timer()

    # * * * * *

    # Back to normal space
    worker.g.phi[worker.nL:worker.nR] *= K
    if grid == 'CG' :
        worker.g.grad_phi[worker.nL:worker.nR] *= K
    elif grid == 'QG' :
        worker.g.grad_phi[worker.nL:worker.nR] *= K
        worker.g.curv_phi[worker.nL:worker.nR] *= K

    # MPI Send-Receive == Collect result --> to Rank 0
    if rank != root :
        comm.Send([worker.g.phi[worker.nL:worker.nR],MPI.DOUBLE], dest=root, tag=0)
        if grid == 'CG' :
            comm.Send([worker.g.grad_phi[worker.nL:worker.nR],MPI.DOUBLE], dest=root, tag=1)
        elif grid == 'QG' :
            comm.Send([worker.g.grad_phi[worker.nL:worker.nR],MPI.DOUBLE], dest=root, tag=1)
            comm.Send([worker.g.curv_phi[worker.nL:worker.nR],MPI.DOUBLE], dest=root, tag=2)

    if rank == root :
        phi_numer = np.empty(nx, dtype=float)
        for i in range(size-1):
            #if i != root : <-- root = size-1
            comm.Recv(phi_numer[workload[i]:workload[i+1]], source=i, tag=0)
        phi_numer[workload[-2]:workload[-1]] = worker.g.phi[worker.nL:worker.nR]

        # Numerical solution
        main = worker_init(x[nl_sigma], phi_numer)

        if grid == 'CG' :
            for i in range(size-1):
                comm.Recv([main.g.grad_phi[workload[i]:workload[i+1]],MPI.DOUBLE], source=i, tag=1)
            main.g.grad_phi[workload[-2]:workload[-1]] = worker.g.grad_phi[worker.nL:worker.nR]
        elif grid == 'QG' :
            for i in range(size-1):
                comm.Recv([main.g.grad_phi[workload[i]:workload[i+1]],MPI.DOUBLE], source=i, tag=1)
                comm.Recv([main.g.curv_phi[workload[i]:workload[i+1]],MPI.DOUBLE], source=i, tag=2)
            main.g.grad_phi[workload[-2]:workload[-1]] = worker.g.grad_phi[worker.nL:worker.nR]
            main.g.curv_phi[workload[-2]:workload[-1]] = worker.g.curv_phi[worker.nL:worker.nR]

    # * * * * *

    if rank == root :
        endtime_final = timer()
        print("Finish MPI final data transfer with elapsed time (sec) =", (endtime_final-starttime_final)/1E9)
        print('>>> Total elapsed time (sec) =', (endtime_final-starttime_init)/1E9)


    # ------------------------

    if rank == root and cal_error and sampling_pos != None :

        # Analytical solution
        phi_ana = lambda x : Ana_func(x, t0+dt*nt, K)

        print('-----')
        print('')
        print('-- Values at sampling_point --')

        result = np.zeros([2, len(sampling_pos)])
        result[0,:] = find_phi_at(sampling_pos, main.g)
        result[1,:] = phi_ana(np.array(sampling_pos))

        print('Position & $\\Delta x$ & Result & Exact \\\\')
        print('\\hline')
        for i in range(len(sampling_pos)):
            print('%.1f & %.5f & %f & %f \\\\' % (sampling_pos[i], dx, result[0,i], result[1,i]))
        print('\\hline')

        # -------------------------------

        print('')
        print('-- Errors [LX] (every grid point) --')
        result = np.zeros(3)
        g_phi_ana = phi_ana(main.g.x())
        result[0] = L1ErrorNorm(main.g.phi, g_phi_ana)
        result[1] = L2ErrorNorm(main.g.phi, g_phi_ana)
        result[2] = LinfErrorNorm(main.g.phi, g_phi_ana)

        print('$\\Delta x$ & $\\Delta t$ & d & L1 & L2 & Linf \\\\')
        print('\\hline')
        print('%f ' % (dx), end='')
        print('& %f ' % (dt), end='')
        print('& %f ' % (d_coeff), end='')
        print('& %.2e & %.2e & %.2e \\\\' % (result[0], result[1], result[2]))

    if rank == root and savefig != None :
        plt.figure(0, figsize=(9.6, 4.8))
        plt.clf()
        if cal_error :
            plt.plot(main.g.x(), g_phi_ana, 'k', label='Ana')
        plt.plot(main.g.x(), main.g.phi, 'b:x', markersize=6, label=grid)
        plt.legend()
        plt.grid(True, linestyle=':', linewidth=1)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.tight_layout()
        plt.savefig(savefig, dpi=320)

    if rank == root :
        print("\n------------- End -------------\n")
        return main.g.phi, main.g.x()
    else:
        return None, None


