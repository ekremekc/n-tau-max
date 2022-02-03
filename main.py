
from mpi4py import MPI
import numpy as np
from dolfinx.fem import Constant
from petsc4py import PETSc

from helmholtz_x.helmholtz_pkgx.eigensolvers_x import pep_solver, eps_solver
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from active_flame_n_x import ActiveFlame
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
import params


mesh, facet_tags, subdomains = params.mesh, params.facet_tags, params.subdomains



# boundary_conditions = {4: {'Neumann'},
#                        2: {'Robin': params.Y_out},
#                        3: {'Neumann'},
#                        1: {'Robin': params.Y_in}}
#2: {'Dirichlet'}
boundary_conditions = {4: {'Neumann'},
                       2: {'Dirichlet'},
                       3: {'Neumann'},
                       1: {'Neumann'}}
degree = 1

# Define Speed of sound
c =  params.sound_speed(mesh) #Constant(mesh, PETSc.ScalarType(400))#  

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c , degree = degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

# target = 150 # Hz 250Hz for 2nd mode 

targets = [150, 280]


def fixed_point_iteration_1(operators, target, nev=2, i=0,
                              tol=1e-8, maxiter=50,
                              print_results=False,
                              problem_type='direct'):

    A = operators.A
    C = operators.C
    B = operators.B
    if problem_type == 'adjoint':
        B = operators.B_adj

    omega = np.zeros(maxiter, dtype=complex)
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)
    E = pep_solver(A, B, C, target, nev, print_results=print_results)
    vr, vi = A.getVecs()
    eig = E.getEigenpair(i, vr, vi)
    omega[0] = eig
    alpha[0] = .5

    domega = 2 * tol
    k = - 1

    # formatting
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)

    while abs(domega) > tol:

        k += 1

        D = ActiveFlame(mesh, subdomains,
                    params.x_r, params.rho_in, params.Q, params.U,
                    params.n, params.tau, omega[k], 
                    degree=degree)
        D.assemble_submatrices()
        D_Mat = D.matrix
        if problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix

        nlinA = A - D_Mat
        E = pep_solver(nlinA, B, C, target, nev, print_results=print_results)
        eig = E.getEigenpair(i, vr, vi)
        f[k] = eig

        if k != 0:
            alpha[k] = 1 / (1 - ((f[k] - f[k-1]) / (omega[k] - omega[k-1])))

        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        domega = omega[k+1] - omega[k]
        if MPI.COMM_WORLD.rank == 0:
            print('iter = {:2d},  omega = {}  {}j,  |domega| = {:.2e}'.format(
                k + 1, s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)
            ))

    return E

def fixed_point_iteration_2(operators, target, nev=2, i=0,
                              tol=1e-8, maxiter=50,
                              print_results=False,
                              problem_type='direct',
                              two_sided=False):

    A = operators.A
    C = operators.C
    B = operators.B
    if problem_type == 'adjoint':
        B = operators.B_adj

    omega = np.zeros(maxiter, dtype=complex)
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)

    E = eps_solver(A, C, target, nev, print_results=print_results)
    eig = E.getEigenvalue(i)
    print("First Eig is: ", np.sqrt(eig))

    omega[0] = np.sqrt(eig)
    alpha[0] = 0.5

    domega = 2 * tol
    k = - 1

    # formatting
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)

    while abs(domega) > tol:

        k += 1

        D = ActiveFlame(mesh, subdomains,
                    params.x_r, params.rho_in, params.Q, params.U,
                    params.n, params.tau, omega[k], 
                    degree=degree)
        print("worked 1")
        D.assemble_submatrices()
        D_Mat = D.matrix
        if problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix

        if not B:
            nlinA = A - D_Mat
        else:
            nlinA = A + (omega[k] * B) - D_Mat

        E = eps_solver(nlinA, C, target, nev, two_sided=two_sided, print_results=print_results)
        eig = E.getEigenvalue(i)

        f[k] = np.sqrt(eig)

        if k != 0:
            alpha[k] = 1/(1 - ((f[k] - f[k-1])/(omega[k] - omega[k-1])))

        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        domega = omega[k+1] - omega[k]
        if MPI.COMM_WORLD.rank == 0:
            print('iter = {:2d},  omega = {}  {}j,  |domega| = {:.2e}'.format(
                k + 1, s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)
            ))

    return E

for target in targets:

    from datetime import datetime
    before = datetime.now()
    target *= 2 * np.pi
    E = fixed_point_iteration_1(matrices, target, nev=2, i=0, print_results= False)
    # E = fixed_point_iteration_2(matrices, target**2, nev=2, i=0, print_results= False)

    omega, p = normalize_eigenvector(mesh, E, 0, degree=1, which='right')
    frequency = omega / (2 * np.pi)
    print("The mode frequency is: ",frequency)

    from dolfinx.io import XDMFFile
    p.name = "Acoustic_Wave"
    mode_name = "Results/" + str(int(frequency.real)) + "Hz.xdmf"
    with XDMFFile(MPI.COMM_WORLD, mode_name, "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(p)
    print("Computation time for this mode: ",(datetime.now()-before))