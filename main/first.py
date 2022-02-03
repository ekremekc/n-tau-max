from dolfinx.fem import Function, FunctionSpace, Constant
from ufl import TrialFunction, TestFunction, dx, inner
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


from dolfinx.fem.assemble import assemble_vector, assemble_scalar
import n_tau_dolfinx as data

n = data.n
tau = data.tau
mesh = data.mesh

V = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
phi_k = TestFunction(V)

V_fl = MPI.COMM_WORLD.allreduce(assemble_scalar(Constant(mesh, PETSc.ScalarType(1))*dx), op=MPI.SUM)
b = Function(V)
b.x.array[:] = 0
const = Constant(mesh, (1/V_fl))

omega = 3+2j

original_tau = 0.003 #s
tau_func = Function(V)
tau_func.x.array[:] = np.exp(omega*1j*tau.x.array) 
print(len(tau_func.x.array[:]), len(np.exp(omega*1j*tau.x.array)) )
from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "tau_func.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(tau_func)

a = assemble_vector(b.vector, n * tau_func * inner(const, phi_k)*dx)
# vector = assemble_vector(n*v*dx)
