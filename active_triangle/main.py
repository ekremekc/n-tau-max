
from helmholtz_x.helmholtz_pkgx.active_flame_x import ActiveFlameNT

from dolfinx.mesh import meshtags, locate_entities
from mpi4py import MPI
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import pep_solver, fixed_point_iteration_ntau
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.geometry_pkgx.xdmf_utils import load_xdmf_mesh, write_xdmf_mesh
from helmholtz_x.helmholtz_pkgx.dolfinx_utils import interpolator
from triangle_geom import geometry

import dolfinx.io
import numpy as np
import params
# Generate mesh

if MPI.COMM_WORLD.rank == 0:
    geometry(fltk=False)
write_xdmf_mesh("MeshDir/triangle",dimension=2)
# Read mesh 
mesh, cell_tags, facet_tags = load_xdmf_mesh("MeshDir/triangle")                        

def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_fl = params.x_f[0][0]
    a_fl = params.a_f
    return np.logical_and(x_fl - a_fl - eps  <= x, x <= x_fl + a_fl + eps)

tdim = mesh.topology.dim
marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
fl = 0
subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

# Define the boundary conditions

boundary_conditions = {5: {'Neumann'},
                       4: {'Neumann'},
                       3: {'Dirichlet'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Define Speed of sound
# c = dolfinx.Constant(mesh, PETSc.ScalarType(1))
c = params.sound_speed(mesh)

deg = 1

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree =deg)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

# # -------------------------- TAU AND N DATA - INTERPOLATION ---------------------------------

tau = interpolator(params.xs, params.ys, params.tt, mesh)
n = interpolator(params.xs, params.ys, params.nn, mesh)

from datetime import datetime
before = datetime.now()
target = 150 * 2 * np.pi

E = fixed_point_iteration_ntau(matrices, target, mesh, subdomains,
                    params.x_r, params.rho_in, params.Q, params.U,
                    n, tau, nev=2, i=0, print_results= False)


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

targets = [150, 300]


# for target in targets:

#     from datetime import datetime
#     before = datetime.now()
#     target *= 2 * np.pi
#     E = fixed_point_iteration_1(matrices, target, nev=2, i=0, print_results= False)
    

#     omega, p = normalize_eigenvector(mesh, E, 0, degree=1, which='right')
#     frequency = omega / (2 * np.pi)
#     print("The mode frequency is: ",frequency)

#     from dolfinx.io import XDMFFile
#     p.name = "Acoustic_Wave"
#     mode_name = "Results/" + str(int(frequency.real)) + "Hz.xdmf"
#     with XDMFFile(MPI.COMM_WORLD, mode_name, "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(p)
#     print("Computation time for this mode: ",(datetime.now()-before))


### CALCULATION OF RAYLEIGH INDEX

from dolfinx.fem import assemble_scalar, Function, FunctionSpace, form
from ufl import dx
tau_omega = Function(FunctionSpace(mesh, ("Lagrange", 1)))
tau_omega.x.array[:] = np.exp(omega*1j*tau.x.array) # tau^{i* \omega * \tau}

Rayleigh_Index = assemble_scalar(form(p*n*tau_omega*dx))

print("Rayleigh_Index is: ", Rayleigh_Index)
