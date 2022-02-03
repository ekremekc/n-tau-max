"""
A note on the nomenclature:
dim ~ dimensional quantity
ref ~ reference quantity for the non-dimensionalization
in ~ inlet, same as u ~ upstream (of the flame)
out ~ outlet, same as d ~ downstream (of the flame)
"""

from math import *
import numpy as np
from dolfinx.fem import Function,FunctionSpace
from dolfinx.mesh import create_rectangle
from mpi4py import MPI
from dolfinx.mesh import MeshTags, locate_entities

from numpy.linalg import lstsq
import matplotlib.pyplot as plt
# ------------------------------------------------------------

L_ref = 1.  # [m]

# ------------------------------------------------------------

r = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.16  # [kg/m^3] 1.16 gives 300K inlet temp

T_amb = p_amb/(r*rho_amb)  # [K]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_in_dim = rho_amb  # [kg/m^3]
rho_out_dim = 0.385  # [kg/m^3] # default was 0.8, 0.38 gives 916K

T_in_dim = p_amb/(r*rho_in_dim)  # [K]
T_out_dim = p_amb/(r*rho_out_dim)  # [K]

print("Inlet Temp: ",T_in_dim, "Outlet Temp: ", T_out_dim)
c_in_dim = sqrt(gamma*p_amb/rho_in_dim)  # [kg/m^3]
c_out_dim = sqrt(gamma*p_amb/rho_out_dim)  # [kg/m^3]

print(c_in_dim, c_out_dim )
# Reflection coefficients

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]

# Specific impedance

Z_in = (1 + R_in)/(1 - R_in)
Z_out = (1 + R_out)/(1 - R_out)

# print('Z_in =', Z_in)
# print('Z_out =', Z_out)

# Specific admittance

Y_in = 1/Z_in
Y_out = 1/Z_out


# ------------------------------------------------------------
L = 1387  * 1e-3 #mm length of combustor default:1387mm
H = 152.4* 1e-3 #mm height of combustor default:152.4mm
W = 127   * 1e-3 #mm width  of combustor default:127mm
D = 38.1 * 1e-3 # mm
m_dot = 0.35 # kg/s
A = W * (H ) # m2 cross sectional area

U = m_dot / (rho_amb * A) # m/s
print("U bulk:", U)
Q = 5000 # Watt

flame_location = 864/1387 #[-]

x_f = np.array([[L*flame_location, 0., 0.]])  # [m]
a_f = 0.025  # default:0.025[m]
x_r = np.array([[L*flame_location, 0., 0.]])  # [m]

mesh = create_rectangle(MPI.COMM_WORLD,
                        [np.array([0, -H/2]), np.array([L, +H/2])],
                        [956, 603]     )
# if no interpolation ->  mesh size is [956, 603]                           

def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_fl = x_f[0][0]
    a_fl = a_f
    return np.logical_and(x_fl - a_fl - eps <= x, x <= x_fl + a_fl + eps)

tdim = mesh.topology.dim
marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
fl = 0
subdomains = MeshTags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], L)),
              (3, lambda x: np.isclose(x[1], -H/2)),
              (4, lambda x: np.isclose(x[1], +H/2))]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tags = MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "paraview/mf.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(subdomains)

V = FunctionSpace(mesh, ('CG',1))


tau = Function(V)
n = Function(V)

import data_read as datas

tt = datas.final_tt * -1
nn = datas.final_n

(row, col) = tt.shape
for i in range(row):
    for j in range(col):
        if tt[i][j] < 0.0:
            tt[i][j] = 0.0 
        if nn[i][j] < 0.0:
            nn[i][j] = 0.0 

print("Size of the n-tau data: ",tt.shape)

mesh_coord_V = V.tabulate_dof_coordinates()

ind = np.lexsort((mesh_coord_V[:,0],mesh_coord_V[:,1]))    

tau_flatten = tt.flatten() 
tau.x.array[ind] = tau_flatten

n_flatten = nn.flatten()
n.x.array[ind] = n_flatten


from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "paraview/new_tau.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(tau)
    

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "paraview/new_n.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(n)



U_ref = c_amb  # [m/s]
p_ref = p_amb  # [Pa]

# ------------------------------------------------------------

rho_in = rho_in_dim*U_ref**2/p_ref
rho_out = rho_out_dim*U_ref**2/p_ref

T_in = T_in_dim*r/U_ref**2
T_out = T_out_dim*r/U_ref**2

T_gas = 300 #K
T_top = 300 #K
T_bottom = 300 #K

T_top_increment = 1500 #K 
T_bottom_increment = 1500 #K
T_peak = 1700 #K

def temperature(mesh):
    V = FunctionSpace(mesh, ("DG", 0))
    temp = Function(V)
    x = V.tabulate_dof_coordinates()
    x_fl = x_f[0][0]
    a=D/2
    b=10
    c=0
    L_end = L/100
    # We scale a and b from flame location(x_f) to end of the tube(L) 
    b_start, b_end = 1, 0.05
    points = [(x_f[0][0], b_start),(L+L_end, b_end)]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m, n = lstsq(A, y_coords, rcond=None)[0]
    # print("Line Solution is y = {m}x + {c}".format(m=m,c=n))
    # print(x_f[0][0]*m+n)
    a_start, a_end = 1, 3
    points2 = [(x_f[0][0], a_start),(L+L_end, a_end)]
    x_coords2, y_coords2 = zip(*points2)
    A2 = np.vstack([x_coords2,np.ones(len(x_coords2))]).T
    m2, n2 = lstsq(A2, y_coords2, rcond=None)[0]
    # print("For a: ", m2,"x + ",n2)
    # print("For b: ", m,"x + ",n)
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[0]< x_fl:
            temp.vector.setValueLocal(i, T_gas)
        else:
            decay = m*midpoint[0]+n
            decay2 = m2*midpoint[0]+n2
            value = T_gas + T_peak/(1+np.abs((midpoint[1] - c ) / (a*decay2))**(2*b*decay))
            temp.vector.setValueLocal(i, value)
            # temp.vector.setValueLocal(i, T_peak)

    return temp

def sound_speed(mesh):
    temp = temperature(mesh)
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    c.x.array[:] =  20.05 * np.sqrt(temp.x.array)
    return c

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "paraview/T.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(temperature(mesh))
    

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "paraview/c.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(sound_speed(mesh))