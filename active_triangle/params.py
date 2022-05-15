from dolfinx.fem import Function,FunctionSpace
from dolfinx.io import XDMFFile
from numpy.linalg import lstsq
from math import *

import matplotlib.pyplot as plt
import numpy as np
import data_read as data

# ----------------------- INITIAL PARAMETERS   ------------------------------------------------

rho_in = 1.16  # [kg/m^3] 1.16 gives 300K inlet temp
rho_out = 0.385  # [kg/m^3] # default was 0.8, 0.38 gives 916K

# Reflection coefficients
R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]

# Specific impedance
Z_in = (1 + R_in)/(1 - R_in)
Z_out = (1 + R_out)/(1 - R_out)

Y_in = 1/Z_in
Y_out = 1/Z_out

# -------------------------- MESH - BOUNDARIES  - SUBDOMAINS ----------------------------------

L = 1387  * 1e-3 #mm length of combustor default:1387mm
H = 152.4* 1e-3 #mm height of combustor default:152.4mm
W = 127   * 1e-3 #mm width  of combustor default:127mm
D = 38.1 * 1e-3 # mm one edge of bluff body
A = W * (H ) # m2 cross sectional area

m_dot = 0.35 # kg/s
U = m_dot / (rho_in * A) # m/s
Q = 500000 # Watt
flame_location = 864 * 1e-3 #mm

x_f = np.array([[flame_location, 0., 0.]])  # [m]
a_f = 0.025  # default:0.025[m]
x_r = np.array([[flame_location, 0., 0.]])  # [m]

# # -------------------------- TAU AND N DATA - INTERPOLATION ---------------------------------

tt = data.final_tt*(-1)
tt *= (tt>0)

(size_y_old, size_x_old) = tt.shape
number_of_rows_del = 25
rows_to_be_deleted = np.arange(int(size_y_old/2)-number_of_rows_del,int(size_y_old/2)+number_of_rows_del)
tt = np.delete(tt,(rows_to_be_deleted),axis=0)

nn = data.final_n
nn *= (nn>0)
nn = np.delete(nn,(rows_to_be_deleted),axis=0)

(size_y, size_x) = tt.shape
xs = np.linspace(0, L, size_x)
ys = np.linspace(-H/2,H/2,size_y)

# # -------------------------- TEMPERATURE AND SPEED OF SOUND  ---------------------------------
T_gas = 300 #K
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

    return temp

def sound_speed(mesh):
    temp = temperature(mesh)
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    c.x.array[:] =  20.05 * np.sqrt(temp.x.array)
    return c
