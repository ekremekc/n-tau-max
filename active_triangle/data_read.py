from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat("data_600.mat")

nn_lsgen = data["nn_lsgen"] 
tt_unwrapped = data["tt_unwrapped"] 

# plt.contourf(tt_unwrapped)
# plt.colorbar()
# plt.show()

nr, nc = nn_lsgen.shape

hv = 0.10;         # Height of v-shaped bluff body (m)

# Set the number of points upstream and downstream of the flame
nu  = nc # 
# ------------------------------------------------------------
L = 1387 #mm length of combustor default:1387mm
H = 152.4 #mm height of combustor default:152.4mm

flame_location = 864/1387 #[-]

x_shape = int(nc/(1-flame_location))
x = np.linspace(0, 1, x_shape, endpoint=True)

# Set the y abscissa
y = hv*(np.arange(0,nr+1,1)/nr)

# Calculate the n field
nn = np.zeros((len(y),x_shape))
nn[1:nr+1,(x_shape-nc):] = nn_lsgen

# Calculate the tau field
tt = np.zeros((len(y),x_shape))# initialize
tt[1:nr+1,(x_shape-nc):] = tt_unwrapped

final_tt = np.concatenate((np.flipud(tt), tt),axis=0)
final_n = np.concatenate((np.flipud(nn), nn),axis=0)

# plt.contourf(final_tt)
# plt.colorbar()
# plt.show()
