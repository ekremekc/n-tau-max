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

#  (THERE ARE GUESSES. YOU WILL NEED TO GET THE EXACT VALUES FROM THE PAPERS
#  AND FROM TIM GALLAGHER AT AFOR)

hv = 0.10;         # Height of v-shaped bluff body (m)

# Set the number of points upstream and downstream of the flame
nu  = nc # it was 181
# nd  = 181

scaler = 1 # np.pi/2 # 10.2 is default
# ------------------------------------------------------------
L = 1387 #mm length of combustor default:1387mm
H = 152.4 #mm height of combustor default:152.4mm

flame_location = 864/1387 #[-]

x_shape = int(nc/(1-flame_location))

x = np.linspace(0, 1, x_shape, endpoint=True)

# Set the y abscissa
y = hv*(np.arange(0,nr+1,1)/nr)
# Create a grid with these abscissae
[xx,yy] = np.meshgrid(x,y)

# Calculate the n field
nn = np.zeros_like(xx) # initialize
nn[1:nr+1,(x_shape-nc):] = nn_lsgen

# Calculate the tau field
tt = np.zeros_like(xx) # initialize
tt[1:nr+1,(x_shape-nc):] = tt_unwrapped
# print(tt, tt.shape)

final_tt = np.concatenate((np.flipud(tt), tt),axis=0)

final_n = np.concatenate((np.flipud(nn), nn),axis=0)

# plt.contourf(final_tt)
# plt.colorbar()
# plt.show()
