from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sph_ic.particle_ensemble import particles

# want random positions, not SPH glass
glass=False

# number of stars
N_part=5000

# mass of cluster (M_sun)
M=1.

# velocity dispersion (km s^-1)
sigma_v=0.2

# temperature (zero for star cluster)
T=0.

# virial parameter (sets R)
Q_vir=0.5

# fBm H exponent
H_fbm=0.5

# density scale (can be "power-law" or "lognormal")
density_scale="lognormal"
sigma_fbm=2. # scale if lognormal
alpha_fbm=3. # scale if power-law

# velocity stucture power specturm P(k)~k^-alpha
# alpha_turb=-2. is white noise (Maxwell-Boltzmann distribution)
# alpha_turb=2. is "turbulence" spectrum used in most hydro simulations
# Note: -velocity and density stucture are uncorrelated
#       -probably best to set individual random velocities afterwards if you want M-B distribution
alpha_turb=2.


A=particles(N_part,M,sigma_v,T,Q_vir,H_fbm=H_fbm,density_scale=density_scale,sigma_fbm=sigma_fbm,
            alpha_fbm=alpha_fbm,alpha_turb=alpha_turb,glass=glass)



fig_1=plt.figure()
ax = fig_1.add_subplot(111, projection='3d')
ax.scatter(A.r[:,0],A.r[:,1],A.r[:,2],marker=".",depthshade=False,s=1.5)
ax.set_title("density structure")


stride=5
fig_2=plt.figure()
ax = fig_2.add_subplot(111, projection='3d')
ax.quiver(A.r[::stride,0],A.r[::stride,1],A.r[::stride,2],
        A.v[::stride,0],A.v[::stride,1],A.v[::stride,2],length=0.035)
ax.set_title("velocity structure")


# save to file
A.save_to_file("cluster.dat")