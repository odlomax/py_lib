import numpy as np
from matplotlib import pyplot as plt
from astropy import units
from astropy import constants
from sph_ic.particle_ensemble import particles


# set number of initial conditions
N_ic=10

# set number of particles
N_part=1000000

# set primary physical properties
M=10.           # Mass (M_sun)
T_bg=10.        # Background temperature (K)
Q_vir=0.5       # Virial ratio
m_bar=2.353     # Mean molecular mass (u)
Ma=3.           # Mach number
H_fbm=1.        # Hurst index
alpha_turb=2.   # velocity field spectral index

# set run id
run_id="particles"

# calculate secondary physical properties

# velocity dispersion (km s^-1)
sigma_v=(Ma*np.sqrt(T_bg*units.Unit("K")*constants.k_B/(m_bar*units.Unit("u")))).to("km s^-1").value

# density logarithmic standard deviation
sigma_fbm=np.log(Ma**2)

for i in range(N_ic):
    print(i+1,"of",N_ic)
    
    # initialise particle ensemble
    sph_particles=particles(N_part,M,sigma_v,T_bg,Q_vir,H_fbm=H_fbm,density_scale="lognormal",sigma_fbm=sigma_fbm,r_seed=i)
    sph_particles.save_to_file(run_id+"_"+str(i+1).zfill(3)+".ascii",run_id+"_"+str(i+1).zfill(3)+".pickle")
    
    



"""
fig_1=plt.figure()
ax = fig_1.add_subplot(111, projection='3d')
ax.scatter(sph_particles.r[::1,0],sph_particles.r[::1,1],sph_particles.r[::1,2],marker=".",depthshade=False,s=1.5)
"""