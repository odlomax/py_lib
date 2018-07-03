import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sph_ic.particle_ensemble import particles

seed=2

#A=particles(10000,10,0.2,10,0.5,H_fbm=0.5,density_scale="power-law",alpha_fbm=2.5,r_seed=seed)


B=particles(10000,10,0.2,10,0.5,H_fbm=0.0,density_scale="lognormal",sigma_fbm=1.39,r_seed=seed)





H_list=[0.0,0.25,0.5,0.75,1.]
alpha_list=[2,3,4,5]
rho_bins=np.exp(np.linspace(-10,10,500))
drho=rho_bins[1:]-rho_bins[:-1]

"""
plt.figure(1)
C=[]
for i in range(len(alpha_list)):
    
    C.append(particles(50,10,0.2,10,0.5,H_fbm=0.5,density_scale="power-law",alpha_fbm=alpha_list[i],r_seed=seed,n_grid=100,n_dim=3))
    C_hist,C_bins=np.histogram(C[i].rho.flatten(),rho_bins)
    plt.plot(np.log10(rho_bins[:-1]),np.log10(C_hist/drho))
    
"""

"""

plt.figure(2)
sigma_list=[0.25,0.5,1.0,2.0]


D=[]
for i in range(len(sigma_list)):

    D.append(particles(50,10,0.2,10,0.8,H_fbm=0.5,density_scale="lognormal",sigma_fbm=sigma_list[i],r_seed=seed,n_grid=100))
    D_hist,D_bins=np.histogram(D[i].rho.flatten(),rho_bins)
    plt.plot(np.log(rho_bins[:-1]),np.log10(C_hist/drho))

"""

"""
plt.figure(2)
D=[]
for i in range(len(H_list)):
    
    D.append(particles(50,10,0.2,10,0.5,H_fbm=H_list[i],density_scale="lognormal",sigma_fbm=1.4,r_seed=seed))
    D_hist,D_bins=np.histogram(np.log(D[i].rho.flatten()),bins)
    plt.plot((0.5*(D_bins[:-1]+D_bins[1:])),np.log(D_hist))

"""

"""

fig_1=plt.figure()
ax = fig_1.add_subplot(111, projection='3d')
ax.scatter(A.r[:,0],A.r[:,1],A.r[:,2],marker=".",depthshade=False,s=1.5)

"""


fig_2=plt.figure()
ax = fig_2.add_subplot(111, projection='3d')
ax.scatter(B.r[:,0],B.r[:,1],B.r[:,2],marker=".",depthshade=False,s=1.5)