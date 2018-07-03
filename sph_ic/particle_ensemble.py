import numpy as np
import pickle
import os
from astropy import units
from astropy import constants
from scipy.interpolate import interpn
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from maths.fields.gaussian_random_field import scalar_grf
from maths.fields.gaussian_random_field import vector_grf
from maths.fields.potential_field import dft_potential
from maths.random.probability_density_function import pdf
from maths.points.custom_lattice import cube


# set directory of SPH glass file
glass_file=os.path.dirname(__file__)+"/glass_cube.dat"

class particles:
    
    """
    Class for building an ensemble of SPH particles (does not calc H or rho)
    
    """
    
    glass_file=os.path.dirname(__file__)+"/glass_cube.dat"
    
    def __init__(self,N,M,sigma_v,T,Q_vir,n_dim=3,n_grid=100,glass=True,m_bar=2.353,H_fbm=0.5,density_scale="lognormal",sigma_fbm=1.,alpha_fbm=3.,alpha_turb=2.,turb_type="thermal",
                 m_unit="M_sun",r_unit="pc",v_unit="km s^-1",T_unit="K",m_bar_unit="u",r_seed=None,direct_PE=False):
        
        """
        Subroutine: initialises particle ensemble
        
        Arguments
        ---------
        
        N: int
            number of particles
            
        M: float
            total mass of particles [M_sun]
        
        sigma_v: float
            velocity dispersion of particles [kms^-1]
            
        T: float
            temperature of gas [K]
            
        n_dim: int
            number of dimensions
        
        glass: boolean
            use SPH glass distribution if possible
            
        m_bar: float
            gas mean molecular mass [amu]
            
        H_fbm: float
            drift index of ln density field
            
        density_scale: string
            type of density scaling ("lognormal","power-law")
            
        sigma_fbm: float
            standard deviation of ln density field
            
        alpha_fbm:
            power-law density scaling exponent: p(rho) ~ rho**(-alpha_fbm)
            
        alpha_turb: float
            spectral index of velocity field
            
        turb_type: string
            type of velocity field ("thermal", "solenoidal" or "compressive")
            
        m_unit: string
            mass unit
            
        r_unit: string
            position unit
            
        v_unit: string
            velocity unit
            
        T_unit: string
            temperature unit
            
        m_bar_unit: string
            molecular mass unit
            
        r_seed: int
            random seed
            
        direct_PE: boolean
            directly calculate the gravitational potential energy of particles (very slow!)
            
        """
        
        # set parameters
        self.N=N
        self.M=M*units.Unit(m_unit)
        self.sigma_v=sigma_v*units.Unit(v_unit)
        self.T=T*units.Unit(T_unit)
        self.Q_vir=Q_vir
        self.n_dim=n_dim
        self.n_grid=n_grid
        self.glass=glass
        self.m_bar=m_bar*units.Unit(m_bar_unit)
        self.H_fbm=H_fbm
        self.density_scale=density_scale
        self.sigma_fbm=sigma_fbm
        self.alpha_fbm=alpha_fbm
        self.alpha_turb=alpha_turb
        self.turb_type=turb_type
        self.r_seed=r_seed

        # set random seed        
        np.random.seed(self.r_seed)
        
        # generate particle positions
        
        self.fbm_positions()
        
        # generate particle velocites
        
        self.turb_velocities()
        
        
        # calculate thermal energy
        self.TE=(1.5*self.M*self.T*constants.k_B/self.m_bar).to("J")
        
        # calculate kinetic energy
        self.KE=(0.5*self.M*self.sigma_v**2).to("J")
        
        # set target gravitational potential energy
        self.PE=-(self.KE+self.TE)/self.Q_vir
        
        # calculate R times gravitational potential energy
        self.R_PE=0.5*(((self.rho/self.rho.size)*self.phi).sum()*self.M**2*constants.G).to("J "+r_unit)
        
        # calculate size of distribution
        self.R=self.R_PE/self.PE
        
        # set size of particle ensemble
        self.r*=self.R.value
        self.r*=self.R.unit
        
        # set particle masses
        self.m=np.ones(self.N)*self.M/self.N
        
        # set particle temperatures
        self.t=np.ones(self.N)*self.T
        
        
        # calculate gravitational potential energy from partices
        if direct_PE:
            r_grid=squareform(pdist(self.r.value))*self.r.unit

            self.PE_direct=0.
            for i in range(self.N-1):
                self.PE_direct-=self.m[i]*(self.m[i+1:]/r_grid[i,i+1:]).sum()
                
            self.PE_direct=(self.PE_direct*constants.G).to("J")
            
            
        return
    
    def fbm_positions(self):
        
        """
        Subroutine: generates an fbm distribution of particles
        
        """
        
        # set spectral index
        beta=self.n_dim+2.*self.H_fbm
        
        # generate fbm field
        fbm_field=scalar_grf([self.n_grid]*self.n_dim,beta)
        
        # exponentiate and shift to com
        if self.density_scale=="lognormal":
            fbm_field.normalise(sigma=self.sigma_fbm,exponentiate=True)
        elif self.density_scale=="power-law":
            fbm_field.normalise(alpha=self.alpha_fbm,power_law=True)
        else:
            print("Warning: setting density to default lognormal")
            fbm_field.normalise(sigma=self.sigma_fbm,exponentiate=True)
            
        fbm_field.com_shift()
        
        # make probability density function
        fbm_pdf=pdf(fbm_field.signal.real)
        
        # get uniform distribution of points in interval [0,1]
        if self.n_dim==3 and self.glass:
            sph_glass=cube(self.N,glass_file)
        else:
            print("Warning: randomly sampling points.")
            sph_glass=np.random.uniform(size=(self.N,self.n_dim))
        
        # set particle positions
        self.r=fbm_pdf.stretch_lattice(sph_glass)
        
        
        # store density and potential fields (assume grid edge length=1 and mass=1)
        self.rho=fbm_field.signal.real*fbm_field.signal.size/(fbm_field.signal.real.sum())
        self.phi=dft_potential(self.rho)
        
        return
    
    def turb_velocities(self):
        
        """
        Subroutine: calculates a set of turbulent velocities for particles
                
        """
        
        # set spectral index
        beta=self.n_dim+self.alpha_turb-1
        
        # generate turbulent field
        turb_field=vector_grf([self.n_grid]*self.n_dim,beta,field_type=self.turb_type)
        
        # apply velocities to particles
        self.v=np.zeros(self.r.shape)
        for i in range(self.n_dim):
            self.v[:,i]=interpn([np.linspace(0,1,self.n_grid)]*self.n_dim,turb_field.signal.real[...,i],self.r)
            
        # normalise velocities
        self.v-=self.v.mean(axis=0,keepdims=True)
        self.v*=np.sqrt(self.n_dim)*self.sigma_v.value/self.v.std()
        self.v*=self.sigma_v.unit
        
        return
    
    
    def save_to_file(self,ascii_file=None,pickle_file=None):
        
        
        """
        
        Subroutine: output particle ensemble to file
        
        Arguments
        ---------
        
        ascii_file: string
            name of file containing particle information
            
        pickle_file: string
            file name for pickled class
        
        """
        
        # output particle data to file
        if ascii_file:
            
            # make header line
            header="".join(["x"+str(i)+" " for i in range(self.n_dim)])
            header+="".join(["v"+str(i)+" " for i in range(self.n_dim)])
            header+=" M T"
            
            # make units line
            units="".join([str(self.r.unit).replace(" ","")+" " for i in range(self.n_dim)])
            units+="".join([str(self.v.unit).replace(" ","")+" " for i in range(self.n_dim)])
            units+=str(self.m.unit).replace(" ","")+" "+str(self.t.unit).replace(" ","")
            
            # make data columns
            data=np.zeros((self.N,self.n_dim*2+2))
            data[:,:self.n_dim]=self.r
            data[:,self.n_dim:self.n_dim*2]=self.v
            data[:,self.n_dim*2]=self.m
            data[:,self.n_dim*2+1]=self.t
            
            # save file
            np.savetxt(ascii_file,data,header=header+"\n"+units)
            
        
        # pickle class
        if pickle_file:
            file_object=open(pickle_file,"wb")
            pickle.dump(self,file_object)
            file_object.close()
        
        return