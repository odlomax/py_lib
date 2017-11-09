import numpy as np

class scalar_grf:
    
    """
    Data
    ----
    
    spectrum[...]: float
        spectrum of gaussian random field
    
    signal[...]: float
        signal of gaussian random field
    
    """
    
    def __init__(self,n_shape,alpha,real_field):
        
        """
        Subroutine: wrapper for dimensional init routines
        
        Arguments
        ---------
        
        n_shape(,) int tuple
            list of dimensions
            
        alpha: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
        
        """
        
        
        if len(n_shape)==1:
            self.init_1d(n_shape[0],alpha,real_field)
        elif len(n_shape)==2:
            self.init_2d(n_shape[0],n_shape[1],alpha,real_field)
        elif len(n_shape)==3:
            self.init_3d(n_shape[0],n_shape[1],n_shape[2],alpha,real_field)
        else:
            print("Cannot make field with rank",len(n_shape))
            
            
    def init_1d(self,n_x,alpha,real_field):
        
        """
        Subroutine: generate a 1D random gaussian field with a power spectrum
        
        Arguments
        ---------
        
        n_dim: int
            number of dimensions
            
        n_x: int
            edge-length of field
            
        alpha: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
        
        """
        
        # create array of k values from [-n_x/2:n_x/2]
        if np.mod(n_x,2)==0:
            k=np.arange(-n_x//2,n_x//2+1,dtype=np.float)
        else:
            k=np.arange(-n_x//2+1,n_x//2+1,dtype=np.float)
            
        # create amplitude array a(k)
        a_k=np.where(np.abs(k)>0,np.abs(k)**(-0.5*alpha),0.)
        
        # create phase array phi(k)
        phi_k=np.random.random(k.shape)*2.*np.pi
        
        # make spetrum Hermitian if signal is real
        if real_field:
            # make array of phi(-k)
            phi_k_minus=phi_k[::-1]            
            phi_k=phi_k-phi_k_minus
        
        # create complex spectrum
        self.spectrum=np.zeros(k.shape,dtype=np.complex)
        self.spectrum.real=a_k*np.cos(phi_k)
        self.spectrum.imag=a_k*np.sin(phi_k)
        
        # resize and shift array (if size is even, n//2 frequency is superposition of n//2 and -n//2)
        if np.mod(n_x,2)==0:
            self.spectrum[0]+=self.spectrum[-1]
            self.spectrum=self.spectrum[:n_x]
            self.spectrum=np.roll(self.spectrum,n_x//2)
        else:
            self.spectrum=np.roll(self.spectrum,n_x//2+1)
        
        # perform FFT
        self.signal=np.fft.ifft(self.spectrum)
        
        
    def init_2d(self,n_x,n_y,alpha,real_field):
        
        """
        Subroutine: generate a 2D random gaussian field with a power spectrum
        
        Arguments
        ---------
        
        n_dim: int
            number of dimensions
            
        n_x,n_y: int
            edge-length of field
            
        alpha: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
        
        """
        
        # create array of k_x values from [-n_x/2:n_x/2]
        if np.mod(n_x,2)==0:
            k_x=np.arange(-n_x//2,n_x//2+1,dtype=np.float)
        else:
            k_x=np.arange(-n_x//2+1,n_x//2+1,dtype=np.float)
            
        # create array of k_y values from [-n_y/2:n_y/2]
        if np.mod(n_y,2)==0:
            k_y=np.arange(-n_y//2,n_y//2+1,dtype=np.float)
        else:
            k_y=np.arange(-n_y//2+1,n_y//2+1,dtype=np.float)
            
        # convert k_x and k_y to grid format
        k_x,k_y=np.meshgrid(k_x,k_y,indexing='ij')
        k=np.sqrt(k_x**2+k_y**2)

        # create amplitude array a(k)
        a_k=np.where(k>0,k**(-0.5*(alpha+1.)),0.)
        
        # create phase array phi(k)
        phi_k=np.random.random(k.shape)*2.*np.pi
        
        # make spetrum Hermitian if signal is real
        if real_field:
            # make array of phi(-k)
            phi_k_minus=phi_k[::-1,::-1]            
            phi_k=phi_k-phi_k_minus
        
        # create complex spectrum
        self.spectrum=np.zeros(k.shape,dtype=np.complex)
        self.spectrum.real=a_k*np.cos(phi_k)
        self.spectrum.imag=a_k*np.sin(phi_k)
        
        # resize and shift array (if size is even, n//2 frequency is superposition of n//2 and -n//2)
        if np.mod(n_x,2)==0:
            
            self.spectrum[0,:]+=self.spectrum[-1,:]
            self.spectrum=self.spectrum[:n_x,:]
            
            self.spectrum=np.roll(self.spectrum,n_x//2,0)
        else:
            self.spectrum=np.roll(self.spectrum,n_x//2+1,0)
            
        if np.mod(n_y,2)==0:
            
            self.spectrum[:,0]+=self.spectrum[:,-1]
            self.spectrum=self.spectrum[:,:n_y]
            
            self.spectrum=np.roll(self.spectrum,n_y//2,1)
        else:
            self.spectrum=np.roll(self.spectrum,n_y//2+1,1)
            
        # perform FFT
        self.signal=np.fft.ifftn(self.spectrum)
        
    def init_3d(self,n_x,n_y,n_z,alpha,real_field):
        
        """
        Subroutine: generate a 3D random gaussian field with a power spectrum
        
        Arguments
        ---------
        
        n_dim: int
            number of dimensions
            
        n_x,n_y,n_z: int
            edge-length of field
            
        alpha: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
        
        """
        
        # create array of k_x values from [-n_x/2:n_x/2]
        if np.mod(n_x,2)==0:
            k_x=np.arange(-n_x//2,n_x//2+1,dtype=np.float)
        else:
            k_x=np.arange(-n_x//2+1,n_x//2+1,dtype=np.float)
            
        # create array of k_y values from [-n_y/2:n_y/2]
        if np.mod(n_y,2)==0:
            k_y=np.arange(-n_y//2,n_y//2+1,dtype=np.float)
        else:
            k_y=np.arange(-n_y//2+1,n_y//2+1,dtype=np.float)
            
        # create array of k_z values from [-n_z/2:n_z/2]
        if np.mod(n_z,2)==0:
            k_z=np.arange(-n_z//2,n_z//2+1,dtype=np.float)
        else:
            k_z=np.arange(-n_z//2+1,n_z//2+1,dtype=np.float)
            
        # convert k_x and k_y to grid format
        k_x,k_y,k_z=np.meshgrid(k_x,k_y,k_z,indexing='ij')
        k=np.sqrt(k_x**2+k_y**2+k_z**2)
            
        # create amplitude array a(k)
        a_k=np.where(k>0,k**(-0.5*(alpha+2.)),0.)
        
        # create phase array phi(k)
        phi_k=np.random.random(k.shape)*2.*np.pi
        
        # make spetrum Hermitian if signal is real
        if real_field:
            # make array of phi(-k)
            phi_k_minus=phi_k[::-1,::-1,::-1]
            phi_k=phi_k-phi_k_minus
        
        # create complex spectrum
        self.spectrum=np.zeros(k.shape,dtype=np.complex)
        self.spectrum.real=a_k*np.cos(phi_k)
        self.spectrum.imag=a_k*np.sin(phi_k)
        
        # resize and shift array (if size is even, n//2 frequency is superposition of n//2 and -n//2)
        if np.mod(n_x,2)==0:
            self.spectrum[0,:,:]+=self.spectrum[-1,:,:]
            self.spectrum=self.spectrum[:n_x,:,:]
            self.spectrum=np.roll(self.spectrum,n_x//2,0)
        else:
            self.spectrum=np.roll(self.spectrum,n_x//2+1,0)
        
        if np.mod(n_y,2)==0:
            self.spectrum[:,0,:]+=self.spectrum[:,-1,:]
            self.spectrum=self.spectrum[:,:n_y,:]
            self.spectrum=np.roll(self.spectrum,n_y//2,1)
        else:
            self.spectrum=np.roll(self.spectrum,n_y//2+1,1)
            
        if np.mod(n_z,2)==0:
            self.spectrum[:,:,0]+=self.spectrum[:,:,-1]
            self.spectrum=self.spectrum[:,:,:n_z]
            self.spectrum=np.roll(self.spectrum,n_z//2,2)
        else:
            self.spectrum=np.roll(self.spectrum,n_z//2+1,2)
            
        # perform FFT
        self.signal=np.fft.ifftn(self.spectrum)
        
    def normalise(self,sigma=1.,exponentiate=False,exp_base=np.e):
        
        """
        
        Subroutine: normalise signal to a standard deviation
        
        Arguments
        ---------
        
        sigma: float
            standard deviation
            
        exponentiate: boolean
            exponentiate signal?
            
        exp_base: float
            base of exponential
        
        """
        
        # Normalise signal
        self.signal*=sigma/np.std(np.abs(self.signal))
        
        if exponentiate:
            self.signal=np.exp(self.signal*np.log(exp_base))
        

    def com_shift(self):
        
        """
        
        Subroutine: shifts signal to periodic centre of mass (signal should be positive for all x)
        
        """
        
        
        # set theta arrays
        theta_i=[]
        for i in range(len(self.signal.shape)):
            theta_i.append(np.arange(self.signal.shape[i],dtype=np.double)*2.*np.pi/(self.signal.shape[i]))
            
        # convert to grid format
        theta=np.meshgrid(*theta_i,indexing='ij')
        
        # loop over axes
        for i in range(len(theta)):
            
            # calculate shift
            xi=np.cos(theta[i])*np.abs(self.signal.real)
            zeta=np.sin(theta[i])*np.abs(self.signal.real)
            theta_bar=np.arctan2(-zeta.mean(),-xi.mean())+np.pi
            shift=np.int((self.signal.shape[i])*0.5*theta_bar/np.pi)
            
            # shift array
            self.signal=np.roll(self.signal,self.signal.shape[i]//2-shift,i)
            