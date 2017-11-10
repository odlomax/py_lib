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
        Subroutine: generate a nd random gaussian field with a power spectrum
        
        Arguments
        ---------
        
        n_shape(,): tuple, int
            shape of field
            
        alpha: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
        
        """
        
        # make k coordinate arrays
        k_list=[]
        for i in n_shape:
            k_list.append(np.arange(-i//2+np.mod(i,2),i//2+1))

        # generate array of k values
        if len(n_shape)>1:
            k=np.zeros(np.array(n_shape)+np.mod(np.array(n_shape)+1,2))
            for i in np.meshgrid(*k_list,indexing="ij"):
                k+=i**2
        else:
            k=k_list[0]**2
        k=np.sqrt(k)
            
        # create amplitude array a(k)
        a_k=np.where(k>0,k**(-0.5*(alpha+len(n_shape)-1)),0.)
        a_k/=np.sqrt((a_k**2).sum())
        
        # create phase array phi(k)
        phi_k=np.random.random(k.shape)*2.*np.pi
        
        # make spetrum Hermitian if signal is real
        if real_field:
            # make array of phi(-k)
            phi_k_minus=phi_k[[slice(None,None,-1)]*len(n_shape)]
            phi_k=phi_k-phi_k_minus
        
        # create complex spectrum
        self.spectrum=np.zeros(k.shape,dtype=np.complex)
        self.spectrum.real=a_k*np.cos(phi_k)
        self.spectrum.imag=a_k*np.sin(phi_k)
        
        # resize array (if size is even, n//2 frequency is superposition of n//2 and -n//2)
        for i in range(len(n_shape)):
            if np.mod(n_shape[i],2)==0:
                i_plus=[slice(None)]*len(n_shape)
                i_plus[i]=-1
                i_minus=[slice(None)]*len(n_shape)
                i_minus[i]=0
                self.spectrum[i_minus]+=self.spectrum[i_plus]
                indices=[slice(None)]*len(n_shape)
                indices[i]=slice(None,-1)
                self.spectrum=self.spectrum[indices]
                
        # shift array
        offset=(np.array(n_shape)+np.mod(np.array(n_shape),2))//2
        axis=np.arange(len(n_shape))
        self.spectrum=np.roll(self.spectrum,offset,axis)
            
        # perform FFT
        self.signal=self.spectrum.size*np.fft.ifftn(self.spectrum)
        
        
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
            
        # update spectrum
        self.spectrum=np.fft.fftn(self.signal)
        

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
            
        # update spectrum
        self.spectrum=np.fft.fftn(self.signal)
        
    def gauss_conv(self,sigma):
        
        """
        
        Subroutine: applies Gaussian convolution to signal
        
        Arguments
        ---------
        
        sigma: float
            width of Gaussian (pixels)
            
        """ 
            
        # make list of axis coodinates
        axes=[]
        for i in self.signal.shape:
            axes.append(np.arange(i)-(i-1)/2.)
                        
        # make r squared array
        if len(self.signal.shape)>1:
            r_sqd=np.zeros(self.signal.shape)
            for j in np.meshgrid(*axes,indexing="ij"):
                r_sqd+=j**2
        else:
            r_sqd=axes[0]**2
        
        # generate kernel function
        kernel=np.zeros(self.signal.shape,dtype=np.complex)
        kernel.real=np.exp(-r_sqd/(2.*sigma**2))
        
        # normalise and shift kernel
        kernel.real/=np.sum(kernel.real)
        offset=(np.array(kernel.shape)+np.mod(np.array(kernel.shape),2))//2
        axis=np.arange(len(kernel.shape))
        kernel=np.roll(kernel,offset,axis)
        
        # convolve signal with kernel
        self.spectrum=np.fft.fftn(self.signal)
        self.spectrum*=np.fft.fftn(kernel)
        self.signal=np.fft.ifftn(self.spectrum)
        
        
        
            
        
        
        
        
        
        