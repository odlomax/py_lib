import numpy as np
from maths.array.integrate import cum_dist_func
from maths.array.interpolate import grid_interp

class pdf:
    
    """
    
    Class to buld probability density function
    
    """
    
    def __init__(self,f_x,axes=None):
        
        """
        Subroutine: initialises pdf class
        
        Arguments
        ---------
        f_x[...]: float
            ndarray of pdf values
        
        axes[:]: float
            list/tuple of axes values
        
        """
        
        self.f_x=f_x
        
        # make tuple of axes
        if axes==None:
            axes=[]
            for i in self.f_x.shape:
                axes.append(np.linspace(0.,1.,i))
        self.axes=tuple(axes)
        
        # set marginalised distribution
        f_x_marg=self.f_x

        # integrate out axes > 0        
        if len(self.axes)>1:
            for i in range(len(self.axes)-1,0,-1):
                f_x_marg=np.trapz(f_x_marg,self.axes[i],i)
        
        
        self.f_cum=cum_dist_func(self.axes[0],f_x_marg)
        
        # normalise field
        self.f_x/=self.f_cum[-1]
        self.f_cum/=self.f_cum[-1]
        
    
    def stretch_value(self,x):
        
        """
        
        Function: stretces x value to new position based on pdf
        
        Arguements
        ----------
        x[len(self.f_x.shape)]: float
            x value between 0 and 1
            
        Result
        ------
        
        x_new[shape(x)]: float
            stretched value of x
        
        """
    
        # set x_new array
        x_new=[0.]*len(x)
    
        # get first element
        x_new[0]=np.interp(x[0],self.f_cum,self.axes[0])
        
        # get additional elements
        if len(x)>1:
            
            # get slice from self.f_x
            slice_f_x=grid_interp(x_new[0],self.axes[0],self.f_x,0)
            
            # define new pdf class from slice
            slice_pdf=pdf(slice_f_x,self.axes[1:])
            
            # get values
            x_new[1:]=slice_pdf.stretch_value(x[1:])
            
        return x_new
            
            
    def random(self,n=1):
            
        """
        Function: returns array of random variates from pdf
        
        Arguments
        ---------
        n: int
            number of random variates
            
        Result
        ------
        
        r_num[n,len(self.f_x.shape)]: float
            array of random variates
            
        """
        
        # set r_num array
        r_num=np.random.random((n,len(self.f_x.shape)))
        
        for i in range(n):
            r_num[i,:]=self.stretch_value(r_num[i,:])
        
        return r_num
    