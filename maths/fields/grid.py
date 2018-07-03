import numpy as np

def r_grid(shape,delta_r=[]):
    
    """
    
    Function: make a centred grid of distances
    
    Arguments
    ---------
    
    shape[]: int
        shape of r grid
        
    delta_r[]: float
        grid edge-lengths
        
    Result
    ------
    
    r[...]: float
        ndarray of r values
          
    """
    
    # calculated dr
    if len(delta_r)==len(shape):
        dr=np.array(delta_r)/(np.array(shape))
    else:
        dr=np.full(len(shape),1.)
        
    # make lists of r values 
    r_list=[]
    for i in range(len(shape)):
        r_list.append(np.ceil(np.arange(-shape[i]/2,shape[i]/2))*dr[i])
        
    # make grids of r values
    r_grid=np.meshgrid(*r_list,indexing="ij")
    
    # make grid of r magnitudes
    r=np.sqrt((np.stack(r_grid)**2).sum(axis=0))
    
    return r
        