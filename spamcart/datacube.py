import numpy as np
import sys
from maths.array.interpolate import grid_interp
from astropy import constants
from astropy.convolution import convolve_fft

class datacube:
    
    """
    SPAMCART datacube class
        
    Data
    ----
    
    lambda_array[:]: float
        array of wavelength values
    
    x_array[:]: float
        array of x position values
    
    y_array[:]: float
        array of y position values
        
    sigma_array[y_array.shape[0],x_array.shape[0]]: float
        array of column density values
    
    i_array[lambda_array.shape[0],y_array.shape[0],x_array.shape[0]]: float
        array of intensity values
    """
    
    # some standard variables
    lambda_label="I_lambda"
    lambda_unit="micron"
    
    
    def __init__(self,file_name):
        
        """
        Subroutine: Initialises SC_datacube class
        
        Arguments
        ---------
        
        file_name: string
            gives the location of the SPAMCART datacube.dat file
            
        """
        
        # Read wavelength values from header
        lambda_list=[]
        with open(file_name) as file:
            n_head=0
            i_data_start=sys.maxsize
            i_data_end=0
            for file_line in file:
                if len(file_line.strip())>0:
                    n_head+=1
                    if self.lambda_label in file_line:
                        i_data_start=min(n_head-1,i_data_start)
                        i_data_end=max(n_head,i_data_end)
                        string_start=file_line.index("=")+1
                        string_end=file_line.index(self.lambda_unit)
                        lambda_list.append(file_line[string_start:string_end].strip())
                else:
                    break
        # Assign lambda_array
        self.lambda_array=np.array(lambda_list,dtype=np.double)
        
        # Load column data from text file
        data=np.loadtxt(file_name,skiprows=n_head+1,dtype=np.double)

        # Make temporary columns
        temp_x=data[:,0]
        temp_y=data[:,1]
        temp_sigma=data[:,2]
        temp_i=data[:,i_data_start:i_data_end]
        
        # Loop over temp_x to find number of x elements
        for i in range(temp_x.shape[0]):
            if temp_x[i+1]<temp_x[i]:
                # Assign x_array
                self.x_array=temp_x[:i+1]
                break

        # Assign y_array
        self.y_array=temp_y[::self.x_array.shape[0]]

        # Reshape column density column to grid
        self.sigma_array=temp_sigma.reshape((self.y_array.shape[0],self.x_array.shape[0]))

        # Reshape intensity column to grid
        self.i_array=np.zeros((self.lambda_array.shape[0],self.y_array.shape[0],self.x_array.shape[0]),dtype=np.double)
        for i in range(self.i_array.shape[0]):
            self.i_array[i,:,:]=temp_i[:,i].reshape((self.y_array.shape[0],self.x_array.shape[0]))
        
        print()
        print("dimensions of lambda array:",self.lambda_array.shape)
        print("dimensions of x array     :",self.x_array.shape)
        print("dimensions of y array     :",self.y_array.shape)
        print("dimensions of sigma array :",self.sigma_array.shape)
        print("dimensions of I array     :",self.i_array.shape)
        print()
        
    def get_mono_map(self,wavelength):
        
        """
        Function: non convolved monochromatic intensity map [erg s^-1 sr^-1 cm^-2 micron^-1]
        
        Arguments
        ---------
        
        wavelength[:]: float
            wavelength values (microns)
            
        Result
        ------
        
        mono_map[self.y_array.shape[0],self.x_array.shape[0]]:
            monochromatic intensity map
        
        """
        
        # Return mono_map        
        return grid_interp(wavelength,self.lambda_array,self.i_array,0)
    
    def get_spectrum(self):
        
        """
        Function: get spectrum of datacube
        
        Result
        ------
        spectrum[self.lambda_array.shape[0]]
        
        """
        
        # Return spectrum
        return np.trapz(np.trapz(self.i_array,self.x_array,2),self.y_array,1)
        
    def get_band_map(self,wavelength,spectral_response,angle,beam_profile,distance,ref_wavelength):
        
        """
        Function: produces convolved intensity map over a finite band [erg s^-1 sr^-1 cm^-2 micron^-1]
        
        Arguments
        ---------
        
        wavelength[:]: float
            wavelength values (microns)
        spectral_response[wavelength.shape[0]]: float
            spectral response (arbitary units) as a function of wavelength
        angle[:]: float
            angular displacement (arcseconds) from centre of beam
        beam_profile[angle.shape[0]]: float
            beam profile (arbitrary units) as a function of angle
        distance: float
            distance to source (parsecs)
        ref_wavelength: float
            reference wavelength (microns) of beam profile
            
        Result
        ------
        
        band_map[self.y_array.shape[0],self.x_array.shape[0]]:
            wavelength-integrated convolved intensity map
        """
        
        # Initialise intensity map
        intensity_map=np.zeros((wavelength.shape[0],self.y_array.shape[0],self.x_array.shape[0]),dtype=np.double)
        
        # Calculate normaliseation value so the spectral response integrates to unity
        spectral_norm=1./np.trapz(spectral_response,wavelength)
        
        # Interpolate values from self.i_array onto intensity map
        for i in range(intensity_map.shape[0]):
            # Get interpolation values
            intensity_map[i,:,:]=grid_interp(wavelength[i],self.lambda_array,self.i_array,0)*spectral_norm*spectral_response[i]
        
        # Make array of angle to physical size conversion factors
        beam_conv=(np.pi/6.48e+5)*distance*constants.pc.cgs.value*wavelength/ref_wavelength
        
        # Make grids of x and y values
        x_mid=0.5*(self.x_array[-1]+self.x_array[0])
        y_mid=0.5*(self.y_array[-1]+self.y_array[0])
        x_grid,y_grid=np.meshgrid(self.x_array,self.y_array)
        r_grid=np.sqrt((x_grid-x_mid)**2+(y_grid-y_mid)**2)
        
        # Initialise beam map
        beam_map=np.zeros(intensity_map.shape,dtype=np.double)
        for i in range(beam_map.shape[0]):
            # set beam value
            beam_map[i,:,:]=np.interp(r_grid,beam_conv[i]*angle,beam_profile)            
            # normalise beam map
            beam_map[i,:,:]=beam_map[i,:,:]/beam_map[i,:,:].sum()
        
        for i in range(intensity_map.shape[0]):
            intensity_map[i,:,:]=convolve_fft(intensity_map[i,:,:],beam_map[i,:,:],boundary="wrap")
        
        # Integrate spectral response
        return np.trapz(intensity_map,wavelength,axis=0)

                    
        
        
