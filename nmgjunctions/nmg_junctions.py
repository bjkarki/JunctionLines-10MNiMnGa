##########################################################################################################

## required packages
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

##########################################################################################################

class NMG_junction:
    
    ######################################################################################################
    
    def __init__(
        self,
        a = 0.5972,
        b = 0.5944,
        c = 0.5584,
        gamma = np.deg2rad(90.37)
    ):
        """
        Initialize the NMG_junction class with lattice parameters
        
        Parameters
        ----------
        a: float, lattice parameter a in nm
        b: float, lattice parameter b in nm
        c: float, lattice parameter c in nm
        gamma: float, lattice parameter gamma in radians
        """
        
        self.lattice_parameter = [a,b,c,gamma]
        self.G = np.array([[a**2,a*b*np.cos(gamma),0],[a*b*np.cos(gamma),b**2,0],[0,0,c**2]])
    
    ######################################################################################################
    
    def load_irr_elem(self):
        """
        Loads irrationa elements of type I, type II and non-conventional twins
        """
        a,b,c,gamma = self.lattice_parameter
        self.q3 = (2*a*b*np.cos(gamma) - np.sqrt(a**4 + b**4 + 2*a**2*b**2*np.cos(2*gamma)))/(a**2 - b**2)
        self.q4 = (2*a*b*np.cos(gamma) + np.sqrt(a**4 + b**4 + 2*a**2*b**2*np.cos(2*gamma)))/(a**2 - b**2)
    
    ######################################################################################################
    
    def load_twin_mode(self):
        """
        Loads compound and non-conventional twinning mode
        """
        
        q3 = self.q3
        q4 = self.q4
        
        self.compound_twin = {
            'ab_twin': {
                'k1': np.array([1,1,0]),
                'k2': np.array([-1,1,0]),
                'gamma1:': np.array([-1,1,0]),
                'gamma2': np.array([1,1,0])
            },
            'modulation_twin': {
                'k1': np.array([0,1,0]),
                'k2': np.array([1,0,0]),
                'gamma1:': np.array([1,0,0]),
                'gamma2': np.array([0,1,0])
            }
        }
        
        self.nc_twin = {
            'k1': np.array([1,q3,0]),
            'k2': np.array([-1,-q4,0]),
            'gamma1': np.array([q3,-1,0]),
            'gamma2': np.array([q4,-1,0])
        }
    
    ######################################################################################################
    
    def OR_ref_frame(self,K1,eta1,C,rPm,normal_pos=True):
        """
        Returns orientation relationship (OR) of twins in reference frame
        
        Parameters
        ----------
        K1: numpy.ndarray(1,3), twin boundary in crystal frame
        eta1: numpy.ndarray(3,1), shear direction in crystal frame
        C: numpy.ndarray(3,1), Correspondence Matrix
        rPm: numpy.ndarray(3,3), transformation matrix (crystal frame ↔ reference frame)
        normal_pos: bool, plane normal points to the twin (default true) 

        Returns
        -------
        rL: numpy.ndarray(3,3), returns OR of NC twins in reference frame
        """
        
        G = self.G
        
        ## convert K1 and eta1 if plane normal is not positive
        #Warning!: Doesn't work for NC twins
        #For NC twins, the correspondence has to be switched
        if normal_pos == False:
            eta1 = -eta1
            K1 = -K1
        
        #shear value
        s = np.sqrt(np.trace(C.T @ G @ C @ inv(G))-3)
        
        ## obtain plane normal (m) and shear direction (l) in R-frame:
        rm = K1 @ inv(rPm)
        rm = rm / np.sqrt(rm @ rm.T) #unit vector

        rl = rPm @ eta1
        rl = rl / np.sqrt(rl @ rl.T)
        
        ## shear matrix: Using einstein convention
        rS = np.zeros([3,3]) #initialize

        #1st row
        rS[0,0] = 1 + s * rl[0] * rm[0] #i=0,j=0
        rS[0,1] = 0 + s * rl[0] * rm[1] #i=0,j=1
        rS[0,2] = 0 + s * rl[0] * rm[2] #i=0,j=2
        #2nd row
        rS[1,0] = 0 + s * rl[1] * rm[0] #i=1,j=0
        rS[1,1] = 1 + s * rl[1] * rm[1] #i=1,j=1
        rS[1,2] = 0 + s * rl[1] * rm[2] #i=1,j=2
        #3rd row
        rS[2,0] = 0 + s * rl[2] * rm[0] #i=0,j=0
        rS[2,1] = 0 + s * rl[2] * rm[1] #i=0,j=1
        rS[2,2] = 1 + s * rl[2] * rm[2] #i=0,j=2

        ## Orientation relationship in R-frame
        rL = self.applythresh(rPm @ C @ inv(rS @ rPm))
        
        return rL
    
    ######################################################################################################
    
    def applythresh(self,P,t=10**-9):
        """
        Applies custom threshold of given precision for floating point output (list,numpy.ndarray)

        Parameters
        ----------
        P: float or list or numpy.ndarray(1,3), matrix where custrom precision is applied
        t: float, the custom threshold (default 10⁻⁹)

        Returns
        -------
        returns P with custom precision
        """

        ## find indices that are below threshold
        ind = np.absolute(P) < t

        ##set the value of indices to 0
        P[ind] = 0

        #return array
        return P
    
    ######################################################################################################