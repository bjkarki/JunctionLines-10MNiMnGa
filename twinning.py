## required packages
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt


## NOTE:
  # r is the reference frame (R-frame)
  # m is the crystal frame (m-frame)

#################################################################################################
#################################################################################################
#################################################################################################

def angleTBpair(rPm, k1_m,k2_m):
    """
    Calulcates angle between two given planes.
    
    Parameters
    ----------
    rPm: numpy.ndarray(3,3), transformation matrix (crystal frame ↔ reference frame)
    k1_m: numpy.ndarray(1,3), plane 1 in crystal frame
    k2_m: numpy.ndarray(1,3), plane 2 in crystal frame
    
    Returns
    -------
    float, angles in degrees
    """
    
    ## plane normal in reference frame
    k1_r = k1_m @ inv(rPm)
    k2_r = k2_m @ inv(rPm)
    
    ## slope of the twin boundary line in [x,y]ᵣ
    m1 = -k1_r[0]/k1_r[1] # slope of TB k1_m
    m2 = -k2_r[0]/k2_r[1] # slope of TB k2_m
    
    ## phi is the angle between two lines with slope m1 and m2
    phi = np.arctan(np.abs((m1-m2)/(1+(m1*m2))))
    
    ## return phi in degrees
    return np.degrees(phi)

#################################################################################################
#################################################################################################
#################################################################################################

## latticevectors_ab plots lattice vectors a and b in the plane (R-frame)
def latticevectors_ab(rPm):
    """
    Plots lattice vectors aₘ and bₘ
    Assumptions: cₘ is orthogonal to both aₘ and bₘ
    
    Parameters
    ----------
    rPm: numpy.ndarray(3,3), transformation matrix (crystal frame ↔ reference frame)
    
    Returns
    -------
    plot, returns quiver plot of aₘ and bₘ
    """
    
    rPm = np.asarray(rPm)

    ## Plot lattice vectors a and b in the plane
    fig, ax = plt.subplots() #define figure and axes

    ## Quiver plot paramaters
    x = [0,0]
    y = [0,0]
    ux = rPm[0][0:2]
    uy = rPm[1][0:2]
    
    ## quiver plot
    ax.quiver(x,y,ux,uy,scale=2.5)

    ## Axis options
    ax.set_aspect('equal') #aspect ratio 1:1

    ax.set_xlabel(r'$\mathbf{a}_R\ \rightarrow$', fontsize=16) #xlabel
    ax.set_ylabel(r'$\mathbf{b}_R\ \rightarrow$',fontsize=16) #ylabel

    ax.spines['right'].set_visible(False) #eliminate spines
    ax.spines['top'].set_visible(False) #eliminate spines
    ax.spines['bottom'].set_visible(False) #eliminate spines
    ax.spines['left'].set_visible(False) #eliminate spines
    
    ## return fig and axis
    return fig, ax

#################################################################################################
#################################################################################################
#################################################################################################

def rotationmatrix(r,theta):
    """
    Returns rotation matrix in cartesian coordinates
    
    Parameters
    ----------
    r: numpy.ndarray(3,1), axis of rotation in reference frame
    theta: float, rotation angle in radians
    
    Returns
    -------
    numpy.ndarray(3,3), retruns rotation matrix: theta rotation about r
    """
    
    ## convert to unit vector
    r = np.squeeze(np.asarray(r))
    r = r / np.sqrt(r @ r.T)
    
    ux, uy, uz = r; #obtain each element
    
    ## each row of matrix separately
    row1 = [np.cos(theta)+(ux**2*(1-np.cos(theta))),
            (ux*uy*(1-np.cos(theta)))-(uz*np.sin(theta)),
            (ux*uz*(1-np.cos(theta)))+(uy*np.sin(theta))
           ] #row1 of rotation matrix
    row2 = [(uy*ux*(1-np.cos(theta)))+(uz*np.sin(theta)),
            np.cos(theta)+(uy**2*(1-np.cos(theta))),
            (uy*uz*(1-np.cos(theta)))-(ux*np.sin(theta))
           ] #row2 of rotation matrix
    row3 = [(uz*ux*(1-np.cos(theta)))-(uy*np.sin(theta)),
            (uz*uy*(1-np.cos(theta)))+(ux*np.sin(theta)),
            np.cos(theta) + (uz**2*(1-np.cos(theta)))
           ] #row3 of rotation matrix
    
    ## the rotation amtrix
    R = np.array([row1,row2,row3]);
    del row1, row2, row3
    
    ## return the rotation matrix
    return applythresh(R)

#################################################################################################
#################################################################################################
#################################################################################################

## OR_NCtwins calculates the OR in the R-frame
def OR_NCtwins(K1,eta1,rPm,a,normal_pos=True):
    """
    Returns orientation relationship (OR) of non-conventional (NC) twins in reference frame
    
    Parameters
    ----------
    K1: numpy.ndarray(1,3), twin boundary in crystal frame
    eta1: numpy.ndarray(3,1), shear direction in crystal frame
    rPm: numpy.ndarray(3,3), transformation matrix (crystal frame ↔ reference frame)
    a: float, lattice parameter of aₘ
    normal_pos: bool, plane normal points to the twin (default true) 
    
    Returns
    -------
    numpy.ndarray(3,3), returns OR of NC twins in reference frame
    """
    
    # convert to array
    K1 = np.squeeze(np.asarray(K1))
    eta1 = np.squeeze(np.asarray(eta1))
    rPm = np.asarray(rPm)
    
    # convert K1 and eta1 if plane normal is not positive
    if normal_pos == False:
        eta1 = -eta1
        K1 = -K1
    
    ## metric tensor
    X = rPm * a #revert the scale used in rPm
    G = X.T @ X
    
    ## correspondence matrix:
    C = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    ## calculate shear
    s = np.sqrt(np.trace(C.T @ G @ C @ inv(G))-3)
    del X,G
    
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
    rL = rPm @ C @ inv(rPm) @ inv(rS)
    return rL

#################################################################################################
#################################################################################################
#################################################################################################

def applythresh(P,t=10**-9):
    """
    Applies custom threshold of 9-digits precision for floating point output (list,numpy.ndarray)
    
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

#################################################################################################
#################################################################################################
#################################################################################################