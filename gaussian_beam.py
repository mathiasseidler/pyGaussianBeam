from __future__ import division
import numpy as np


'''
complex beam parameter: q = z + i*z0; 1/q = 1/R(z) - i*wl/(pi*n*w(z)**2)
'''



def ABCD_thin_lens(f):
    return np.array([[1,0],
                    [-1/f,1]])
    
def ABCD_propagate(d):
    return np.array([[1,d],
                    [0,1]])

def ABCD_refraction_flat_interface(n1,n2):
    return np.array([[1,0],
                    [0,float(n1)/n2]])
                    
def get_w_from_q(q, wl, n):
    imag = np.imag(1/q[0])
    return np.sqrt((-wl)/(imag*np.pi*n))
    
def get_R_from_q(q):
    return 1/np.real(1/q[0])
    
def new_gauss_beam(q,wl):
    z_R = np.imag(q)
    z = -np.real(q)
    w0 = np.sqrt(z_R*wl/np.pi)
    return Gaussian_Beam(w0,z,wl)



def getbeamparameters(w1, w2, x, wl, option=0, plot=0):
    # from Alessandros python script
    """
    Function to calculate the parameters of the profile of a gaussian beam from the measurements of the waist
    at two positions.
    It obtain the parameters by solving the 4th order polynomial equation for w0 (beam waist).
    All parameters need to be expressed in the same units.

    :param w1: measured waist at position 1
    :param w2: measured waist at position 2
    :param x: distance between the measurements
    :param wl: wavelength
    :param option: 0 - beam waist located between the measurements; 1 - beam waist located outside of the measurements
    :param plot: 0 - do not plot the profile; 1 - plot the profile.
    :return: list of parameters: q0, waist, rayleigh, origin
    """
    # In case the two waists are equal the beam waist is in the center of
    # the two.
    if w2 == w1:
        z = x/2.
    else:
        # define these to clean up the notation
        delta = w2**2-w1**2
        lp = wl/pi
        # define the coefficients of z in the quadratic equation in standard form
        a = delta+lp**2*4*x**2/delta
        b = (lp**2*4*x**3)/delta-(2*x*w1**2)
        c = (lp**2*x**4)/delta-(x*w1)**2

        # Solve the quadratic formula
        # This root corresponds to a waist between the measurements
        z1 = (- b-np.sqrt(b**2-4*a*c))/(2*a)

        # This root corresponts to a waist outside the measurements
        z2 = (- b+np.sqrt(b**2-4*a*c))/(2*a)
        if (b**2-4*a*c) < 0:
            z1 = 0
            z2 = 0
            print('No solution exists for this combination of measurements.')

        if option == 1:
            z = z1
        else:
            z = z2

    # Calculate zR
    rayleigh = wl/pi*(2*x*z+x**2)/(w2**2-w1**2)

    # turn zR into some other useful forms
    q0 = 1j*rayleigh
    waist_0 = np.sqrt(wl*rayleigh/pi)

    # decide which side the beam waist is on
    # if (w1 > w2):
    #     origin = z
    # else:
    #     origin = -z
    origin = z
    if option == 1:
        origin = -z
   # print(
   #'Guesses for curve fit \n Beam waist: \t {0:.3f} micro m\nPositioned at \t {1:.2f} mm from first waist measurement'.format(waist_0*1000,
   #                                                                                                       origin))
    if option == 1:
        zrange = np.linspace(-origin*1.05, (x-origin)*1.05, 100)
        #plotbeam(waist_0, wl, zrange)
        #plt.vlines(-origin, 0, w1, color='r')
        #plt.vlines((x-origin), 0, w2, color='r')
    else:
        if w1 > w2:
            origin = z
            zrange = np.linspace(0, origin*1.05, 100)
        else:
            origin = z
            zrange = np.linspace(0, (x+origin)*1.05, 100)
        #plotbeam(waist_0, wl, zrange)
        #plt.vlines(origin, 0, w1, color='r')
        #plt.vlines((origin+x), 0, w2, color='r')
    if plot != 0:
        plt.show()

    return q0, waist_0, rayleigh, origin


def fit_profile(z_list, w_list, wl, plot=0):
    # from Alessandros python script
    """
    :param z_list: list of position for the waist measurements
    :param w_list: list of measured waist. The order should match the order of the measurement positions
    :param wl: wavelength
    :param plot: 0 - no plot, 1 - plot the data and fit
    :return:list of coefficients from the fit
    """

    def profile(z, w0, z0):
        return w0*np.sqrt(1+((z-z0)*wl/pi/w0**2)**2)

    # estimate initial parameters
    mid_point = len(z_list) / 2
    if z_list[mid_point] < max([z_list[0], z_list[-1]]):
        flag = 0
    else:
        flag = 1
    q0, waist_0, rayleigh, origin = getbeamparameters(w_list[0], w_list[-1], np.abs(z_list[-1] - z_list[0]), wl, flag)
    popt, pcov = curve_fit(profile, z_list, w_list, [waist_0, origin])
    if plot != 0:
        pass
    return popt



class Gaussian_Beam:
    '''
    This class represents a gaussian beam
    '''
    def __init__(self, w0=0.3e-6, z0=0, wl=810e-9, n=1):
        self.n = n    # refractive index
        self.wl = wl  # wavelength
        self.z0 = z0  # postion of w0, NOT the rayleigh range
        self.set_w0(w0) # sets w0 and z_R rayleigh range
        
    def set_w0(self,w0):
        self.w0 = w0
        self.z_R = self.n*np.pi*w0**2/self.wl    
    
    def get_R(self,z):
        '''
        param z: postion
        return: Curvature at position z
        '''
        if z == self.z0:
            return np.inf
        return (z-self.z0)*(1+(self.z_R/(z-self.z0))**2)
        
    def get_waist(self,z):
        '''
        param z: position
        return: Beam waist at position z
        '''
        return self.w0*np.sqrt(1+((z-self.z0)/self.z_R)**2)
        
    def get_q(self,z,n):
        '''
        --
        param z: postion
        param n: refractive index of material
        return: the complex beam parameter at postion z
        --
        '''
        q_inv = np.complex(1/self.get_R(z), -self.wl/(np.pi*n*(self.get_waist(z))**2))
        return 1/q_inv
    
    def get_q_vector(self,z,n):
        '''
        returns the complex beam vector at position z
        '''
        return np.array([self.get_q(z,n), 1])
     
    
        
        
