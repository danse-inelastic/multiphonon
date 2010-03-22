import numpy as np

from scipy.constants import physical_constants as constants
k_boltzmann = constants['Boltzmann constant in eV/K'][0]*10**3 #eV/K**meV/eV=meV/K
h_planck_Js = constants['Planck constant'][0] #J*s
from scipy.constants import m_n
mass_neutron_kg = m_n
J2meV = constants['joule-electron volt relationship']*10**3

def theta(Ei, E, Q):
    """ give angle of deflection from Q mag, Ei, and E
    """
    Q_meters = Q*10**10
    theta = np.arccos((2*Ei - E - (h_planck_Js*Q_meters)^2/(2*mass_neutron_kg)*J2meV)/
                      (2*np.sqrt(Ei*(Ei - E)))) 
    return theta

def H1(Q, E, Ei, crossSection=None, density=None, thickness=None):
    """the scattering plate thickness factor--see the drawing in the directory"""
    if crossSection and density and thickness:
        Sigma = crossSection*density
        d = thickness
        percentageScatterer = Sigma*d
    else:
        percentageScatterer = 0.1
    return np.exp(-percentageScatterer/np.cos(np.pi/4)) - \
        np.exp(-percentageScatterer/np.abs(np.cos(theta(Ei, E, Q) - np.pi/4)))
        
def H1Mat(Qrange, Erange, Ei, crossSection=None, density=None, thickness=None):
    return np.array([[H1(Q,E,Ei,crossSection, density, thickness) for E in Erange] for Q in Qrange])
        
def B1(theta1, crossSection=None, density=None, thickness=None):
    """a second scattering geometrical quantity--see the drawing in the directory"""
    if crossSection and density and thickness:
        Sigma = crossSection*density
        d = thickness
        percentageScatterer = Sigma*d
    else:
        Sigma = 1
        percentageScatterer = 0.1
    return 0.75/Sigma*(1-np.exp(-2/3*percentageScatterer/np.abs(np.cos(theta1 - np.pi/4))))
                       
def H2(Q, E, theta1, Ei, crossSection=None, density=None, thickness=None):
    return H1(Q, E, Ei, crossSection, density, thickness)*B1(theta1, crossSection, density, thickness)
