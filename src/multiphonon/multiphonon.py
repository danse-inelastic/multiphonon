#!/usr/bin/python
"""
Module containing functions to calculate incoherent scattering, including
multiphonon and approximated multiple scattering, from a DOS.
  
An instance of expSqe `sqe` is also required, but the actual scattering
data is never used (q, de, M, beta, zeroInd are used, however).

Names for functions are taken from appendix A of Sears, "Phonon density of 
states in vanadium".  Currently found at 
.../multiphonon/doc/Sears--Phonon_density_of_states_in_vanadium.pdf
"""

import numpy
nar = numpy.add.reduce
from scipy.misc import factorial

from constants       import A2m,h_b,J2meV
from doubleDos       import doubleDos
from geometry import H1Mat

def sqeSears(sqe,dos,N):#,C_ms):
    """ Takes an expSqe, a densityOfStates, and a maximum term in the multiphonon 
    expansion and returns a list of arrays S_n(Q,E) for 0 < n <= N.
    
    It used to also take a multiplier called C_ms taking multiphonon scattering to multiphonon 
    plus multiple scattering.
    """
    SqeSears = []
    ANE = AthroughN(sqe,dos,N)
    SNQ = SthroughN(sqe,dos,N)
    debug.log( 'sqe.shape=%s' % ( sqe.shape, ) )
    debug.log( "dos.gz.shape=%s" % (dos.gz.shape, ) )
    for i in range(len(ANE)):
        debug.log( "ANE[i].shape=%s" % (ANE[i].shape,) )
        debug.log( "sqe.mask.shape=%s" % (sqe.mask.shape,) )
        SqeSears.append( numpy.outer(SNQ[i], ANE[i]) * sqe.mask ) #chopped along with mask;
    SqeSears = numpy.array(SqeSears)
    return SqeSears/nar(nar(nar(SqeSears)))# is this normalization factor correct?
##    SQE[1:] *= C_ms
#    return Sqe/nar(nar(nar(Sqe)))

def sqeSearsGeometryEffects(sqe, sqeSears, Ei, crossSection=None, density=None, thickness=None):
    """ Takes an expSqe, a sears sqe, some geometrical terms
     and returns a list of arrays S_n(Q,E) for 0 < n <= N with geometry effects.
    """
    SqeSearsGeometryEffects = []
    for nPhononSqe in sqeSears:
        SqeSearsGeometryEffects.append(nPhononSqe*H1Mat(sqe.q, sqe.e, Ei, crossSection, density, thickness))
    return SqeSearsGeometryEffects

def AthroughN(sqe,dos,N):
    """ Takes an expSqe, a densityOfStates and a maximum term in the 
    multiphonon expansion and returns an array of A_n(E) for all 0 < n <= N 
    """
    ANE = []
    ANE.append(getA1E(sqe,dos))
    for i in range(2,N+1): # I am very surprised to see this; just 5 phonon process is taken into account?
        ANE.append(getANE(sqe,ANE[0],ANE[-1]))
    return numpy.array(ANE)

def SthroughN(sqe,dos,N):
    """ Takes an expSqe, a densityOfStates and a maximum term in the 
    multiphonon expansion and returns an array of S_n(Q) for all 0 < n <= N 
    """
    SNQ = []
    DW2 = debyeWallerExp(sqe,dos)
    for i in range(1,N+1):
        SNQ.append(getSNQ(DW2,i))
    return numpy.array(SNQ)

def getA1E( sqe,dos ):
    """ Takes an expSqe and a densityOfStates and returns the shape of the 
    1-phonon incoherent scattering, A_1(E)
    """
    g0 =  gamma0(sqe,dos)
    dDos = doubleDos( dos )
    debug.log( 'g0=%s' % g0 )
    debug.log( 'dDos.shape=%s' % (dDos.gz.shape,) )
    res = dDos.gz / g0 / ( dDos.e * ( 1 - numpy.exp(-dDos.e*sqe.beta ) ) )# information of SQE is not really used except Temp;
    z = sqe.zeroInd
    res[z] = 2.0*( res[z+1] + ( res[z+1] - res[z+2] ) ) # Not sure why thi sis introduced;
    debug.log( str(res.shape) )
    return res/nar(res)/dos.de
 
def getANE(sqe, A1E, ANMinus1E): #I may want to Nikolay for exact following
    """ Takes an expSqe the 1-phonon incoherent scattering as calculated by
    getA1E(...) and the N-1-phonon incoherent scattering as calculated by 
    this function and  returns the N-phonon incoherent scattering, A_N(E)
    """
    Y = numpy.zeros( 4*len(ANMinus1E),'d' )
    Y[len(A1E):2*len(A1E)] = ANMinus1E
    y = numpy.zeros( 3*len(A1E),'d' )
    y = y.tolist() + A1E.tolist()
    y.reverse()
    y = numpy.array(y)
    M = convMatrix(y)
    res = numpy.inner(M,Y)
    res /= nar(res)*sqe.de
    return res[len(A1E)/2+1:len(A1E)+len(A1E)/2+1]     

def getSNQ(DW2,N):
    """ Takes the exponent for the Debye Waller factor `DW2` = 2W, and an
    integer N indicating a term in the phonon expansion and returns the 
    intensity of the N-phonon incoherent scattering S_N(Q)
    """
    return DW2**N * numpy.exp(-1*DW2) / float(factorial(N))

def debyeWallerExp(sqe,dos):
    """ Takes an expSqe and a densityOfStates and returns 2W, the exponent
    of the Debye Waller factor. 
    """
    g0 =  gamma0(sqe,dos)
    Er = recoilE(sqe)
    return Er*g0

def recoilE(sqe):
    """ Takes an expSqe and returns the recoil energy E_r(Q) """
    return J2meV * ( h_b*sqe.q/A2m )**2.0 / 2.0 /sqe.M

def gamma0(sqe,dos):
    """ Takes an expSqe and a densityOfStates and returns thermal factor 
    gamma_0 
    """
    res = ( numpy.cosh(sqe.beta*dos.e/2.0)/ \
          numpy.sinh(sqe.beta*dos.e/2.0)  )*(dos.gz/dos.e)*dos.de
    res[0] = res[1] + ( res[1] - res[2] ) # Define unidentified value, i.e. initial value.
    return nar(res) # this is kind of integration over energy;

def convMatrix(y):
    """  Returns matrix M, whose rows are filled with shifted copies of vector y"""
    M = numpy.zeros( ( len(y),len(y) ) , 'd' )
    for i in range(len(y)):
        M[i,i:] = y[:len(y)-i]
    return M

try:
    import journal
    debug = journal.debug( "multiphonon.multiphonon" )
except ImportError:
    class _Debug:
        def log(self, msg): return #print msg
        line = log
    debug = _Debug()
#debug.activate()
