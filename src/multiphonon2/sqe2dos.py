#!/usr/bin/python
""" Function to take 1-phonon scattering to a phonon DOS """

import numpy as np
#nar = np.add.reduce

from densityOfStates import densityOfStates
from multiphonon     import debyeWallerExp,gamma0
from geometry import H1




def sqe2dos(sqe, Ei):
    """Takes an instance `sqe` of class expSqe and returns `res`, an instance 
    of class densityOfStates, assuming that `sqe` is 1-phonon, single scattering 
    only. Sample shape and orientation are taken into account.  This is equivalent to first
    order multiple scattering correction.
    """ 
    dos = densityOfStates(sqe)
    g0 = gamma0( sqe, dos )
    DW2 = debyeWallerExp(sqe,dos)
    #res = nar( sqe.e*sqe.sqe*g0*(1 - np.exp(-sqe.e*sqe.beta)))[sqe.zeroInd:]
    geRaw = np.zeros(len(sqe.q))
    for enInd in range(len(sqe.e)):
        for qInd in range(len(sqe.q)):
            geRaw[enInd] += sqe.e[enInd]*sqe.sqe[qInd][enInd]*g0*(1 - \
                np.exp(-sqe.e[enInd]*sqe.beta))/(H1(sqe.q[qInd], sqe.e[enInd], Ei)*DW2*np.exp(-DW2)) 
    res = densityOfStates(dos.e, geRaw[sqe.zeroInd:], cutRange=sqe.cutRange)
    return res

