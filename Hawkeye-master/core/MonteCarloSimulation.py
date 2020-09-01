#!/usr/bin/env python
# coding: utf-8

import numpy as np


__all__ = ['MCsimulate']


def IIDsimulate(nIter,dists):
    return [dist.rvs(nIter) if dist else dist for dist in dists]
    # rvs = []
    # for dist in dists:
        # if isinstance(dist,(list,tuple)):
            # uniform_rvs = np.random.rand(nIter)
            # rvs.append([d.ppf(uniform_rvs) for d in dist])
        # else:
            # rvs.append(dist.rvs(nIter))
    # return rvs

def MCsimulate(nIter,*dists,method="IID"):
    nargs = len(dists)
    if method=="IID":
        if nargs == 1:
            return IIDsimulate(nIter,dists[0])
        else:
            return [IIDsimulate(nIter,d) for d in dists]
    
    elif method=="Latin Hypercube":
        raise ValueError("in developing!")
    elif method=="Antithetic":
        raise ValueError("in developing!")
    