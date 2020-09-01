#!/usr/bin/env python
# coding: utf-8

import scipy.stats
import numpy as np


__all__ = ['fetch_dist','get_stats']


class Bernoulli_ME(object):
    "faster performance than scipy.stats, allow mutually exclusive sampling"
    def __init__(self,p,lp=0):
        self.p = p
        self.lp = lp
        
    def set_pBoundary(self,lp):
        "p value boundary for mutually exclusive sampling"
        self.lp = lp
        self.up = self.p + lp
        if self.lp < 0 or self.up > 1:
            raise ValueError("Mutually exclusive risks pvals > 1!")
        
    def rvs(self,size=1):
        if 0 <= self.p <= 1:
            if isinstance(size,int): size = (size,)
            return self.ppf(np.random.rand(*size))
        else:
            raise ValueError("Domain error in arguments.")
            
    def ppf(self,q):
        res = np.zeros(q.shape)
        if self.lp == 0:
            res[q<=self.p] = 1
        else:
            res[(self.lp<q) & (q<=self.up)] = 1
        return res


class NoImpact(object):
    def __init__(self,*args):
        pass
    def rvs(self,size=1):
        return


class NoDistribution(object):
    def __init__(self,minimum,mostlikely,maximum):
        self.constant = mostlikely
        
    def rvs(self,size=1):
        return np.full(shape=size,fill_value=self.constant)


class SumOfAllTasks(object):
    def __init__(self,minimum,mostlikely,maximum):
        pass
    def rvs(self,size=1):
        return np.empty(size)


def Uniform(minimum,mostlikely,maximum):
    return scipy.stats.uniform(loc=minimum,scale=maximum-minimum)


def Pert(minimum,mostlikely,maximum):
    alpha = 1 + 4*(mostlikely-minimum)/(maximum-minimum)
    beta = 6 - alpha
    return scipy.stats.beta(a=alpha,b=beta,loc=minimum,scale=maximum-minimum)


def Normal_6sigma(minimum,mostlikely,maximum):
    mu = (minimum+maximum)/2
    sigma = (maximum-mu)/3
    return scipy.stats.norm(loc=mu,scale=sigma)


def Triangular(minimum,mostlikely,maximum):
    scale = maximum-minimum
    c = (mostlikely-minimum)/scale
    return scipy.stats.triang(c=c,loc=minimum,scale=scale)


# ------------- self-defined dists-------------
definedDist = {'No Distribution': NoDistribution,
               'Project Duration': SumOfAllTasks,
               'Uniform': Uniform,
               'Pert': Pert,
               'Normal': Normal_6sigma,
               'Triangular': Triangular,
               'Bernoulli': Bernoulli_ME,
               'Poisson': scipy.stats.poisson,
               'N/A': NoImpact}

# ------------- self-defined stats -------------
definedStats = {'Minimum': np.min,
                'Maximum': np.max,
                'Mean': np.mean,
                'Median': np.median,
                'Std. Deviation': np.std,
                'Skewness': scipy.stats.skew,
                'Kurtosis': scipy.stats.kurtosis,
                }
percentage = range(1,100)
for q in percentage:
    definedStats[f'{q}th Percentile'] = \
        lambda x,q=q,**kwds:np.percentile(x,q=q,**kwds)


def fetch_dist(dist_name, *args, **kwds):
    "fetch and validate distribution for self-defined and also from scipy.stats"
    if dist_name in definedDist:
        dist = definedDist[dist_name](*args, **kwds)
    elif hasattr(scipy.stats,dist_name):
        dist = getattr(scipy.stats,dist_name)(*args, **kwds)
    else:
        raise AttributeError(f"Distribution '{dist_name}' not found in neither"
                             " self-defined distribution nor scipy.stats!")
    # validate distribution
    _ = dist.rvs()
    # store dist_name, args, kwargs
    dist.__name__ = dist_name
    dist._args, dist._kwargs = args,kwds
    return dist
    

def get_stats(arr,funcs):
    "reduce arr using funs along axis=0"
    if np.isscalar(funcs):
        return definedStats[funcs](arr,axis=0)
    else:
        return np.array([definedStats[func](arr,axis=0) for func in funcs])
            