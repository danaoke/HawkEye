#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:09:49 2020

@author: jiahexu
"""
import numpy as np
from collections import defaultdict,namedtuple
from itertools import product,chain
from core import *
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

__all__ = ['Optimization']


Project = namedtuple('Project',["id","name","constraints","npvDist"])

# read input params/functions for Optimization file
def read_Project(project_raw):
    nConstraints = len(project_raw[0]) - 6 # constraints # at [2:2+nConstraints)
    msg = lambda Id,name: f" ProjectId={Id},ProjectName='{name}'"
    
    result = []
    for row in project_raw:
        # skip row that row[0] == ""
        if row[0] == "": continue
        
        # process project Id and name
        Id = row[0]
        name = row[1]
        constraints = list(map(int,row[2:2+nConstraints]))
        
        # process&validate distribution columns
        try:
            dist = fetch_dist(*row[2+nConstraints:6+nConstraints])
        except Exception as e:
            raise ValueError(str(e)+msg(Id,name))
            
        result.append(Project(Id,name,constraints,dist))
    return sorted(result)

class dMCKnapsack(object):
    '''
    
    adjusted n-Dimension Multiple Choice Knapsack (n-MCKP) Problem
    Notation below following Knapsack Problems, H. Kellerer, U. Pferschy, D. Pisinger
    
    Problem Description:
        we have n items with weight matrix W[d*n] and profit array P[n], find items 
        set X = {x_1,...,x_n}, x_j \in {0,1}, subject to weight capacity array C[d], 
        and Other Constraints(Fixed Either-Or Set & If-Then Set).
        
        Fixed set FIX = {fix_1,...,fix_nFIX}
            item fix_m must be selected, m = 1,...,nFIX
        Either-Or set EO = {(eo_{1,1},...),...,(eo_{nEO,1},...)}:
            choose & only choose 1 item from (eo_{l,1},...) for EO_l, l = 1,...,nEO
        If-Then set IT = {(if_1,then_1),...(if_nIT,then_mIT)}:
            if item if_k selected, item then_k must be included, k = 1,...,nIT
    
    Mathematical Formula:
        maximize    \sum_{j=1}^n p_j*x_j
        subject to  \sum_{j=1}^n w_ij*x_j <= c_i, i = 1,...,d
                    x_{j in FIX} == 1, FIX:= Fixed set
                    \sum_{j \in EO_l} x_j == 1, EO_l:= Either-Or set, l = 1,...,nEO
                    valid set S = \bigcup_{(if_k,then_k)\in IT} {~if_k | then_k}
                    x_j \in {0,1}, j = 1,...,n.
        -pending
    '''
    def __init__(self,weight,capacity,fixed,eitheror,ifthen):
        self.Ws = np.array(weight) # weight matrix for m Knapsacks
        self.n, self.d = self.Ws.shape # item & constraints(c) number
        
        self._add_Constraints(capacity,fixed,eitheror,ifthen)
        self.optimizeFlag = False
    
    def _add_Constraints(self,capacity,fixed,eitheror,ifthen):
        "validate & simplify other constraints"
        self.Cs = tuple(capacity)
        self.capacityChanged = True
        
        self.FIX,self.EO,self.IT = set(fixed),eitheror,ifthen
        self.nFIX,self.nEO,self.nIT = list(map(len,[self.FIX,self.EO,self.IT]))
        
        self._identify_complexity()
        
    def _identify_complexity(self):
        '''
        identify estimated complexity of current problem setting
        complexity = 
        '''
        self.complexity = 2**(self.n-self.nFIX) * np.prod(
            [l/2**l for l in map(len,self.EO)]) * (3/4)**self.nIT
        self.spaceComplex = 2**(self.n-self.nFIX) * self.n
#         print(f"Estimated Complexity: {self.complexity:,.0f}\n"\
#               f"Space Complexity: {self.spaceComplex/2**10:,.2f} kiB")
        if self.spaceComplex > 2**30:
            raise ValueError("Limit Exceeded!\nOptimize current problem would require "\
                            f"at least {self.spaceComplex/2**30:.2f} GiB.\n Consider "\
                             "reducing Projects number or adding Fixed Projects")
        
    def bool2ids(self,bools):
        "transform bool array to project IDs"
        return [i for i,b in enumerate(bools) if b]
    
    def optimize(self,profit):
        if not self.optimizeFlag: self._BruteForce()
        # identify maximized
        profits = np.matmul(self.validPorts,profit)
        maxProfit = profits.max()
        idx = np.where(maxProfit==profits)[0]
        if len(idx)==1:
            idx = idx[0]
        else: 
            # multiple solution, choose minimum weight consumption
            weightsConsump = self.WXs[idx].tolist()
            idx = idx[min(range(len(idx)),key=weightsConsump.__getitem__)]
        weightsConsump = self.WXs[idx].tolist()
        portfolio = self.bool2ids(self.validPorts[idx])
        return maxProfit,weightsConsump,portfolio
        
    def _BruteForce(self):
        "calculate valid portfolio set subject to Fixed, Either-Or Set & If-Then Set"
        iters = product(*([True] if j in self.FIX else [False,True] 
                          for j in range(self.n)))
        nparr = np.fromiter(chain.from_iterable(iters),dtype=bool).reshape(-1,self.n)
        bools = np.all([nparr[:,eo].sum(axis=1)==1 for eo in self.EO] + 
                       [~nparr[:,it[0]]|nparr[:,it[1]] for it in self.IT],axis=0)
        # insert valid portfolio set satisfy Fixed, Either-Or Set & If-Then Set
        validPorts = nparr if bools is np.bool_(1) else nparr[bools]
        # identify valid situation satisfy weight constraints
        WXs = np.matmul(validPorts,self.Ws) # capacity used for valid situations
        isvalid = np.all(WXs<=self.Cs,axis=1)
        if not isvalid.any():
            raise ValueError("No feasible solution found!")
        self.validPorts,self.WXs = validPorts[isvalid],WXs[isvalid]
        self.optimizeFlag = True
        
    def maxProfits_along(self,profit,weightId=0):
        "get profits based on descending self.WXs[:,weightId]"
        if not self.optimizeFlag: self._BruteForce()
        idx = np.argsort(self.WXs[:,weightId])
        totalWeight_tmp = self.WXs[idx,weightId] # sorted weight loading, not unique
        totalProfit_tmp = np.matmul(self.validPorts[idx],profit) # corresponding profit
        # return dominant portfolio performance
        totalWeight,totalProfit = [0],[0]
        for w,p in zip(totalWeight_tmp,totalProfit_tmp):
            if p>totalProfit[-1] and w == totalWeight[-1]:
                totalProfit[-1] = p # rewrite maxprofit for current weight loading
            elif p>totalProfit[-1] and w > totalWeight[-1]:
                # append new dominant portfolio performance
                totalWeight.append(w); totalProfit.append(p)
        return totalWeight[1:]+[totalWeight_tmp[-1]],totalProfit[1:]+totalProfit[-1:]

class Optimization(object):
    unit = "$" # result unit
    fmt = lambda _,x,*args: f"-${-x:,.0f}" if x<0 else f"${x:,.0f}" # o-stream format
    title = "Portfolio Optimization"
    def __init__(self,input_dir):
        "setup projectIds, distributions, pre-process constraints"
        self._fetch_params(input_dir)
        self._process_params()
    
    def _fetch_params(self, input_dir):
        "fetch parameters in 'Portfolio Optimization' sheet in excel"
        project_raw = read_excel_raw(input_dir,sheets="Portfolio Optimization",skiprows=2)
        self.projects = read_Project(project_raw)
        
    def _process_params(self):
        "internal/external id mapping, greatest common divisor for constraints"
        self.nProjects = len(self.projects)
        self.names = [p.name for p in self.projects]
        # internal to external id mapping
        self.ids = [p.id for p in self.projects] # internal to external
        # external to internal id mapping
        self.id_ex2in = dict([(ex,i) for i,ex in enumerate(self.ids)])
        
        self.constraints = [p.constraints for p in self.projects]
        # deterministic & stochastic project NPVs
        self.projNPV_deter = [p.npvDist._args[1] for p in self.projects]
        self.projNPV_stoch = [p.npvDist.mean() for p in self.projects]
        self.NPVs_mean = self.projNPV_stoch # initialize simulated mean
        
        # initialize constraints
        self.capacity,self.fixed,self.eitheror,self.ifthen = (),(),(),()
        # process greatest common divisor (gcd) for constraints
        # self.gcd = np.gcd.reduce(self.constraints,axis=0)
        
    def get_ids(self,Ids):
        if isinstance(Ids,str):
            if Ids not in self.id_ex2in:
                raise ValueError(f"Project '{Ids}' not found!")
            return self.id_ex2in[Ids]
        elif isinstance(Ids,int):
            if Ids >= self.nProjects:
                raise ValueError(f"Project ID '{Ids}' not found!")
            return self.ids[Ids]
        elif hasattr(Ids,'__len__'):
            return tuple(self.get_ids(Id) for Id in Ids)
        else:
            raise TypeError(f"Unexpected arg Ids type: '{type(Ids)}'")
            
    def setup_Constraints(self,capacity,fixed,eitheror,ifthen):
        "validate constraints & identify deterministic&stochastic optimized portfolios"
        idx_hasrestrict = [i for i,c in enumerate(capacity) if c] # idx nonzero capacity
        capacity = tuple(c for c in capacity if c) # all nonzero capacity
        fixed,eitheror,ifthen = list(map(self.get_ids,[fixed,eitheror,ifthen]))
        assert any([capacity,fixed,eitheror,ifthen]), "No Constraints been set!"
        assert 0 in idx_hasrestrict, "Investment Constraint Not set!"
        if any([capacity!=self.capacity,fixed!=self.fixed,
                eitheror!=self.eitheror,ifthen!=self.ifthen]):
            constraints = np.array(self.constraints)[:,idx_hasrestrict]
            self.solver = dMCKnapsack(constraints,capacity,fixed,eitheror,ifthen)
            self.totalNPVs_deter,self.totalWeight_deter,self.port_deter = \
                self.solver.optimize(self.projNPV_deter)
            self.totalNPVs_stoch_,self.totalWeight_stoch,self.port_stoch = \
                self.solver.optimize(self.projNPV_stoch)
            self.capacity,self.fixed,self.eitheror,self.ifthen = \
                capacity,fixed,eitheror,ifthen
            
    def get_Constraints(self):
        return (self.capacity,*map(self.get_ids,[self.fixed,self.eitheror,self.ifthen]))
        
    def simulate(self,nIter):
        "simulate & calculate profits for stochastic NPVs"
        self.nIter = nIter
        self.NPVs_stoch = np.array(
            MCsimulate(self.nIter,[p.npvDist for p in self.projects]))
        self.NPVs_mean = self.NPVs_stoch.mean(axis=1)
        self.totalNPVs_stoch = sum(self.NPVs_mean[i] for i in self.port_stoch)
        self.portNPVs = self.NPVs_stoch[self.port_stoch].sum(axis=0)
        
    def get_xyaxies(self,x,y):
        curve = PchipInterpolator(x, y)
        x_plot = np.linspace(x[0],x[-1],1000)
        y_plot = curve(x_plot)
        return x_plot,y_plot
    
    def plot_step(self,isstoch=True):
        npvs = self.NPVs_mean if isstoch else self.projNPV_deter
        _,ax = plt.subplots()
        title = ("Stochastic" if isstoch else "Deterministic")+" Efficient Frontier"
        xlabel,ylabel = "Portfolio Investment","Portfolio Net Present Value"
        x_plot,y_plot=self.get_xyaxies(*self.solver.maxProfits_along(npvs,0))   
        ax.plot(x_plot,y_plot,color=None,lw=2)
        set_plotparams(ax,title=title,xlabel=xlabel,ylabel=ylabel,xfmt=self.fmt,yfmt=self.fmt,showfig=True)
        
    def plot_pdfcdf(self):
        label = "Optimum Portfolio NPV"
        plt_pdfcdf(self.portNPVs,title=label,xlabel=label,xfmt=self.fmt)
        
    def export(self):
        excelapi = excel_api(mode='w')
        header = ['Iteration \ Project Name','Portfolio NPV']+\
                 [self.names[i] for i in self.port_stoch]
        data = np.concatenate((self.portNPVs[:,None],self.NPVs_stoch[self.port_stoch].T),axis=1)
        excelapi.add_sheet('Optimization', data, header, index=range(1,self.nIter+1))
        x_plot,y_plot=self.get_xyaxies(*self.solver.maxProfits_along(self.NPVs_mean,0)) 
        excelapi.add_sheet('Efficient Frontier Data', np.array([x_plot,y_plot]).T.tolist(), ['Steps','Portfolio Investment','NPV'], index=range(1,self.nIter+1))
        excelapi.displayExcelApp()


if __name__ == "__main__":
    from configs import HOME
    filedir = HOME+"\\Excel Templates\\Portfolio Optimization\\Portfolio Optimization testcase.xlsx"
    # filedir="/Users/jiahexu/Downloads/hawkeyecode/Portfolio Optimization testcase.xlsx"
    capacity = [4500,45]
    fixed = ["K"]
    eitheror = [('B','C')]
    ifthen = [('G','H')]
    
    obj = Optimization(filedir)
    obj.setup_Constraints(capacity,fixed,eitheror,ifthen)
    obj.simulate(10000)
    
    obj.plot_step() # Effiecient Frontier
    obj.plot_pdfcdf()
    obj.export()