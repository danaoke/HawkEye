#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict,namedtuple
import matplotlib.pyplot as plt

from core import *

__all__ = ['NPV']


CashFlow = namedtuple('CashFlow',["year","name","dist","disp"])


#######################################################################
# read input params/functions for NPV file
def read_CFs(NPV_raw):
    msg = lambda year,name: f" Year {year} '{name}:' in sheet 'Net Present Value'"
    result = []
    for row in NPV_raw:
        # skip row that row[0] == ""
        if row[0] == "": continue
        
        # process year & interest rate
        year = float(row[0]) if row[0]%1 else int(row[0]) # for printing
        
        # process&validate investment,revenue,cost distribution columns
        n = 1 # start column number
        for name in ("Capital Cost","Revenue","Cost"):
            if row[n]:
                try:
                    dist = fetch_dist(*row[n:n+4])
                except Exception as e:
                    raise ValueError(str(e)+msg(year,name))
                result.append(CashFlow(year,name,dist,f"Year {year}: {name}"))
            n += 4
            
    return sorted(result)


def sensitivity1D(xs,ys,q=[5,95]):
    "sensitivity for tornado plots in Schedule/Cost, return conditional mean of ys"
    assert isinstance(xs,np.ndarray) and xs.ndim==2, \
        "args xs restricted to be 1D numpy.ndarray"

    # calculate y average and sensitivity to x change
    ymean = ys.mean() # y avearge
    perc5,perc95 = np.percentile(xs, q, axis=0) # q percentiles
    lows = [ys[b].mean() for b in (xs<=perc5).T] # bottom 5% mean
    highs = [ys[b].mean() for b in (xs>=perc95).T] # top 5% mean

    return ymean,lows,highs


# data processing class code for Module NPV
class NPV(object):
    unit = "$" # result unit
    fmt = lambda _,x,*args: f"-${-x:,.0f}" if x<0 else f"${x:,.0f}" # o-stream format
    title = "Net Present Value"
    def __init__(self,input_dir):
        "setup taskIds, distributions, risk events"
        self._fetch_params(input_dir)
        self._process_NPVParams()
        
    def _fetch_params(self, input_dir):
        "fetch parameters in 'Net Present Value' sheet in excel"
        NPV_raw = read_excel_raw(input_dir,sheets="Net Present Value", skiprows=2)
        self.CFs = read_CFs(NPV_raw)
    
    def _process_NPVParams(self):
        "cashflows and identify time points"
        self.nCFs = len(self.CFs)
        self.years = [CF.year for CF in self.CFs]
        self.disps = [CF.disp for CF in self.CFs]
        
    def setup_DiscountFactors(self,rates):
        "calculate discount factor"
        self.rates = rates
        self.nRates = len(self.rates)
        self.DFs = (1+np.array(self.rates)[:,None])**(-np.array(self.years))
    
    def simulate(self,nIter):
        "simulate CashFlows using MonteCarloSimulation module function"
        self.nIter = nIter
        self.cashflows = np.array(MCsimulate(self.nIter,[c.dist for c in self.CFs])).T
        
    def calculate_simresults(self):
        "calculate Net Present Values"
        # change sign of cash flows and calculate NPV
        DFs = self.DFs*[-1 if "Cost" in CF.name else 1 for CF in self.CFs]
        self.NPVs = np.matmul(self.cashflows, DFs.T)
    
    def plot_PDFCDF(self, rateId=0, figtype='PDF&CDF'):
        xlabel = self.title
        legend = [f"Discount Rate: {r:.2%}" for r in self.rates]
        if figtype=='PDF&CDF':
            xlabel += f", discount rate: {self.rates[rateId]:.2%}"
            plt_pdfcdf(self.NPVs[:,rateId],xlabel=xlabel,xfmt=self.fmt)
        elif figtype=='PDF':
            _, ax = plt.subplots()
            bins = np.linspace(self.NPVs.min(),self.NPVs.max(),71)
            _ = [plt_hist(var,ax=ax,bins=bins,color=None,alpha=0.5) 
                 for var in self.NPVs.T[:-1]]
            plt_hist(self.NPVs[:,-1],ax=ax,bins=bins,color=None,alpha=0.5,
                     xlabel=xlabel,xfmt=self.fmt,legend=legend,showfig=True)
        elif figtype=='CDF':
            _, ax = plt.subplots()
            _ = [plt_cdf(var,ax=ax,color=None) for var in self.NPVs.T[:-1]]
            plt_cdf(self.NPVs[:,-1],ax=ax,color=None,xlabel=xlabel,xfmt=self.fmt,
                    legend=legend,showfig=True)
            
    def plot_performance(self, rateId=0, figtype='Tornado Diagram'):
        y = self.disps
        title = f"{self.title} {figtype}, Discount Rate: {self.rates[rateId]:.2%}"
        if figtype=='Tornado Diagram':
            mean,lows,highs = sensitivity1D(self.cashflows,self.NPVs[:,rateId])
            width = np.subtract(highs,lows)
            xlabel = f'Average: {self.fmt(mean)}'
            plt_sortedbarh(y,width,lows,vline=mean,txtfmt=self.fmt,
                           xlabel=xlabel,title=title,xfmt=self.fmt)
        
    def browse(self):
        excelapi = excel_api(mode='w')
        # add Cash Flows sheets
        header = ['Iteration \ Cash Flows']+self.disps
        excelapi.add_sheet('CashFlows', self.cashflows, header, range(1,self.nIter+1))
        # add NPV sheets
        header = ['Iteration \ Discount Rates']+[f"{r:.2%}" for r in self.rates]
        excelapi.add_sheet('NPVs', self.NPVs, header, range(1,self.nIter+1))
        excelapi.displayExcelApp()


if __name__ == "__main__":
    from configs import HOME
    filedir = HOME+"\\Excel Templates\\Net Present Value\\Net Present Value.xlsx"
    obj = NPV(filedir)
    obj.simulate(10000)
    obj.setup_DiscountFactors([0.15,0.20,0.25,0.3])
    obj.calculate_simresults()
    
    obj.plot_PDFCDF()
    obj.plot_PDFCDF(figtype="PDF")
    obj.plot_PDFCDF(figtype="CDF")
    obj.plot_performance()