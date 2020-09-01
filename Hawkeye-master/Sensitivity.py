#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict,namedtuple
from itertools import product,chain

from core import *
from NPV import CashFlow, NPV
from ScheduleCost import TaskMainProp, Cost, Schedule


__all__ = ['SensitivityMods','Sensitivity']


#######################################################################
# overwritted func & params for sensitivity module
def validata(dist,minimum,mostlikely,maximum):
    "distribution detecter overwrite for sensitivity module"
    isvalid = mostlikely != "" and isinstance(mostlikely,(int,float))
    return isvalid,('No Distribution',minimum,mostlikely,maximum)


def read_CFs(NPV_raw):
    result = []
    for row in NPV_raw:
        # skip row that row[0] == ""
        if row[0] == "": continue
        
        # process year & interest rate
        year = float(row[0]) if row[0]%1 else int(row[0]) # for printing
        
        # process&validate investment,revenue,cost distribution columns
        n = 1 # start column number
        for name in ("Capital Cost","Revenue","Cost"):
            isvalid,distargs = validata(*row[n:n+4])
            if isvalid:
                dist = fetch_dist(*distargs)
                result.append(CashFlow(year,name,dist,f"Year {year}: {name}"))
            n += 4
    return sorted(result)


def read_Task(task_raw,sheet='Schedule'):
    msg = lambda Id,name: f" TaskId={Id},TaskName='{name}' in sheet '{sheet}'"
    result = []
    for row in task_raw:
        # skip row that row[0] == ""
        if row[0] == "": continue
        
        # process task Id and name
        Id = int(row[0])
        name = row[1]
        
        # process&validate distribution columns
        isvalid,distargs = validata(*row[2:6])
        if isvalid:
            dist = fetch_dist(*distargs)
        else:
            raise ValueError("Most Likely value not specified!"+msg(Id,name))
        
        # process predecessor
        if sheet == 'Schedule':
            tmp = row[6]
            try:
                if isinstance(tmp,(float,int)):
                    predecessor = [int(tmp)]
                elif isinstance(tmp,str):
                    predecessor = list(map(int,tmp.split()))
            except:
                raise ValueError(f"Unexpected value '{tmp}' in column 'Predecessor' for"
                                 +msg(Id,name))
        else:
            predecessor = None
        result.append(TaskMainProp(Id,name,dist,predecessor))
    return sorted(result)


#######################################################################
# Sensitivity Module Data Processing Part
class SensiParent(object):
    title = "" # title in Tornado plot
    def _fetch_params(self, input_dir):
        "overwrite inputs reader, constrain varible name as follow"
        # original class var name to allow original process params
        self.vars = [] # list of namedtuples storing variable info
        self.disps = [] # list of display str for each 
    
    def setup_userInputs(self, percents, *args):
        "setup user input from GUI if provided"
        self.percents = [p/100 for p in percents[:2]]
    
    def analyze(self):
        # construct varValues, 
        # row 0 mostlikely values, row 1:end +/- percent for each var
        nVars = self.nVars = len(self.vars)
        varValues = np.repeat([[float(v.dist._args[1]) for v in self.vars]],
                              repeats=2*nVars+1, axis=0)
        varValues[range(1,1+nVars),range(nVars)] += varValues[0] * self.percents[0]
        varValues[range(-nVars,0),range(nVars)] += varValues[0] * self.percents[1]
        
        # calculate base/+/- results, inherited class overwrite functions
        results = self.get_sensitivity(varValues)
        self.baseValue = results[0]
        self.minus,self.plus = results[1:1+nVars],results[-nVars:]
        
    def get_sensitivity(self,varValues):
        "2D -> 1D reduce function for mapping varValues to total NPV/Cost/Duration"
        pass
    
    def plot_Tornado(self,topN=10):
        topN = int(topN) if topN > 1 else int(np.ceil(self.nVars*topN))
        width = np.subtract(self.plus,self.minus)
        txtfmt = getattr(self,'txtfmt',self.fmt)
        xlabel = f'Base Case: {txtfmt(self.baseValue)}'
        title = f"{self.title} - Top {topN} Impacts"
        plt_sortedbarh(self.disps,width,self.minus,vline=self.baseValue,topN=topN,
                       txtfmt=txtfmt,xlabel=xlabel,title=title,xfmt=self.fmt)
   
    def export(self):
        title = "Sensitivity Analysis | "+self.title
        excelapi = excel_api(mode='w')
        header=[title,f"{self.percents[0]:.2%}","Base Case",f"+ {self.percents[1]:.2%}"]
        data = np.array([self.minus,[self.baseValue]*self.nVars,self.plus]).T
        excelapi.add_sheet(self.title, data, header, self.disps)
        excelapi.displayExcelApp()


class NPVSensi(NPV,SensiParent):
    title = "Net Present Value"
    def _fetch_params(self, input_dir):
        "overwrite inputs reader, constrain varible name as follow"
        NPV_raw = read_excel_raw(input_dir, sheets="Net Present Value", skiprows=2)
        # list of namedtuples storing variable info
        self.CFs = self.vars = read_CFs(NPV_raw)
        
    def setup_userInputs(self, percents, *args):
        "setup user input from GUI if provided"
        super().setup_userInputs(percents,*args)
        self.DFs = (1+args[0]/100)**(-np.array(self.years))
        
    def get_sensitivity(self,varValues):
        "2D -> 1D reduce function for mapping varValues to total NPV/Cost/Duration"
        DFs = self.DFs*[-1 if "Cost" in CF.name else 1 for CF in self.CFs]
        return np.matmul(varValues, DFs)


class CostSensi(Cost,SensiParent):
    title = "Project Cost"
    def _fetch_params(self, input_dir):
        "overwrite inputs reader, constrain varible name as follow"
        task_raw = read_excel_raw(input_dir, sheets='Cost', skiprows=1)
        # list of namedtuples storing variable info
        self.tasks = self.vars = read_Task(task_raw,'Cost')
        self.risks = [] # no risk applied
        self.disps = [f"{s.id}: {s.name}" for i,s in enumerate(self.tasks)]
        
    def get_sensitivity(self,varValues):
        "2D -> 1D reduce function for mapping varValues to total NPV/Cost/Duration"
        return varValues.sum(axis=1)


class ScheduleSensi(Schedule,SensiParent):
    title = "Project Duration, days"
    txtfmt = lambda _,x: f"{x:,.0f} days"
    def _fetch_params(self, input_dir):
        "overwrite inputs reader, constrain varible name as follow"
        task_raw = read_excel_raw(input_dir, sheets='Schedule', skiprows=1)
        # list of namedtuples storing variable info
        self.tasks = self.vars = read_Task(task_raw,'Schedule')
        self.risks = [] # no risk applied
        self.disps = [f"{s.id}: {s.name}" for i,s in enumerate(self.tasks)]
        
    def get_sensitivity(self,varValues):
        "2D -> 1D reduce function for mapping varValues to total NPV/Cost/Duration"
        return self.get_totalDuration(self.get_cumDuration(varValues))


SensitivityMods = {
    "Net Present Value": NPVSensi,
    "Project Cost": CostSensi,
    "Project Schedule": ScheduleSensi
}

# function wrapper
def Sensitivity(input_dir, mod): 
    assert mod in SensitivityMods, f"Module {mod} not available yet!"
    return SensitivityMods[mod](input_dir)


if __name__ == "__main__":
    from configs import HOME
    testcases = [("Net Present Value",'Net Present Value Deterministic.xlsx'),
                 ("Project Cost",'Project Zulu Cost Deterministic.xlsx'),
                 ("Project Schedule",'Project Zulu Schedule Deterministic.xlsx')]
    for mod, filename in testcases:
        filedir = HOME+"\\Excel Templates\\Sensitivity Analysis\\"+filename
        obj = SensitivityMods[mod](filedir)
        obj.setup_userInputs([-20,10],20)
        obj.analyze()
        obj.plot_Tornado()
    obj.export()