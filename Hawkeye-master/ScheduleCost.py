#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict,namedtuple
import matplotlib.pyplot as plt

from core import *


__version__ = 'v2.0' # stable version 2, add inheritances
__all__ = ['Cost','Schedule','ScheduleCost']


TaskMainProp = namedtuple('TaskMainProp',["id","name","dist","predecessor"])
TaskCost = namedtuple('TaskCost',["id","name","dailyCost","fixedCost"])
Risk = namedtuple('Risk',["id","name","relaventId","freqDist",
                          "scheduleImpactDist","scheduleTaskId","scheduleOperator",
                          "costImpactDist","costTaskId","costOperator"])

allowOperators = {"Add":np.add, "Substract":np.subtract, 
                  "Maximum":np.maximum,"Minimum":np.minimum}


#######################################################################
# read input params/functions for Schedule/Cost file
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
        try:
            dist = fetch_dist(*row[2:6])
        except Exception as e:
            raise ValueError(str(e)+msg(Id,name))
        
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


def read_Cost(cost_raw):
    msg = lambda Id,name: f" TaskId={Id},TaskName='{name}' in sheet 'Cost'"
    PMcoststr = "PM cost as Percentage of Sum Cost of All Tasks:" # identify str
    result,PMcost = [],0
    for row in cost_raw:
        # skip row that row[0] == ""
        if row[0] == "": 
            if not any(row[1:7]) and row[7]==PMcoststr and row[11]:
                PMcost = float(row[11])
            continue
        
        # process task Id and name
        Id = int(row[0])
        name = row[1]
        
        # process daily cost
        if row[6]!="":
            dailyCost = float(row[6])
        else:
            raise ValueError("Daily Cost missing for "+msg(Id,name))
        # process fixed cost
        try:
            fixedCost = 0. if row[7] == "" else fetch_dist(*row[7:11])
        except Exception as e:
            raise ValueError(str(e)+msg(Id,name))
        
        result.append(TaskCost(Id,name,dailyCost,fixedCost))
    return sorted(result), PMcost


def read_Risk(risk_raw,impactsApplied=['Schedule','Cost']):
    msg = lambda Id,name: f" RiskId={Id},RiskName='{name}' in sheet 'Risk Register'"
    result = [] 
    for row in risk_raw:
        # skip row that row[0] == ""
        if row[0] == "": continue
        
        # process risk Id and name
        Id = int(row[0])
        name = row[1]
        
        # process relaventId, freqDist
        tmp = row[3]
        if tmp and row[4] != 'Bernoulli':
            raise ValueError("Mutually Exclusive only support for Risk Frequency "
                             "Distribution is 'Bernoulli'!"+msg(Id,name))
        try:
            if isinstance(tmp,(float,int)):
                relaventId = [int(tmp)]
            elif isinstance(tmp,str):
                relaventId = list(map(int,tmp.split()))
        except:
            raise ValueError(f"Unexpected value '{tmp}' in column 'Risk Id' "
                             "Mutually Exclusive to' for"+msg(Id,name))
        try:
            freqDist = fetch_dist(*row[4:6])
        except Exception as e:
            raise ValueError(str(e)+" Risk Frequency Distribution,"+msg(Id,name))
            
        # process Schedule&Cost Impact if Impact Distribution column is specified
        n = 6 # start column number
        values = [] # temp list to store ImpactDist,TaskId,Operator
        for s in ("Schedule","Cost"):
            if s not in impactsApplied or row[n] in ("No Impact",""):
                values += [None] * 3
                if s not in impactsApplied: continue
            else:
                # try to construct ImpactDist
                try:
                    dist = fetch_dist(*row[n:n+4])
                except Exception as e:
                    raise ValueError(str(e)+f" {s} Impact Distribution, "+msg(Id,name))
                
                TaskId = row[n+4] # TaskId impacted
                assert TaskId!="", f"{s} Task Id Impacted missing! "+msg(Id,name)
                
                operator = row[n+5] # Operator applied
                assert operator in allowOperators, \
                    s+" Impact Operator should be one of " + \
                    ",".join(map(str,allowOperators)) + msg(Id,name)
                values += [dist,int(TaskId),allowOperators[operator]]
            n += 6
        result.append(Risk(Id,name,relaventId,freqDist,*values))
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


#######################################################################
# algorithm functions for Schedule/Cost
def TopoSort(edges,nNodes):
    '''
    Topological sorting for Directed Acyclic Graph (DAG)
    
    ref: https://www.geeksforgeeks.org/topological-sorting/
    '''
    # add directed graph
    graph = defaultdict(list)
    for u,v in edges:
        graph[u].append(v)
    
    def topologicalSortUtil(v,visited,stack): 
        # Mark the current node as visited. 
        visited[v] = True
        
        # Recur for all the vertices adjacent to this vertex 
        for i in graph[v]:
            if not visited[i]: 
                topologicalSortUtil(i,visited,stack) 
                
        # store current vertex to stack which stores result 
        stack.append(v)
        
    # The function to do Topological Sort. It uses recursive topologicalSortUtil
    visited = [False]*nNodes
    stack = [] 
    
    # Call the recursive helper function to store Topological 
    # Sort starting from all vertices one by one 
    for i in range(nNodes): 
        if not visited[i]: 
            topologicalSortUtil(i,visited,stack) 

    # return contents of the stack 
    return stack[::-1]


def KahnTopoSort(predecessor,successor,nNodes):
    '''
    Kahn’s algorithm Topological sorting for Directed Acyclic Graph (DAG)
    
    ref: https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
    '''
    # indegrees of all vertices as length of predecessor
    in_degree = list(map(len,predecessor))
    # Create an queue and enqueue all vertices with indegree 0 
    queue = [i for i,l in enumerate(in_degree) if l==0]
    queueN = len(queue)
    
    top_order = [] # result vector(A topological ordering of the vertices) 
    count = 0 # Initialize count of visited vertices 
    # One by one dequeue vertices from queue and enqueue 
    # adjacents if indegree of adjacent becomes 0 
    while queueN>count:
        # Extract front of queue (or perform dequeue) 
        # and add it to topological order 
        u = queue[count]
        top_order.append(u) 
        
        # Iterate through all neighbouring nodesof dequeued node u and 
        # decrease their in-degree by 1 
        for i in successor[u]: 
            in_degree[i] -= 1
            # If in-degree becomes zero, add it to queue 
            if in_degree[i] == 0:
                queueN += 1
                queue.append(i)
        
        count += 1
    
    if count != nNodes: raise ValueError("Cycle detected in the Task Predecessor!")
    return top_order # return contents of the stack 


def DFSConnectedComponents(edges,nNodes):
    '''
    DFS for finding Connected Components in an undirected graph
    
    ref: https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
    '''
    # construct undirected graph
    graph = [[] for _ in range(nNodes)]
    for u,v in edges:
        graph[u].append(v)
        graph[v].append(u)
        
    def DFSUtil(v,visited,temp):
        visited[v] = True # Mark the current vertex as visited 
        temp.append(v) # Store the vertex to list 
        
        # recursive function to find all linked vertices
        for i in graph[v]:
            if not visited[i]:
                DFSUtil(i,visited,temp)
    
    visited = [False] * nNodes
    cc = []
    # Method to retrieve connected components in an undirected graph 
    for i in range(nNodes):
        if not visited[i]:
            temp = []
            DFSUtil(i,visited,temp)
            cc.append(temp)
    
    return cc


#######################################################################
# class code for Schedule/Cost
class Cost(object):
    riskImpacts = 'Cost' # risk impacts applied, for _fetch_params and simulate
    unit = "$" # result unit
    fmt = lambda _,x,*args: f"-${-x:,.0f}" if x<0 else f"${x:,.0f}" # o-stream format
    title = lambda _,withrisk,arg='Project': f"{arg} Cost" if withrisk is None else (
        f"{arg} Cost (with{'' if withrisk else 'out'} Risk Events)")
    def __init__(self,input_dir=None):
        "setup taskIds, distributions, risk events"
        if input_dir is not None:
            self._fetch_params(input_dir)
            self._process_taskParams()
            self._process_riskParams()
    
    def _fetch_params(self, input_dir):
        "fetch parameters in 'Schedule'/'Cost','Risk Register' sheet in excel"
        task_raw, risk_raw = read_excel_raw(
            input_dir,sheets=['Cost','Risk Register'], skiprows=[1,2])
        
        # defined function for reading schedule,risk paramters
        self.tasks = read_Task(task_raw,'Cost')
        self.risks = read_Risk(risk_raw,self.riskImpacts)
        
    def _process_taskParams(self):
        "internal/external id mapping"
        self.nTasks = len(self.tasks)
        # internal to external id mapping
        self.ids = [t.id for t in self.tasks] # internal to external
        # external to internal id mapping
        self.id_ex2in = dict([(ex,i) for i,ex in enumerate(self.ids)])
        
        # raise Exception if external TaskId is not unique
        assert len(self.id_ex2in) == self.nTasks, \
            "Duplicated Task Id found in Excel Sheet!"
        
    def _process_riskParams(self):
        "risk internal/external id mapping, dist mutual exclusive relationship"
        self.nRisks = len(self.risks)
        # internal to external risk id mapping
        self.riskIds = [r.id for r in self.risks] 
        # external to internal risk id mapping
        self.riskId_ex2in = dict([(e,i) for i,e in enumerate(self.riskIds)])
        
        # identify connected riskIds, unique randomness number 
        edges = [(i,self.riskId_ex2in[j]) for i,r in enumerate(self.risks) 
                 for j in r.relaventId]
        self.risk_connected = DFSConnectedComponents(edges,self.nRisks)
        self.freqRandomN = len(self.risk_connected) # unique randomness number
        
        # link risk to freqRandomId and insert Bernoulli dist lower bound
        self.freqRandomIds = [None] * self.nRisks
        for i,riskIds in enumerate(self.risk_connected):
            lp = 0 # lower bound for pvalue in Bernoulli
            ismutualexclusive = len(riskIds) > 1
            for riskId in riskIds:
                # identify unique frequency randomness to use for sampling
                self.freqRandomIds[riskId] = i
                # set task Bernoulli dist p bounds
                if ismutualexclusive:
                    dist =  self.risks[riskId].freqDist
                    dist.set_pBoundary(lp)
                    lp += dist.p
                    
    def simulate(self,nIter):
        "simulate task cost&risk impact using MonteCarloSimulation module function"
        self.nIter = nIter
        # simulate cost_norisk
        self.task_norisk = np.array(MCsimulate(self.nIter,[t.dist for t in self.tasks])).T
        
        # simulate risk frequnecy/impacts
        freqRandom = [np.random.rand(nIter) for _ in range(self.freqRandomN)]
        self.riskFreqs = [r.freqDist.ppf(freqRandom[randomId]).astype(int) 
                          for r,randomId in zip(self.risks,self.freqRandomIds)]
        self._simulate_impact(self.riskImpacts)
        
    def _simulate_impact(self,riskimpacts):
        if isinstance(riskimpacts,str) and riskimpacts in ('Schedule','Cost'):
            s = riskimpacts.lower()+'ImpactDist' # attribute name for risk impact
            severitys = [getattr(r,s).rvs(self.nIter) if getattr(r,s)
                         else None for r in self.risks]
            # calculate simulated risk impact
            self.__dict__[riskimpacts.lower()+'Impact'] = self.get_impact(severitys)
        elif isinstance(riskimpacts,(list,tuple)):
            [self._simulate_impact(impact) for impact in riskimpacts]
        else:
            raise ValueError(f"Unexpected arg riskimpacts value: {riskimpacts}")
        
    def get_impact(self,severity):
        "calculte risk impact using freq&severity, return list of np.array/None"
        impact = []
        for freq,severity in zip(self.riskFreqs,severity):
            if severity is None:
                impact.append(None)
            else:
                # if severity is not None, calculate impact
                if not any(freq > 1):
                    # bernoulli frequency
                    impact.append(freq*severity)
                else:
                    # frequency greater than one, impact=sum(impactDist.rvs(freq))
                    rvs_needed = freq.sum()+1 # severity rvs needed
                    if len(severity)>rvs_needed:
                        severity = severity[:rvs_needed]
                    elif len(severity)<rvs_needed:
                        # sample more rvs if rvs number less that needed
                        _rvs = risk.scheduleImpactDist.rvs(rvs_needed-len(severity))
                        severity = np.concatenate((severity,_rvs))
                        
                    impact.append(np.diff(
                        severity.cumsum()[freq.cumsum()],prepend=severity[0]))
        return impact
    
    def calculate_simresults(self):
        "calculate all simulated cost with/without risk"
        self.cost_norisk = self.task_norisk # rename cost without risk
        self.cost_risk = self.get_withrisk(self.cost_norisk,self.costImpact)
        
        # get Project Total Cost with/without risk
        self.totalCost_norisk = self.cost_norisk.sum(axis=1)
        self.totalCost_risk = self.cost_risk.sum(axis=1)
        
    def get_withrisk(self,norisk,impact,name='cost'):
        "calculate task duration/cost with risk using *_norisk and impact"
        withrisk = norisk.copy()
        if isinstance(withrisk,np.ndarray): withrisk = withrisk.T
        for i,risk in enumerate(self.risks):
            if impact[i] is None: continue # no impact, skip to next risk
            
            taskId = self.id_ex2in[getattr(risk,name+'TaskId')] # task impacted
            operator = getattr(risk,name+'Operator') # operator applied
            if operator != np.minimum:
                # directly applied operator to whole array
                # _ = operator(withrisk[taskId],impact[i],out=withrisk[taskId])
                withrisk[taskId] = operator(withrisk[taskId],impact[i])
            else:
                # only applied operator to iterations that impacted
                isimpacted = self.riskFreqs[i]>0 # iteration impacted
                tmp = np.empty_like(impact[i]); tmp[:] = withrisk[taskId]
                withrisk[taskId] = operator(tmp,impact[i],where=isimpacted,out=tmp)
                
        return withrisk.T if isinstance(withrisk,np.ndarray) else withrisk
        
    def get(self,varnames,withrisk=None):
        "get variables in self.__dict__"
        if isinstance(varnames,str):
            if withrisk is not None: varnames += '_risk' if withrisk else '_norisk'
            return self.__dict__[varnames]
        elif isinstance(varnames,(list,tuple)):
            return [self.get(var,withrisk) for var in varnames]
        
    def plot_PDFCDF(self, var='totalCost', figtype='PDF&CDF', withrisk=True):
        xlabel = self.title(None); legend=['With Risk Events','Without Risk Events']
        if figtype=='PDF&CDF':
            plt_pdfcdf(self.get(var,withrisk),xlabel=self.title(withrisk),xfmt=self.fmt)
        elif figtype=='PDF':
            _, ax = plt.subplots()
            bins = plt_hist(self.get(var,True),ax=ax,color='tab:red',alpha=0.5)
            plt_hist(self.get(var,False),ax=ax,bins=bins,alpha=0.5,xlabel=xlabel,
                     xfmt=self.fmt,legend=legend,showfig=True)
        elif figtype=='CDF':
            _, ax = plt.subplots()
            plt_cdf(self.get(var,True),ax=ax,color='tab:red')
            plt_cdf(self.get(var,False),ax=ax,color='tab:blue',xlabel=xlabel,
                    xfmt=self.fmt,legend=legend,showfig=True)
            
    def plot_performance(self, figtype='Tornado Diagram', withrisk=True):
        y = [f"{s.id}: {s.name}" for i,s in enumerate(self.tasks)]
        title = f"{self.title(withrisk)} {figtype}"
        if figtype=='Tornado Diagram':
            cost,totalCost = self.get(['cost','totalCost'],withrisk)
            mean,lows,highs = sensitivity1D(cost,totalCost)
            width = np.subtract(highs,lows)
            xlabel = f'Average: {self.fmt(mean)}'
            plt_sortedbarh(y,width,lows,topN=10,vline=mean,txtfmt=self.fmt,
                           xlabel=xlabel,title=title,xfmt=self.fmt)
        elif figtype=='Correlation Coefficients':
            corr = np.corrcoef(self.get('cost',withrisk).T, 
                               self.get('totalCost',withrisk))[-1,:-1]
            corr = np.where(np.isnan(corr), 0, corr)
            txtfmt = "{x:.3f}" # using 3-digit float format for bar width text
            plt_sortedbarh(y,corr,topN=10,vline=0,txtfmt=txtfmt,title=title,xlim=[-1,1])
            
    def _browseCost(self, excelapi, withrisk=True):
        header = [['Task Id','Project Total Cost'] + self.ids, 
                  ['Iteration \ Task Name','Project Total Cost'] + \
                      [task.name for task in self.tasks]]
        cost,totalCost = self.get(['cost','totalCost'],withrisk)
        data = np.concatenate((totalCost[:,None],cost),axis=1)
        excelapi.add_sheet('Cost', data, header, index=range(1,self.nIter+1))
        
    def _browseRiskEvents(self, excelapi, withrisk=True):
        header=[sum([[i,''] for i in self.riskIds],['Risk Id']),
                sum([[r.name,''] for r in self.risks],['Risk Name']),['Iteration']
                +['Risk Frequency',self.riskImpacts+' Impact']*self.nRisks]
        riskImpactData = self.__dict__[self.riskImpacts.lower()+'Impact']
        data = [d.tolist() if d is not None else None for ds in 
                zip(self.riskFreqs,riskImpactData) for d in ds]
        excelapi.add_sheet('Risk Events', data, header, range(1,self.nIter+1), axis=1)
        
    def _browse(self, sheets, withrisk=True):
        excelapi = excel_api(mode='w')
        [getattr(self,'_browse'+s.replace(' ',''))(excelapi,withrisk) for s in sheets]
        excelapi.displayExcelApp()
    
    def browse(self, withrisk=True):
        sheets = ['Cost','Risk Events'] if withrisk else ['Cost'] # set default sheets
        self._browse(sheets, withrisk)


class Schedule(Cost):
    '''
    Data processing part for module Schdule, 
    
    Functionality:
    1. read/process schedule&risk parameters from inputs file, see excel templates
    2. simulate randomness using Module functions in core.MonteCarloSimulation 
    3. calculate simulation results, includes duration/cumDuration with/without risk
    4. other functions for supporting GUI buttons
    '''
    riskImpacts = 'Schedule' # risk impacts applied, for _fetch_params and simulate
    unit = "days" # result unit
    fmt = None # No plot format
    title = lambda _,withrisk,arg='Project': f"{arg} Duration" if withrisk is None else (
        f"{arg} Duration (with{'' if withrisk else 'out'} Risk Events)")
    
    def _fetch_params(self, input_dir):
        "fetch parameters in 'Schedule','Risk Register' sheet in excel"
        task_raw, risk_raw = read_excel_raw(
            input_dir,sheets=['Schedule','Risk Register'], skiprows=[1,2])
        
        # defined function for reading schedule,risk paramters
        self.tasks = read_Task(task_raw,'Schedule')
        self.risks = read_Risk(risk_raw,self.riskImpacts)
        
    def _process_taskParams(self):
        "internal/external id mapping, construct task dependency"
        Cost._process_taskParams(self) # internal/external id mapping inherited
        
        # mapping predecessor id from external to internal
        self.predecessor = [[self.id_ex2in[d] for d in task.predecessor] 
                            for task in self.tasks]
        self.successor = [[] for _ in range(self.nTasks)]
        for i,dep in enumerate(self.predecessor): 
            for d in dep: 
                self.successor[d].append(i)
                
        # Kahn’s algorithm Topological sorting to find execution sequence
        self.executeSeq = KahnTopoSort(self.predecessor,self.successor,self.nTasks)
        
        # project end Id: has predecessor, no successor
        self.endIds = [i for i in range(self.nTasks)
                       if self.predecessor[i] and not self.successor[i]]
        # identity global task that is 'Project Duration'
        self.hammockIds = [self.id_ex2in[task.id] for task in self.tasks 
                           if task.dist.__name__=='Project Duration']
        
    def calculate_simresults(self):
        "calculate all simulated duration/cumDuration with/without risk"
        self.duration_norisk = self.task_norisk # rename duration without risk
        self.duration_risk = self.get_withrisk(self.duration_norisk,self.scheduleImpact,
                                               name=Schedule.riskImpacts.lower())
        
        # calculate cumDurations with/without risk
        self.cumDuration_norisk = self.get_cumDuration(self.duration_norisk)
        self.cumDuration_risk = self.get_cumDuration(self.duration_risk)
        
        # get Project Total Duration with/without risk
        self.totalDuration_norisk = self.get_totalDuration(self.cumDuration_norisk)
        self.totalDuration_risk = self.get_totalDuration(self.cumDuration_risk)
        
        # rewrite task duration for 'Project Duration' with/without risk
        self.duration_norisk[:,self.hammockIds] = self.totalDuration_norisk[:,None]
        self.cumDuration_norisk[:,self.hammockIds] = self.totalDuration_norisk[:,None]
        self.duration_risk[:,self.hammockIds] = self.totalDuration_risk[:,None]
        self.cumDuration_risk[:,self.hammockIds] = self.totalDuration_risk[:,None]
        
    def get_cumDuration(self, duration):
        "get cummulative duration as of the end of each task for each iteration"
        # initialization, column-wise(order='F') order in memory
        cumDuration = np.empty(duration.shape, order='F')
        
        for task in self.executeSeq:
            prerequisite = self.predecessor[task]
            if len(prerequisite)==0:
                cumDuration[:,task] = duration[:,task]
            elif len(prerequisite)==1:
                cumDuration[:,task] = cumDuration[:,prerequisite[0]]+duration[:,task]
            else:
                cumDuration[:,task] = np.max(cumDuration[:,prerequisite],axis=1) + \
                        duration[:,task]
        
        return cumDuration
        
    def get_totalDuration(self, cumDuration):
        "get Project Total Duration for each iteration from cumDuration matrix"
        if len(self.endIds) == 1:
            totalDuration = cumDuration[:,self.endIds[0]]
        elif len(self.endIds) > 1:
            totalDuration = np.max(cumDuration[:,self.endIds],axis=1)
        else:
            raise ValueError("endId not found for this projects")
        
        return totalDuration
        
    def get_criticalPath(self,cumDuration):
        "identify critical path for simulated duration, return bool ndarray"
        # initialization, default False, column-wise(order='F') order in memory
        criticalPath = np.zeros(cumDuration.shape, dtype=bool, order='F')
        row = np.arange(len(cumDuration))
        
        # assign value for hammockIds & endIds
        criticalPath[:,self.hammockIds] = True # set all hammock task as True
        if len(self.endIds) == 1:
            criticalPath[:,self.endIds[0]] = True
        else:
            col = np.argmax(cumDuration[:,self.endIds], axis=1)
            criticalPath[row,np.array(self.endIds)[col]] = True
        
        # iterate over executeSeq
        for task in self.executeSeq[::-1]:
            prerequisite = self.predecessor[task]
            if len(prerequisite)==0:
                pass
            elif len(prerequisite)==1:
                criticalPath[:,prerequisite[0]] |= criticalPath[:,task]
            else:
                col = np.argmax(cumDuration[:,prerequisite], axis=1)
                criticalPath[row,np.array(prerequisite)[col]] |= criticalPath[:,task]
                
        return criticalPath
        
    def plot_PDFCDF(self, var='totalDuration', figtype='PDF&CDF', withrisk=True):
        Cost.plot_PDFCDF(self,var,figtype,withrisk)
        
    def plot_performance(self, figtype='Tornado Diagram', withrisk=True):
        y = [f"{s.id}: {s.name}" for i,s in enumerate(self.tasks)]
        title = f"{self.title(withrisk)} {figtype}"
        if figtype=='Tornado Diagram':
            duration,totalDuration = self.get(['duration','totalDuration'],withrisk)
            mean,lows,highs = sensitivity1D(duration,totalDuration)
            width = np.subtract(highs,lows)
            title += f", {self.unit}"; xlabel = f'Average: {mean:,.2f} {self.unit}'
            plt_sortedbarh(y,width,lows,topN=10,vline=mean,xlabel=xlabel,title=title)
        elif figtype=='Critical Index':
            cri_Idx = self.get_criticalPath(self.get('cumDuration',withrisk)).mean(axis=0)
            txtfmt = "{x:.2%}" # using percent format for bar width text
            plt_sortedbarh(y,cri_Idx,txtfmt=txtfmt,title=title,xlim=[0,1],xfmt="{x:.0%}")
        elif figtype=='Correlation Coefficients':
            corr = np.corrcoef(self.get('duration',withrisk).T, 
                               self.get('totalDuration',withrisk))[-1,:-1]
            corr = np.where(np.isnan(corr), 0, corr)
            txtfmt = "{x:.3f}" # using 3-digit float format for bar width text
            plt_sortedbarh(y,corr,vline=0,topN=10,txtfmt=txtfmt,title=title,xlim=[-1,1])
        elif figtype=='Schedule Impact Indicators':
            cri_Idx = self.get_criticalPath(self.get('cumDuration',withrisk)).mean(axis=0)
            corr = np.corrcoef(self.get('duration',withrisk).T, 
                               self.get('totalDuration',withrisk))[-1,:-1]
            corr = np.where(np.isnan(corr), 0, corr)
            sii = cri_Idx * corr
            txtfmt = "{x:.3f}" # using 3-digit float format for bar width text
            plt_sortedbarh(y,corr,vline=0,topN=10,txtfmt=txtfmt,title=title,xlim=[-1,1])
            
    def _browseDuration(self, excelapi, withrisk=True):
        header = [['Task Id','Project Total Duration'] + self.ids, 
                  ['Iteration \ Task Name','Project Total Duration'] + \
                      [task.name for task in self.tasks]]
        duration,totalDuration = self.get(["duration","totalDuration"],withrisk)
        data = np.concatenate((totalDuration[:,None],duration),axis=1)
        excelapi.add_sheet('Duration', data, header, index=range(1,self.nIter+1))
        
    def browse(self, withrisk=True):
        sheets = ['Duration','Risk Events'] if withrisk else ['Duration'] # default sheets
        self._browse(sheets, withrisk)   


class ScheduleCost(Schedule):
    "Schedule/cost Integration"
    riskImpacts = ['Schedule','Cost'] # risk impacts applied, _fetch_params and simulate
    unit = Cost.unit # result unit
    fmt,title = Cost.fmt,Cost.title
    frequency = {'Monthly': (21,'Month','Months'),'Weekly': (5,'Week','Weeks')}
    def __init__(self,input_dir):
        super().__init__(input_dir)
        self._crosscompare_schedulecost()
        self._process_costParams()
        
    def _fetch_params(self, input_dir):
        "fetch parameters in 'Schedule','Cost','Risk Register' sheet in excel"
        task_raw, cost_raw, risk_raw = read_excel_raw(
            input_dir,sheets=['Schedule','Cost','Risk Register'],skiprows=[1,2,2])
        
        # defined function for reading schedule,risk paramters
        self.tasks = read_Task(task_raw, 'Schedule')
        self.costs, self.PMcost = read_Cost(cost_raw)
        self.risks = read_Risk(risk_raw, self.riskImpacts)
        
    def _crosscompare_schedulecost(self):
        "cross compare schedule and cost input for id and name"
        pass
    
    def _process_costParams(self):
        self.dailyCosts = [c.dailyCost for c in self.costs]
    
    def setup_integrationParams(self,freq='Monthly',customizedDaysPerPeriod=21):
        "receive integration params from GUI"
        if freq in self.frequency:
            self.daysPerPeriod,self.periodUnit,self.displaystr = self.frequency[freq]
        else:
            assert customizedDaysPerPeriod>0, 'Customized Days/Period should be positive!'
            self.daysPerPeriod,self.periodUnit = customizedDaysPerPeriod,'Custom Period'
            self.displaystr = f'Periods ({self.daysPerPeriod} days/period)'
    
    def simulate(self,nIter):
        Schedule.simulate(self,nIter)
        # simulate fixedCost without risk
        self.fixedCost_norisk = MCsimulate(nIter,[c.fixedCost for c in self.costs])
        
    def calculate_simresults(self):
        "calculate simulated duration/cumDuration/cumCost with/without risk"
        Schedule.calculate_simresults(self) # duration/cumDuration
        
        # applied costImpact to fixedCost
        self.fixedCost_risk = self.get_withrisk(self.fixedCost_norisk,self.costImpact)
        
        # convert cummulative cost
        self.cumCost_norisk = self.get_cumCost(
            self.duration_norisk,self.cumDuration_norisk,self.fixedCost_norisk)
        self.cumCost_risk = self.get_cumCost(
            self.duration_risk,self.cumDuration_risk,self.fixedCost_risk)
        
    def get_taskCost(self,duration,fixedCosts):
        taskcost = duration * self.dailyCosts
        for i,fixedCost in enumerate(fixedCosts):
            if fixedCost is not 0.: taskcost[:,i] += fixedCost
        return taskcost * (1+self.PMcost) if self.PMcost else taskcost
        
    def get_cumCost(self,duration,cumDuration,fixedCosts):
        '''calculte cummulative cost at each end of the period
        
        total cost = (variable cost + fixed cost) * (1 + PM cost)
        '''
        # change metric, from day to period, in duration matrix
        taskStart = cumDuration - duration
        
        totalDuration = self.get_totalDuration(cumDuration)
        maxPeriods = int(totalDuration.max()/self.daysPerPeriod) + 1
        endDays = self.daysPerPeriod * np.arange(1,maxPeriods+1)
        
        # calculate variable cost, variable cost = running duration * daily cost
        # running duration as of each period
        #     = min(duration,(asofdays-taskStartDate)^+)
        #     = min(cumDuration,max(asofdays,taskStartDate)) - taskStartDate
        # cumCost = np.array([np.matmul(
            # np.minimum(cumDuration, np.maximum(endDay,taskStart)),
            # self.dailyCosts) for endDay in endDays]).T
        # cumCost -= np.matmul(taskStart,self.dailyCosts)[:,None]
        # completedFlag = np.arange(maxPeriods)*self.daysPerPeriod>totalDuration[:,None]
        # cumCost[completedFlag] = np.nan
        tmp,cumCost = taskStart.copy(),[]
        for endDay in endDays:
            tmp[taskStart<endDay] = endDay
            _ = np.positive(cumDuration,out=tmp,where=cumDuration<endDay)
            cumCost.append(np.matmul(tmp,self.dailyCosts))
        cumCost = np.array(cumCost-np.matmul(taskStart,self.dailyCosts)).T
        
        # apply fixed cost and PM cost to variable cost
        TaskIdHasCosts = [i for i,a in enumerate(fixedCosts) if a is not 0.]
        if TaskIdHasCosts:
            row = np.arange(len(cumDuration))
            cost = np.zeros(cumCost.shape) # fixed cost at each period
            for taskId in TaskIdHasCosts:
                # fixed cost applied at task start date
                col = (taskStart[:,taskId] // self.daysPerPeriod).astype(int)
                cost[row,col] += fixedCosts[taskId]
            cumCost += cost.cumsum(axis=1)
        if self.PMcost: cumCost *= (1+self.PMcost)
        return cumCost
        
    def plot_cumCostStats(self, statsNames, withrisk=True):
        "plot periodic cumCost statistics as Time vs Stats"
        cumCost = self.get('cumCost',withrisk)
        periods = np.arange(1,cumCost.shape[1]+1)
        
        plt.figure()
        plt.plot(periods,get_stats(cumCost, statsNames).T)
        title = self.title(withrisk)+" vs Time"
        set_plotparams(plt.gca(),title,xlabel=self.displaystr,ylabel="Cumulative Cost",
                       yfmt=self.fmt,legend=statsNames,showfig=True)
        
    def _browseRiskEvents(self, excelapi, withrisk=True):
        header = [sum([[i,'',''] for i in self.riskIds],['Risk Id']),
                  sum([[r.name,'',''] for r in self.risks],['Risk Name']), ['Iteration']+\
                  ['Risk Frequency','Schedule Impact','Cost Impact']*self.nRisks]
        data = sum(zip(self.riskFreqs,self.scheduleImpact,self.costImpact),())
        excelapi.add_sheet('Risk Events', data, header, range(1,self.nIter+1), axis=1)
    
    def _browseCost(self, excelapi, withrisk=True):
        cumCost = self.get('cumCost',withrisk)
        header = ['Iteration \ '+self.displaystr] + list(range(1,cumCost.shape[-1]+1))
        excelapi.add_sheet('Cost', cumCost, header, index=range(1,self.nIter+1))
        
    def _browseCostStatistics(self, excelapi, withrisk=True):
        data = get_stats(self.get('cumCost',withrisk),self.statsNames)
        header = ['Statistics \ '+self.displaystr] + list(range(1,data.shape[-1]+1))
        excelapi.add_sheet('Cost Statistics', data, header, index=self.statsNames)
    
    def browse(self, sheets, statsNames, withrisk=True):
        self.statsNames = statsNames
        self._browse(sheets,withrisk)
        del self.statsNames
    
    def copy(self, to):
        "export Schedule Cost Integration obj data to Schedule or Cost obj"
        if to == "Schedule":
            DPobj = Schedule()
            DPobj.__dict__ = self.__dict__.copy() # same data exported to copy obj
        elif to == "Cost":
            DPobj = Cost()
            DPobj.__dict__ = self.__dict__.copy()
            # generate task variable for cost
            DPobj.tasks = [task._replace(dist=fetch_dist('N/A')) for task in self.tasks]
            
            # generate cost data according to task duration & daily cost & fixed cost
            DPobj.cost_norisk = DPobj.task_norisk = self.get_taskCost(
                self.duration_norisk,self.fixedCost_norisk)
            DPobj.cost_risk = self.get_taskCost(self.duration_risk,self.fixedCost_risk)
            DPobj.totalCost_norisk,DPobj.totalCost_risk = \
                DPobj.cost_norisk.sum(axis=1), DPobj.cost_risk.sum(axis=1)
        return DPobj


if __name__ == "__main__":
    from configs import HOME
    filedir = HOME+"\\Excel Templates\\Schedule & Cost Integration\\"+ \
              "Schedule & Cost Integration Example Problem.xlsx"
    nIter = 10000
    stats = ['Minimum','Maximum','Mean','50th Percentile']
    
    # test for schedule module
    schedule_obj = Schedule(filedir)
    schedule_obj.simulate(nIter)
    schedule_obj.calculate_simresults()
    
    schedule_obj.get_criticalPath(schedule_obj.cumDuration_norisk).sum(axis=0)
    schedule_obj.plot_performance(figtype='Tornado Diagram', withrisk=True)
    schedule_obj.plot_performance(figtype='Critical Index', withrisk=True)
    schedule_obj.plot_performance(figtype='Correlation Coeff.', withrisk=True)
    schedule_obj.plot_performance(figtype='Schedule Impact Indicator (SII)', 
                                  withrisk=True)
    
    # test for Schedule/Cost part
    schedulecost_obj = ScheduleCost(filedir)
    schedulecost_obj.simulate(nIter)
    schedulecost_obj.setup_integrationParams()
    schedulecost_obj.calculate_simresults()
    
    schedulecost_obj.plot_cumCostStats(stats,True)