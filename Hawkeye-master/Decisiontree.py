#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:12:52 2020

@author: jiahexu
"""

import numpy as np
from collections import defaultdict,namedtuple
from core import *
import networkx as nx 
import matplotlib.pyplot as plt


__all__ = ['DecisionTree']


TaskDecision = namedtuple('TaskDecision',["id","name","ph","pl","dist","sucess","failue","cost","suce","fai"])

def read_Decisiontree(decision_raw):
    
    msg = lambda Id,name: f" TaskId={Id},TaskName='{name}' in 'DecisionTree'"
    result = []
    for row in decision_raw:
        # skip row that row[0] == ""
        if row[0] == "": continue

        # process task Id and name
        Id = int(row[0])
        name = row[1]
        cost=row[4]
        suce=row[9]
        fai=row[14]
        ph=row[6]
        pl=row[11]
        # process&validate distribution columns
        try:
            dist = fetch_dist(*row[2:6])
        except Exception as e:
            raise ValueError(str(e)+msg(Id,name))
        try:
            sucess = fetch_dist(*row[7:11])
        except Exception as e:
            raise ValueError(str(e)+msg(Id,name))
        try:
            failue = fetch_dist(*row[12:16])
        except Exception as e:
            raise ValueError(str(e)+msg(Id,name))

        result.append(TaskDecision(Id,name,ph,pl,dist,sucess,failue,cost,suce,fai))

    return result

def binomial_grid(n,earn,cost,name,ph,pl,title):
    n=n+2
    G=nx.Graph() 
    edge_labelshigh = {}
    edge_labelslow = {}
    node_labels={}
    node_labelslow={}
    node_labelshigh={}
    for i in range(1,n+1): 
        j=2
        z=3 
        if i<n:              
            G.add_edge((i,j),(i+1,j))
            G.add_edge((i,j),(i+1,z))
           
            edge_labelshigh[((i,j),(i+1,j))] = ph[i-1]
            edge_labelslow[((i,j),(i+1,z))] = pl[i-1]
            node_labelslow[(i+1,j)]=earn[i][0]
            node_labelslow[(i+1,z)]=earn[i][1]
            node_labelshigh[(i+1,j)]=cost[i][0]
            node_labelshigh[(i+1,z)]=cost[i][1]
            node_labels[(i+1,j)]=name[i][0]
            node_labels[(i+1,z)]=name[i][1]
    node_labelshigh[(1,2)]=cost[0][0]
    node_labelslow[(1,2)]=earn[0][0]   
    node_labels[(1,2)]=name[0][0]   
    posG = {}

    deci_map = []
    chance_map=[]
    output_map=[]
    for node in G.nodes():
        if node[0]%2==0 and node[1]==2:
            chance_map.append((node[0],node[1]))
        elif node[0]%2!=0 and node[1]==2 and node[0]!=n:
            deci_map.append((node[0],node[1]))
            
        
        else:
            output_map.append((node[0],node[1]))
        
        posG[node]=(node[0],node[0]-2*node[1])
    
    pos_higher = {}
    pos_lower = {}
    pos_right = {}
    y_off = 0.5  # offset on the y axis
    for k, v in posG.items():
        pos_higher[k] = (v[0], v[1]+y_off)
        pos_lower[k]= (v[0], v[1]-y_off)
        pos_right[k] = (v[0]+(n)/200, v[1])
    fixed_nodes = posG.keys()
    
    plt.figure(figsize=(2*n-1,2*n-1))
    plt.title(title,fontsize=20)
    nx.draw_networkx_edges(G,pos=posG)
    nx.draw_networkx_labels(G,pos=posG,labels=node_labels)
    nx.draw_networkx_labels(G,pos=pos_lower,labels=node_labelslow)
    nx.draw_networkx_labels(G,pos=pos_higher,labels=node_labelshigh)
    nx.draw_networkx_edge_labels(G,pos=pos_higher,edge_labels=edge_labelshigh,font_size=12)
    nx.draw_networkx_edge_labels(G,pos=pos_lower,edge_labels=edge_labelslow,font_size=12)
    nx.draw_networkx_nodes(G,pos=posG,nodelist=deci_map,node_shape="s",node_color='green',label='Decision Node')
    nx.draw_networkx_nodes(G,pos=posG,nodelist=chance_map,node_shape="o",node_color='red',label='Chance Node')
    nx.draw_networkx_nodes(G,pos=pos_right,nodelist=output_map,node_shape="<",node_color='blue',label='End of Branch')
#     pyplot.xlim((0.5,6))
    #plt.ylim((-6,3))
    
    plt.legend(fontsize=15,markerscale=1)
    plt.show()

  
class DecisionTree(object):
    '''
    Data processing part for decisiontree Module,

     Functionality:
     1. read decisiontree parameters from inputs file
     2. simulate randomness using Module functions in core.MonteCarloSimulation
     3. calculate simulation results 
     4. draw decisiontree graph
    '''
    fmt = lambda _,x,*args: f"-${-x:,.0f}" if x<0 else f"${x:,.0f}" # o-stream format
    def __init__(self,input_dir):
        "setup nodeIds, distributions, relevant node"
        self._fetch_params(input_dir)
        
    def _fetch_params(self, input_dir):
        " fetch parameters in excel"
        decision_raw=read_excel_raw(input_dir,sheets=0,skiprows=2,skipcols=0)
        self.decision=read_Decisiontree(decision_raw)
        self.ph=[s.ph for s in self.decision]
        self.pl=[s.pl for s in self.decision]
        self.name=[s.name for s in self.decision]
        self.id=[s.id for s in self.decision]
        self.distribution=[t.dist for t in self.decision]
        
    def simulate(self,nIter):
        "simulate Decision Node Cost&Sucess&failure data"
       
        self.simcost_value = np.array(
            MCsimulate(nIter, [t.dist for t in self.decision]))
        self.simsucceed_value = np.array(
            MCsimulate(nIter, [t.sucess for t in self.decision]))
        self.simfail_value = np.array(
            MCsimulate(nIter, [t.failue for t in self.decision]))
        self.output_value=np.concatenate((self.simcost_value,self.simsucceed_value,self.simfail_value),axis=0).tolist()
        #return self.simcost_value
        
    
    def simul_calculate(self):
        "Calculate Simulated Results Expected Value"
        cost_value=np.copy(self.simcost_value)
        sucess_value=np.copy(self.simsucceed_value)
        failue_value= np.copy(self.simfail_value)
        d_nod=np.zeros(cost_value.shape)
        
        for i in reversed(range(len(self.ph))):
            d_nod[i]=sucess_value[i]*self.ph[i]+failue_value[i]*self.pl[i]
            sucess_value[i-1]=sucess_value[i-1]+d_nod[i]-cost_value[i]
        return d_nod[0]-cost_value[0]
        
    def value_calculate(self,simulate,opera):  
        "Prepare deterministic&Simulated data for plot"
        fmt = lambda x: f"{x*100:.0f}%" 
        nodset=[[0,0]]
        Expectvalue=[]
        vh=[1]+self.ph.copy()
        vl=self.pl.copy()
        if simulate=='n':
            self.cost_value=[s.cost for s in self.decision]
            self.succeed_value=[s.suce for s in self.decision]
            self.fail_value=[s.fai for s in self.decision]

        if simulate=='w':
            self.cost_value= np.average(self.simcost_value,axis=1)
            self.succeed_value=np.average(self.simsucceed_value,axis=1)
            self.fail_value=np.average(self.simfail_value,axis=1)

        for i in range(len(self.ph)):
            if opera=='w':
                v=nodset[-1]
            elif opera=='n':
                v=nodset[0]
            vl[i]=vl[i]*vh[i]
            vh[i+1]=vh[i+1]*vh[i]
            nodset.append([v[0]-self.cost_value[i],v[0]])
            nodset.append([v[0]-self.cost_value[i]+self.succeed_value[i],v[0]-self.cost_value[i]+self.fail_value[i]])
        nodset[-2][0]=nodset[-1][0]*self.ph[-1]+nodset[-1][1]*self.pl[-1]   
        for i in range(len(self.ph)):
            Expectvalue.append(nodset[2*i+1])
            vh[i+1]=fmt(vh[i+1])
            vl[i]=fmt(vl[i])
        for i in reversed( range(len(Expectvalue)-1)):
            if opera=='w':Expectvalue[i][0]=Expectvalue[i+1][0]*self.ph[i]+Expectvalue[i+1][1]*self.pl[i]
            elif opera=='n':Expectvalue[i][0]=Expectvalue[i+1][0]*self.ph[i]+Expectvalue[i+1][1]*self.pl[i]-self.cost_value[i]
        vh[0]=str()
        vl=['']+vl

        return Expectvalue,nodset,vh,vl
    
    def graphprepare(self,simulate,opera,minflag):
        def compare(x,y,z):return (x>y) and z
        Expectvalue,lower_node,vh,vl = self.value_calculate(simulate,opera)
        if simulate=='w'and opera=='n' :
            title="Simulation Results without Sunk Cost"
        elif simulate=='w'and opera=='w' :
            title="Simulation Results with Sunk Cost"
        elif simulate=='n'and opera=='n' :
            title="Deterministic Results without Sunk Cost"
        elif simulate=='n'and opera=='w' :
            title="Deterministic Results with Sunk Cost"
    
        upper_node=[[''] * len(self.cost_value) for _ in range(2*len(self.cost_value)+1)]
        middle_node=list(list('' for c in range(len(self.cost_value))) for w in range(2*len(self.cost_value)+1))
        
        fmt = lambda x: f"-${-x:.0f}" if x<0 else f"${x:.0f}"
        fmt1= lambda x: f"{x:.0f}" 
        lower_node[-1][0]= fmt(lower_node[-1][0])+'\n'+str(vh[-1])
        lower_node[-1][1]= fmt(lower_node[-1][1])+'\n'+str(vl[-1])
        ph=list(str() for c in range(2*len(self.cost_value)))
        pl=list(str() for c in range(2*len(self.cost_value)))
        
        for i in range(len(self.cost_value)):
            ph[2*i+1]='Success\n'+'p='+fmt1(self.ph[i])
            pl[2*i+1]='Failure\n'+'p='+fmt1(self.pl[i])
            ph[2*i]='EV='+fmt(Expectvalue[i][0])
            pl[2*i]='EV='+fmt(Expectvalue[i][1])
            if compare(Expectvalue[i][0],Expectvalue[i][1],minflag):
                ph[2*i]='True'+'\n'+ph[2*i] 
            else: pl[2*i]='True'+'\n'+pl[2*i]
            #upper_node[2*i][0]=str(compare(Expectvalue[i][0],Expectvalue[i][1],minflag))
            lower_node[2*i][0]=str(self.name[i])+'\n'+str(vh[i])
            lower_node[2*i][1]=fmt(lower_node[2*i][1] )+'\n'+str(vl[i])
            lower_node[2*i+1][0]=''
            lower_node[2*i+1][1]=''
            middle_node[2*i][0]='D'+str(i+1)
            upper_node[2*i+1][0]='Cost='+fmt(-self.cost_value[i])
        
        binomial_grid(2*len(self.name)-1,lower_node,upper_node,middle_node,ph,pl,title)
    
    def pdf_button(self,number):
        
        plt_pdf(self.distribution[number-1],xlabel="Cost",xfmt=self.fmt,title=str(self.name[number-1])+'; Distribution: '+self.distribution[number-1].__name__)
    
    def pdfcdf_button(self,dataset,title,xlabel):
       
        plt_pdfcdf(dataset,title=title,xlabel=xlabel,xfmt=self.fmt)
        
    
    def export(self,simuldata):
        excelapi = excel_api(mode='w')
        expectvalue = [simuldata]
        sus_name=list('' for c in range(len(self.name)))
        fail_name=list('' for c in range(len(self.name)))
        for i in range(len(self.name)):
            sus_name[i]='sucess_'+self.name[i]
            fail_name[i]='fail_'+self.name[i]
        final_name=self.name+sus_name+fail_name
        
        nIter = len(self.simul_calculate())
        header = [['Decision Node No.','Expected Value'],
                  ['Iteration \ Task Name','Expected Value'] + final_name]
        data=expectvalue+self.output_value
        excelapi.add_sheet('Decision_tree', data, header, index=range(1,nIter+1),axis=1)
        
        excelapi.displayExcelApp()

if __name__ == "__main__":
    #obj = DecisionTree('/Users/jiahexu/Downloads/hawkeyecode/decision_treetemp.xlsx')
    from configs import HOME
    obj = DecisionTree(HOME+"\\Excel Templates\\Decision Tree\\Decision Tree.xlsx")

    simcost_value=obj.simulate(1000)
    #result=obj.value_calculate('w','w')
    obj.simul_calculate()
    #obj.pdf_button(1)
    #obj.pdfcdf_button(expectvalue,title="Expected Value",xlabel="Expected Value")
    #obj.plt_distr(1)
    #obj.graphprepare('n','n',True)
    #bj.graphprepare('w','n',True)
    obj.graphprepare('n','w',True)
    #obj.graphprepare('w','w',True)
    #obj.export()