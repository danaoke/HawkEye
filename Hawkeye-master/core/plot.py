#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import PchipInterpolator


__all__ = ['set_plotparams','plt_pdf','plt_hist','plt_cdf','plt_pdfcdf','plt_sortedbarh',
           'plt_step','plt_PCHIP']


def set_plotparams(ax,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,
                   xfmt=None,yfmt=None,legend=None,savefig=None,showfig=False):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    for lim,s in [(xlim,'xlim'),(ylim,'ylim')]:
        if isinstance(lim,(list,tuple)):
            left,right = getattr(ax,'get_'+s)()
            if lim[0] is not None: left = max(left,lim[0])
            if lim[1] is not None: right = min(right,lim[1])
            getattr(ax,'set_'+s)(left,right)
    for axis,fmt in [(ax.xaxis,xfmt),(ax.yaxis,yfmt)]:
        if isinstance(fmt,str):
            axis.set_major_formatter(ticker.StrMethodFormatter(fmt))
        elif isinstance(fmt,ticker.Formatter):
            axis.set_major_formatter(fmt)
        elif hasattr(fmt,'__call__'):
            axis.set_major_formatter(ticker.FuncFormatter(fmt))
        
    if legend: ax.legend(legend)
    if savefig: plt.savefig(savefig)
    if showfig: plt.show()
    

def plt_pdf(dist,ax=None,fill=True,color='tab:blue',lw=2,**kwargs):
    "plot theoratical pdf using scipy.stats.dist.pdf(x)"
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    xs = np.linspace(*dist.ppf([1e-5,1-1e-5]),1000)
    pdfs = dist.pdf(xs)
    ax.plot(xs,pdfs,color=color,lw=lw,label="PDF")
    ax.set_ylabel("Probability Density")
    if fill: ax.fill_between(xs,0,pdfs,color='tab:green',alpha=0.6)
    ax.set_ylim(0.)
    set_plotparams(ax,**kwargs)
    
    
def plt_hist(data,ax=None,bins=50,color='tab:blue',alpha=1,**kwargs):
    "plot histogram using simulated data"
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    _,bins,_=ax.hist(data,bins=bins,color=color,alpha=alpha,edgecolor='black',label="PDF")
    ax.set_ylabel("Relative Frequency")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))
    set_plotparams(ax,**kwargs)
    return bins
    
    
def plt_cdf(data_dist,ax=None,color='tab:orange',lw=2,**kwargs):
    "plot cdf curve using simulated data or dist"
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    if hasattr(data_dist,'cdf'):
        # plot line graph using thoeratical cdf
        xs = np.linspace(*data_dist.ppf([1e-5,1-1e-5]),1000)
        cdfs = data_dist.cdf(xs)
        ax.plot(xs,cdfs,color=color,lw=lw,label="CDF")
    elif isinstance(data_dist,(list,np.ndarray)):
        # plot step wise graph using simulated data
        xs = np.sort(data_dist)
        cdfs = np.linspace(0,1,len(data_dist),endpoint=False)
        ax.step(xs,cdfs,where='post',color=color,lw=lw,label="CDF")
    else:
        raise ValueError("argument data_dist either be simulated data or dist obj")
    ax.set_ylim(0.,1.)
    ax.set_ylabel("Cumulative Distribution")
    set_plotparams(ax,**kwargs)
    
    
def plt_pdfcdf(data_dist,ax=None,**kwargs):
    "plot pdf bars and cdf curve using simulated data or dist"
    assert type(data_dist).__name__ in ('list','ndarray','rv_frozen'), \
        "argument data_dist either be simulated data array or distribution obj"
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    pdf_ax,cdf_ax = ax,ax.twinx()
    # plot pdf & cdf
    (plt_pdf if hasattr(data_dist,'pdf') else plt_hist)(data_dist,ax=pdf_ax)
    plt_cdf(data_dist,ax=cdf_ax)
    # plot legend handles
    handles = sum([x.get_legend_handles_labels()[0] for x in [pdf_ax,cdf_ax]],[])
    pdf_ax.legend(handles=handles,loc='best')
    set_plotparams(pdf_ax,**kwargs)
    
    
def plt_sortedbarh(y,width,left=0,ax=None,topN=0,vline=None,shrink=2/3,txtfmt="{x:,.2f}",
                   positive_c='tab:blue',negative_c='tab:red',alpha=0.6,**kwargs):
    "plot top N horizontal bar in descending order"
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    # create np.array, switch left/right if width<0 and sorted by width
    dtype = [('y',object),('width',float),('left',float),('right',float)]
    data = np.empty_like(y,dtype=dtype)
    data['y'] = y; data['width'] = width; 
    data['left'] = left + np.minimum(width,0);data['right'] = left + np.maximum(width,0)
    data = data[np.argsort(abs(data['width']))]
    if isinstance(topN,int): data = data[-topN:] # data with top N largest width
    
    # horizontal bar plot, vertical line, reset xlabel position
    colors = [positive_c if width >=0 else negative_c for width in data['width']]
    ax.barh(data['y'],np.abs(data['width']),left=data['left'],color=colors,alpha=alpha)
    if vline is not None: 
        ax.axvline(x=vline, color='black') 
        ax.xaxis.set_label_coords(vline,0,transform=
                                  ax.xaxis.get_ticklabels()[0].get_transform())
    
    # shrink xaxis to occupy 2/3 of the graph, add width text
    xlim = [data['left'].min(),data['right'].max()] # original xlim
    extra_width = (xlim[1] - xlim[0]) * (1/shrink-1) # extract width added to graph
    ax.set_xlim(left=xlim[0]-extra_width*.4, right=xlim[1]+extra_width*.6) # reset xlim
    funcFmt = (lambda x: txtfmt.format(x=x)) if isinstance(txtfmt,str) else txtfmt
    for y,(width,right) in enumerate(data[['width','right']]):
        ax.text(right+extra_width/10,y,funcFmt(width),va='center')
        
    # remove graph border and move x axis to top
    [ax.spines[pos].set_visible(False) for pos in ('left','right','bottom')]
    ax.xaxis.set_ticks_position('top')
    ax.tick_params('y', length=0)#, pad=0) # remove tick
    set_plotparams(ax,**kwargs) # pass **kwargs


def plt_step(x,y,ax=None,where='pre',color=None,lw=2,**kwargs):
    "plot stepwise figure"
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    ax.step(x,y,where=where,color=color,lw=lw)
    set_plotparams(ax,**kwargs)


def plt_PCHIP(x,y,ax=None,color=None,lw=2,**kwargs):
    '''
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) plot
    reference: 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.
PchipInterpolator.html
    '''
    if ax is None: _,ax = plt.subplots(); kwargs.setdefault('showfig',True)
    # interpolation
    curve = PchipInterpolator(x, y)
    x_plot = np.linspace(x[0],x[-1],1000)
    y_plot = curve(x_plot)
        
    ax.plot(x_plot,y_plot,color=color,lw=lw)
    set_plotparams(ax,**kwargs)
    return x_plot,y_plot


if __name__ == "__main__":
    from dist import fetch_dist
    # PDF, Histogram, CDF Plottings Examples
    title = "PDF Plot Example using dist object"
    xlabel="Task Duration, days"
    plt_pdf(fetch_dist('Pert',12,15,20),xlabel=xlabel,title=title)
    
    data = fetch_dist('Pert',2,5,7).rvs(10000)*fetch_dist('Bernoulli',0.6).rvs(10000)
    title = "Histogram Plot Example using Simulated numpy.ndarray Data"
    plt_hist(data,title=title,xlabel=xlabel)
    
    title = "Multiple Histogram Plots Example"
    _,ax = plt.subplots()
    bins = plt_hist(fetch_dist('Pert',0,10,20).rvs(10000),ax=ax,color='tab:red',alpha=0.5)
    plt_hist(fetch_dist('Pert',5,10,15).rvs(10000),ax=ax,bins=bins,alpha=0.5,title=title,
             xlabel=xlabel,legend=['with risk','without risk',],showfig=True)
    
    title = "CDF Plot Example using dist object"
    plt_cdf(fetch_dist('Pert',2,5,7),title=title,xlabel=xlabel)
    
    title = "CDF Plot Example using Simulated numpy.ndarray Data"
    plt_cdf(data,title=title,xlabel=xlabel)
    
    title = "PDF & CDF Plot Example using dist object"
    plt_pdfcdf(fetch_dist('Normal',2,4.5,7),title=title,xlabel=xlabel)
    
    title = "PDF & CDF Plot Example using Simulated numpy.ndarray Data"
    plt_pdfcdf(fetch_dist('Normal',2,4.5,7).rvs(10000),title=title,xlabel=xlabel)
    
    title = "x/y axis ticker format example"
    xfmt = '${x:,.0f}'; yfmt = ticker.PercentFormatter(xmax=1)
    dist = fetch_dist('Normal',2000,4500,7000)
    plt_pdfcdf(dist,title=title,xlabel="Project Cost",xfmt=xfmt,yfmt=yfmt)
    
    title = "Multiple lines in 1 graph"
    _,ax = plt.subplots()
    plt_cdf(fetch_dist('Normal',12,15,20),ax=ax,color='tab:blue')
    plt_cdf(fetch_dist('Pert',12,15,20),ax=ax,color='tab:red',title=title,xlabel=xlabel,
            legend=['without risk','with risk'],showfig=True)
    
    # Tornado, Critical Index, Correlation Coeff. Examples
    y = [f'Task Name {i}' for i in range(21)]
    basevalue = 0.5
    lows = np.random.rand(21)
    highs = np.random.rand(21)
    title = "Tornado Diagram Example, Top 10 Impact"
    xlabel = f'Average: {basevalue:,.2f} days'
    plt_sortedbarh(y,highs-lows,lows,vline=basevalue,topN=10,xlabel=xlabel,title=title)
    
    criticalindex = np.random.rand(21); criticalindex[0:2] = [0,1]
    title = "Critical Index Example"
    txtfmt = "{x:.2%}" # using percent format for bar width text
    xlim = [0,1]; xfmt = ticker.PercentFormatter(xmax=1)
    plt_sortedbarh(y,criticalindex,txtfmt=txtfmt,title=title,xlim=xlim,xfmt=xfmt)
    
    corr = 1 - 2*np.random.rand(21)
    title = "Correlation Coeff. Example"
    txtfmt = "{x:.2f}" # using percent format for bar width text
    xlim = [-1,1]
    plt_sortedbarh(y,corr,vline=0,txtfmt=txtfmt,title=title,xlim=xlim)
