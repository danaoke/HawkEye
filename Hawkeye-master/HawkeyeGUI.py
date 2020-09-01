#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys,os,time
import tkinter as tk
from tkinter import ttk,messagebox,filedialog
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import configs
from configs import *
from helper import *

from core import *
from Sensitivity import *
from NPV import *
from ScheduleCost import *
from Optimization import *
from DecisionTree import *

# for developing Module_GUI class importing only
    # "from Hawkeye import *" when developing Module_GUI class to get same config setting
__all__ = [
    'tk','ttk','messagebox','filedialog','Module_GUI','App','make_tkTreeview',
    'get_statstable','tkValid','PercentVar','PositiveIntVar','IdVar','ListVar',
    'StatsVar'] + configs.__all__ + ['Cost_GUI','Schedule_GUI','ScheduleCost_GUI']


# ------------- GUI Modules params -------------
# dict for mapping moduleName to the name of classes defined for this module
# if classes is str: 
#     module class code is written follow new format inherated from the Module_GUI 
#     and 'Module', module obj store all frames of module and also the data.
# elif classes is tuple of the name of frames class
#     frames class code is written followed previous code format, only constains 
#     one frame for each frame class
GUI_modules = {
    "Sensitivity Analysis": "Sensitivity_GUI",
    "Project Net Present Value": "NPV_GUI",
    "Project Cost": "Cost_GUI",
    "Project Schedule": "Schedule_GUI",
    "Schedule & Cost Integration": "ScheduleCost_GUI",
    "Decision Tree": "DecisionTree_GUI",
    "Portfolio Optimization": "Optimization_GUI",
}


#######################################################################
# support functions/classes for Hawkeye GUI
time_metric = ['s','ms','µs','ns']
def sec2str(sec):
    "convert second(float) to str based on metrics"
    i = 0
    while sec<1 and i<3:
        i += 1
        sec *= 1000
    return f"{sec:7.3f} {time_metric[i]}"


def make_tkTreeview(table, title, show='tree', header=None, window=None,
                    font=TABLE_FONT, anchor='w'):
    "Add tkTreeview to window, including title, header, table body"
    # create new windom if not specified, write title 
    if not window:
        window = tk.Toplevel()
        window.resizable(width=True, height=True)
        window.title(title) # title at window border
    tk.Label(window, text=title, font=font['title']).pack() # add title
    
    # modify Treeview style: fonts of body & heading and remove boreder
    style = ttk.Style()
    style.configure("mystyle.Treeview.Heading", font=font['header'])
    style.configure("mystyle.Treeview", font=font['body'], rowheight=40)
    style.layout("mystyle.Treeview",[('mystyle.Treeview.treearea',{})])
    
    # initialize tree and insert columns, col width, header
    tree = ttk.Treeview(window, show=show, style="mystyle.Treeview")
    tree.pack(fill='both',expand=1)
    colN = len(table[0])
    tree["columns"] = list(range(colN))
    tree.column("#0", stretch='yes', minwidth=20, width=100)
    _ = [tree.column(col, stretch='yes', anchor=anchor, minwidth=100, width=200) 
         for col in range(colN)]
    # insert header if specified
    if show=='headings' and header is not None:
        _ = [tree.heading(i, text=text) for i,text in zip(range(colN),header)]
    
    # insert table body
    _ = [tree.insert('','end',values=v) for v in table]
    return tree, window


def get_statstable(nparr, title, fmt="$ {x:,.2f}", header=None):
    """
    Display statistic table to user, with args table title and data display format

    Examples:
    fmt="$ {x:,.2f}" #(default) display data as $ XXX,XXX,XXX.XX
    fmt="{x:,.0f} days" # display data as XXX,XXX days
    fmt=lambda x: f"-$ {-x:,.2f}" if x<0 else f"$ {x:,.2f}" # -$ XXX.XX when is negative
    """
    # process fmt to function
    funcFmt = (lambda x: fmt.format(x=x)) if isinstance(fmt,str) else fmt
    assert hasattr(funcFmt,'__call__'), \
        "arg fmt expected to be strFormater or strFunctionFormater,"
    
    # calculate statistics and formatting, make Treeview by calling make_tkTreeview
    stats = ['Minimum','Maximum','Mean','Median','Std. Deviation','Skewness','Kurtosis']
    stats += [f'{p}th Percentile' for p in [5,10,25,50,75,90,95]]
    statsValues = get_stats(nparr,stats)
    if statsValues.ndim == 1:
        vStr = list(map(funcFmt,statsValues))
        vStr[5:7] = [f"{statsValues[5]:.6f}", f"{statsValues[6]:.6f}"]
        table = list(zip(stats[:7],vStr[:7],stats[7:],vStr[7:]))
        make_tkTreeview(table, title)
    elif statsValues.ndim == 2:
        vStr = [[funcFmt(x) for x in xs] for xs in statsValues]
        vStr[5:7] = [[f"{sv:.6f}" for sv in l] for l in statsValues[5:7]]
        table = list(zip(stats,*list(zip(*vStr))))
        make_tkTreeview(table, title, show='headings', header=header, anchor='center')


def tkValid(Varname,validFunc=None,msg=''):
    "decorator to return original tkVariable with validator"
    tkVar = getattr(tk,Varname+'Var')
    class ValidVar(tkVar):
        def __init__(self,master=None,value=None):
            super().__init__(master,value)
            self._var = tkVar(master,value)
            self._value = value or tkVar._default
            self.validFunc,self.msg = validFunc,msg
            self.widges = [] # tk widges listed in current variables
            
        def update_valid(self,validFunc=None,msg=''):
            self.validFunc = validFunc or self.validFunc; self.msg = msg or self.msg
            
        def isvalid(self,newvalue):
            try:
                _value = self._var.get()
                if self.validFunc: assert self.validFunc(_value), self.msg
                if self._value != _value:
                    self._value = _value; # store new value for 
                    self._var.set(_value); self.set(_value) # store new value
                return True
            except Exception as e:
                self._var.set(self._value) # revert to previous value
                messagebox.showinfo("Invalid Entry!",f"Invalid Entry '{newvalue}'\n{e}")
                return False
        
        def Entry(self,frame,validFunc=None,msg='',width=10,**kwargs):
            self.inputMethod = 'Entry'
            self.update_valid(validFunc,msg)
            entry = ttk.Entry(frame,textvariable=self._var,width=width,**kwargs)
            entry['validate'] = 'focusout'
            entry['validatecommand'] = entry.register(self.isvalid), '%P'
            self.widges.append(entry)
            return entry
    return ValidVar


PercentVar = tkValid('Double', validFunc=lambda x:x>=0, 
                     msg="Expected Positive Percentage Value!")
PositiveIntVar = tkValid('Int',validFunc=lambda x:x>0, msg="Expected Positive Integer!")
class IdVar(tkValid('Int')):
    def Combobox(self,frame,text2id=None,width=30,state="readonly",**kwargs):
        self.inputMethod = 'Combobox'
        self.input = tk.StringVar(self._root)
        self.combobox = ttk.Combobox(frame,textvariable=self.input,width=width,state=state)
        if text2id: self.add_ComboboxContent(text2id)
        self.widges.append(self.combobox)
        return self.combobox
    
    def add_ComboboxContent(self,text2id):
        text2id = dict(text2id)
        self.combobox['values'] = list(text2id)
        self.input.trace('w',lambda *args: self.set(text2id[self.input.get()]))
        if text2id: self.combobox.current(0)
#     def get(self):
#         user_input = self.input.get()
#         if user_input not in self._inputsmapper:
#             msg = "Allowed ID: " + ','.join(map(str,self._inputsmapper.key())) \
#                 if self.inputMethod == 'Entry' else "Select a task from the drop down list"
#             raise ValueError(f"Invalid Entry '{user_input}'!\n"+msg)
#         return self._inputsmapper[user_input]
    
    def Label(self,frame,width=12,relief='ridge',**kwargs):
        self.label = tk.StringVar(self._root)
        label = tk.Label(frame,textvariable=self.label,width=width,relief=relief,**kwargs)
        self.widges.append(label)
        return label
    
    def Button(self,frame,text,command,**kwargs):
        button = ttk.Button(frame,text=text,command=command,**kwargs)
        self.widges.append(button)
        return button


class ListVar(object):
    def __init__(self,root,values,default=None):
        self.root = root
        self.values = values
        default = default or [True] * len(values)
        self.BoolVars = [tk.BooleanVar(root,b) for b in default]

    def Radiobuttons(frame_format,texts,variable,values=None,method="pack",padx=2,pady=0,
                     side="left",row=0,colN=2,**kwargs):
        frame = tk.Frame(frame_format,**kwargs)
        values = values or texts
        btns = [ttk.Radiobutton(frame,text=text,variable=variable,value=value) 
                for text,value in zip(texts,values)]
        _ = [btn.pack(side=side,padx=padx,pady=pady) if method=='pack' else 
             btn.grid(row=i//colN+row,column=i%colN,padx=padx,pady=pady,sticky='w')
             for i,btn in enumerate(btns)]
        return frame
        
    def CheckButtons(self,frame_format,texts=None,side="left",padx=2,pady=0,**kwargs):
        frame = tk.Frame(frame_format,**kwargs)
        texts = texts or self.values
        self.buttons = [ttk.Checkbutton(frame,text=text,variable=var)
                        for text,var in zip(texts,self.BoolVars)]
        [button.pack(side=side,padx=padx,pady=pady) for button in self.buttons]
        return frame
    
    def get(self):
        return [v for v,b in zip(self.values,self.BoolVars) if b.get()]


class StatsVar(ListVar):
    def __init__(self,root,values,default=None):
        ListVar.__init__(self,root,values,default)
        # extract percentiles
        tmp = self.values[-1][self.values[-1].index(':')+1:]
        self.percentileVar = tk.StringVar(self.root,tmp)
        
    def CheckButtons(self,frame_format,texts=None,padx=2,**kwargs):
        frame = ListVar.CheckButtons(self,frame_format,texts,padx=padx,**kwargs)
        # create another Entry for accept user defined Percentiles
        ttk.Entry(frame,textvariable=self.percentileVar, width=11).pack(side="left")
        return frame
        
    def get(self):
        stats = [v for v,b in zip(self.values[:-1],self.BoolVars[:-1]) if b.get()]
        if not self.BoolVars[-1].get(): return stats
        try:
            percentiles = list(map(int,self.percentileVar.get().strip().split(',')))
        except:
            raise ValueError("Invalid Entry 'Percentiles'!\n"\
                             "expect integers(0-99) separated by ','")
        return stats + [f'{q}th Percentile' for q in percentiles]


class Module_GUI(object):
    '''
    GUI parent class for each module,
    
    Functionality:
    1. func _intialized_params/clear_frameParams for initial/delete class attribute
    2. func runtime_updatestatus to update/display status at run time
    3. func tkraise to raise module: will raise first page in self.frames
    4. func pageframe_constructer to get default page&frames, support status feature
    5. func gridframe_constructer to add subtitles,first col,seperator to existed frame
    6. func _construct_filePage_frames for construct default file page
    7. decorator showerror for convert error to messagebox
    8. decorator check_requisites(*requisites) to check requisites for button
    9. decorator show_status(timeit=False) to display status before/after execution
    10. button functions, Back/Next
    '''
    # ------------- Module configs -------------
    __name__ = 'Module Name' # Module name, for display purpose in frames
    frames = ['filePage','frameName2'] # self.tkraise will raise the first frames
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'File Name Displayed at filePage' # fileName displayed in filePage
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {'filePage': [], 'pageName': ['param1Name','param2Name']}
    # default params value, used in initialization and clear value
    defaultValue = {'ParamName': 'paramDefaultValue'}
    # tk.Variables for Module, used in initialization only, won't auto clear value
    tkVariable = {'status': (tk.StringVar, ''), 'input_dir': (tk.StringVar, '')}
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['input_dir'],'w','funcNametoCallback')] # ModeName(w/r/u)
    # requisites dict, used in decorator self.check_requisites
    requisites = {'varName': ('satisfiedValue','errorMessage')}
    
    def __init__(self,frame_format,root,**testkwargs):
        self.root = root
        self._intialized_params()
        if testkwargs:
            # test mode for Module_GUI, will display default file page with border
            self._construct_filePage_frames(frame_format,testMode=True)
        
    def _intialized_params(self):
        "initialize default value for parameters and tkVariables"
        self.__dict__.update(self.defaultValue)
        for varName, (tkVarFunc, *defaultValue) in self.tkVariable.items():
            self.__dict__[varName] = tkVarFunc(self.root,*defaultValue)
        # if 'var2Trace' in self.__class__.__dict__:
        for varNames, mode, callbackfunc in self.var2Trace:
            callbackfunc = getattr(self,callbackfunc)
            [self.__dict__[varName].trace(mode,callbackfunc) for varName in varNames]
            
    def funcNametoCallback(self,*args):
        messagebox.showinfo("Template!","Temp function is called when var value changed")
        
    def clear_frameParams(self,frameName):
        "clear/initialize frame params in params_byFrame[frameName]"
        for varName in self.params_byFrame[frameName]:
            if varName in self.defaultValue:
                # has default value, overwrite to default value
                self.__dict__[varName] = self.defaultValue[varName]
            elif varName in self.tkVariable:
                pass # don't clear value for tk.Variables
            else:
                # no default value been set, delete var in self obj
                self.__dict__.pop(varName, None)
                
    def runtime_updatestatus(self,newstatus):
        "update tk.Variable status, and display status at run time"
        if hasattr(self,"status"):
            self.status.set("Status: "+newstatus) # update status
            self.root.update_idletasks() # call root to update status in window
        
    def tkraise(self):
        "raise first page of module"
        self.__dict__[self.frames[0]].tkraise()
        
    def pageframe_constructer(self,frame_format,Nframes,title=None,msg=None,
            helpPage=None,titleHeight=3,titleFont=HEADR_FONT,testMode=False):
        "return page&frames in page in default format, will auto detect status feature"
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {}
        assert isinstance(Nframes,int) and Nframes>=2, \
            "Nframes need >= 2, page should at lease have title&bottom frames"
        # construct page
        page = tk.Frame(frame_format)
        page.grid(row=0, column=0, sticky="nsew")
        # construct frames in page
        frames = [tk.Frame(page,**border) for _ in range(Nframes)]
        
        # frame geometry placement in page
        frames[0].pack(side="top", fill="x", expand=True, anchor="n")
        _ = [frame.pack(side="top", fill="y", expand=True) for frame in frames[1:-1]]
        frames[-1].pack(side="bottom", fill="x", expand=True, anchor="s")
        
        # set title&message in the first frame
        if title: tk.Label(frames[0], text=title, font=titleFont, height=titleHeight,
                           anchor='s').pack(pady=5)
        if msg: tk.Message(frames[0], text=msg, font=MESSG_FONT, width=500, 
                           justify="center").pack()
        
        # helper & status at the last frame
        subframe = tk.Frame(frames[-1],**border)
        tk.Label(subframe, text="").pack(side="right",pady=15) # freeze frame height
        subframe.pack(side='bottom',anchor='w')
        # add help button to display documentations window, if specified helpModPage
        if helpPage:
            # helpModPage is str: only specified page name, take __name__ as module name
            if isinstance(helpPage,str): helpPage = (self.__name__,helpPage)
            root = getattr(self,'root',False) or self # compatible for base module
            ttk.Button(frames[-1],text='Help',command = lambda:root.helper.show_doc(
                helpPage)).pack(side="left", padx=10)
        # display status at bottom left if self has status attribute
        if hasattr(self,'status'): tk.Message(subframe, textvariable=self.status, 
            font=STATUS_FONT, width=300).pack(side="left", anchor="s", padx=5, pady=2)
        return (page,*frames)
    
    def gridframe_constructer(self,frame,subtitles,col1texts,colN=4,separator=True,
                              subtitleFont=SUBHEAD_FONT,framewidth=550,padx=10,pady=5):
        "construct frame with grid placement, including subtitles,first col,seperator"
        subN = len(col1texts) # number of subframes
        if subtitles is None: subtitles = [None] * subN
        col1texts = [texts.copy() for texts in col1texts]
        if isinstance(separator,bool): separator = [separator] * (subN-1)
        
        # identify cum row # of each subtitle
        [texts.insert(0,"") for t,texts in zip(subtitles,col1texts) if t is not None]
        [texts.append("") for sep,texts in zip(separator,col1texts[:-1]) if sep]
        rowN, cumRowNs = list(map(len,col1texts)), [0]
        _ = [cumRowNs.append(cumRowNs[-1]+n) for n in rowN[:-1]]
        
        # construct subtitles, first col and seperator in frame
        _ = [tk.Label(frame, text=title, font=subtitleFont
                     ).grid(row=n,column=0,columnspan=colN,pady=pady)
             for n,title in zip(cumRowNs,subtitles) if title is not None]
        _ = [tk.Label(frame, text=text, justify ='left').grid(row=i,column=0,
             pady=pady,sticky='w') for i,text in enumerate(sum(col1texts,[])) if text]
        _ = [ttk.Separator(frame, orient="horizontal").grid(row=n-1,column=0,
             columnspan=colN,ipadx=framewidth/2,pady=pady,sticky='ew') 
             for n,sep in zip(cumRowNs[1:],separator) if sep]
        return cumRowNs # row # of each subtitle
    
    #############################################
    # GUI frames, default file page
    def _construct_filePage_frames(self,frame_format,msg=None,testMode=False):
        "default file page for module_GUI, ask user to provide i_fileName"
        # process title&message in File Page
        title = self.__name__+" | File Upload"
        if not msg: 
            msg = f"Select the MS Excel file with your {self.i_fileName.lower()} data."
            
        #########################################
        # input file page
        self.filePage, _, file_frame, bottom_frame = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,msg=msg,helpPage='filePage',
            titleHeight=7,testMode=testMode)
        
        # File Page select file part
        tk.Label(file_frame, text=self.i_fileName+" File:", font=MESSG_FONT
                ).grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_dir,width=60
                 ).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse",
                   command=lambda: self.button_selectFile(self.input_dir)
                  ).grid(row=0, column=2, padx=5, pady=5)
        
        # File Page bottom button for switching pages
        ttk.Button(bottom_frame, text="Next", command=self.button_Next
                  ).pack(side="right", padx=10)
        ttk.Button(bottom_frame, text="Back", command=self.button_Back
                  ).pack(side="right")
        
    #############################################
    # class methods as decorators
    def showerror(func):
        "decorator for replacing Exceptions with messagebox.showerror in GUI"
        def wrapper(self, *args,**kwargs):
            try:
                return func(self, *args,**kwargs)
            except Exception as e:
                self.runtime_updatestatus("error encountered!")
                messagebox.showerror("Error!",e)
        wrapper.__name__ = func.__name__
        return wrapper
    
    def show_status(timeit=False,msg=None):
        "decorator for setting status value before/after button cmd is executed"
        def decorator(func):
            def wrapper(self, *args,**kwargs):
                self.runtime_updatestatus("working...") # update status
                tStart = time.time()
                results = func(self, *args,**kwargs)
                tEnd = time.time()
                elaspeT = 'Time Elapsed: '+sec2str(tEnd-tStart) if timeit else ''
                self.runtime_updatestatus("done! "+elaspeT) # update status
                if msg: messagebox.showinfo(*msg)
                return results
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
        
    def check_requisites(*requisites):
        "decorator for checking requisites, raise error if not satisfied"
        def decorator(func):
            def wrapper(self, *args,**kwargs):
                for var in requisites:
                    value, msg = self.requisites[var]
                    assert self.__dict__[var] == value, msg
                return func(self, *args,**kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    #############################################
    # GUI button functions
    def button_selectFile(self,dirVar,purpose='Open'):
        "Displays a Open/Saveas file dialog box to user, dirVar is tk.StringVar object"
        assert purpose in ('Open','Saveas'),"Expected 'Open'/'Saveas' for arg 'purpose'!"
        func = getattr(filedialog,f'ask{purpose.lower()}filename')
        title = f"Select File to {purpose}"
        filetypes = [('Excel Files', '*.xlsx')]
        # ask for filename and write path to stringVar
        dirVar.set(func(initialdir="/Users/Documents",title=title,filetypes=filetypes))
        
    def button_Back(self,curframe="filePage",preframe="ChoosePage"):
        "clear values in current frame, switch to previous frame"
        if hasattr(self,'status'): self.status.set('') # rewrite status to blank
        self.clear_frameParams(curframe) # clear variables in curframe
        if preframe in self.frames:
            self.__dict__[preframe].tkraise()
        else:
            self.root.show_frame(preframe)
            
    def button_Next(self):
        pass


#######################################################################
# base class for the Hawkeye GUI
class App(tk.Tk):
    """ 
    This is the base of the program, its structure is what allows for multiple 'pages'.
    Note: Every 'page' manipulated is defined as its own unique class below this one.
    
    Examples:
    from HawkeyeGUI import *
    
    # display front page and all modules
    hawk = App()
    hawk.mainloop()
    
    # test modules using str, will search class code defined in HawkeyeGUI.GUI_modules
    hawk = App(testModule="Schedule Cost Integration")
    hawk.mainloop()
    
    # test modules using callable class, could pass *testargs, **testkwargs to init class
    # class ScheduleCost_GUI(Module_GUI): "the callable class defined in test scipt"
    testkwargs = {"titleHeight":6,"titleFont":HEADR_FONT,"testMode":True}
    hawk = App(testModule=ScheduleCost_GUI,**testkwargs)
    hawk.mainloop()
    """
    def __init__(self, testModule=None, *testargs, **testkwargs):
        """
        :testModule: None or str in GUI_modules, or module_GUI class for testing
        :param testargs: args passed to init module_GUI class
        :param testkwargs: kwargs passed to init module_GUI class
        """
        tk.Tk.__init__(self)
        self._setup_geometry()
        self._setup_icon_title()
        
        # initialize helper enable doc display
        self.helper = helpClass()
        
        # initialzie _frames,_modules to store frame/module obj
        self._frames,self._modules = {},{}
        self._setup_frame_format()
        self._initialize_modules(testModule, **testkwargs)
        
        if testModule is None:
            # normal mode, construct frames and raise startPage
            self._construct_frames()
            self._frames['StartPage'].tkraise()
        else:
            # test mode, testModule is str defined in GUI_modules or GUI class
            self.wm_title(str(testModule)+" Tester")
            self.show_frame(testModule)
            
    def _setup_geometry(self):
        "set the geometry such that the window show up at the center of user screen"
        # window and screen width&height
        window = (800,600)
        screen = (self.winfo_screenwidth(),self.winfo_screenheight())
        
        self.resizable(width=True, height=True)
        arg1,arg2 = (screen[0]-window[0])//2,(screen[1]-window[1])//2
        self.geometry(f'{window[0]}x{window[1]}+{arg1}+{arg2}')
        
    def _setup_icon_title(self):
        # add window icon&title
        self.iconbitmap(default=os.path.join(HOME,'hawkeye.ico'))
        self.wm_title("HawkEye "+VERSION)
        
    def _setup_frame_format(self):
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
    def _construct_frames(self):
        "GUI default frames in Hawkeye, disabled in testMode"
        #########################################
        # StartPage
        StartPage, _, center_frame, button_frame = Module_GUI.pageframe_constructer(
            self,self.container,3,"HawkEye "+VERSION,titleFont=TITLE_FONT)
        self._frames['StartPage'] = StartPage # add to _frames dict
        
        # Start Page User Agreement, text stored in documentations.xml
        tk.Label(center_frame, text="User Agreement", font=HEADR_FONT).pack()
        for node in self.helper.get_node('base', 'Agreement')[1:]:
            tk.Message(center_frame, text=node.text, justify="left", width=525, 
                       font=MESSG_FONT).pack(pady=5)
        # Start Page Product Introduction
        ttk.Button(center_frame, text="Product Introduction", command=lambda:
                   self.helper.show_doc(('base','Introduction'))).pack(ipadx=10,ipady=4)
        # Start Page bottom button for switching pages
        ttk.Button(button_frame, text="Accept", command=lambda:
                   self.show_frame('ChoosePage')).pack(side="right", padx=10)
        ttk.Button(button_frame, text="Decline", command=self.destroy
                  ).pack(side="right")
        
        #########################################
        # ChoosePage
        title = "Project Risk and Decision Analysis Modules"
        msg = "\nClick on the module of your interest:\n\n"
        ChoosePage, _, center_frame, bottom_frame = Module_GUI.pageframe_constructer(
            self,self.container,Nframes=3,helpPage=('base','ChoosePage'))
        self._frames['ChoosePage'] = ChoosePage # add to _frames dict
        
        # Choose Page title
        tk.Label(center_frame, text=title, font=HEADR_FONT, height=3, anchor='s').pack()
        tk.Message(center_frame, text=msg, justify="center", width=500, 
                   font=MESSG_FONT).pack()
        # Choose Page center button for laoding pages
        for i,moduleName in enumerate(GUI_modules):
            # find frame/module name to raise, disabled button if is not active
            state = "normal" if self.isactive[i] else "disabled"
            ttk.Button(center_frame, text=moduleName, width=30, state=state,
                       command=lambda name=moduleName:self.show_frame(name)).pack(pady=2)
            
        # Choose Page bottom button for switching pages
        ttk.Button(bottom_frame, text="Back", command=self._frames['StartPage'].tkraise
                  ).pack(side="right", padx=10)
        
    def _construct_module(self, moduleName, namespace):
        "construct single module in GUI_modules, fetch code in namespace"
        moduleClassNames = GUI_modules[moduleName]# name of module classes
        try:
            if isinstance(moduleClassNames,str):
                # module classes is one single str, store obj in self._modules
                name = moduleClassNames
                self._modules[name] = getattr(namespace,name)(self.container, self)
            elif isinstance(moduleClassNames,tuple) and len(moduleClassNames)>0:
                # module classes is list of str, store frame obj in self._frames
                for name in moduleClassNames:
                    frame_obj = getattr(namespace, name)(self.container, self)
                    frame_obj.grid(row=0, column=0, sticky="nsew")
                    self._frames[name] = frame_obj
            else:
                raise ValueError("No module classes defined.")
        except Exception as e:
                print(f"{e}, Module '{moduleName}' disabled!")
                return False
        return True
        
    def _initialize_modules(self, testModule=None, *testargs, **testkwargs):
        '''initialize modules testModule or modules in GUI_modules, module class code 
        is written follow new format inherated from the Module_GUI and 'Module', module 
        obj store all frames of module and also the data.
        '''
        namespace = sys.modules[__name__] # namespace __name__ for searching
        if testModule is None:
            # initialize all modules in GUI_modules
            # store boolean of whether module is initialized properly
            self.isactive = [self._construct_module(moduleName,namespace)
                             for moduleName in GUI_modules]
        elif testModule in GUI_modules:
            # testModule is str in GUI_modules, initialize module follow GUI_modules
            self._construct_module(testModule,namespace)
        elif callable(testModule):
            # only initialize testModule
            self._modules[testModule] = testModule(self.container, self, 
                                                   *testargs, **testkwargs)
        else:
            raise ValueError(f"Test Module '{testModule}' not found in GUI_modules " +\
                             "and not callable")
            
    def show_frame(self, frame_module):
        "Raise frame/module, search from self._frames and self._modules dict"
        if frame_module in self._modules:
            # raise first frame (in module.frames) of module
            self._modules[frame_module].tkraise()
        elif frame_module in self._frames:
            # directly raise frame in _frames dict
            self._frames[frame_module].tkraise()
        elif frame_module in GUI_modules:
            vals = GUI_modules[frame_module]
            frame_module = vals if isinstance(vals,str) else vals[0]
            self.show_frame(frame_module)
        else:
            raise ValueError(f"'{frame_module}' not found in frame or module!")


#######################################################################
# classes for the GUI part of each module
class Sensitivity_GUI(Module_GUI):
    """
    Sensitivity_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Sensitivity Analysis' # Module name, for display purpose in frames
    frames = ['outputPage'] # frames contained in Module
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {'outputPage': ['DPobj']}
    # default params value, used in initializationa and clear value
    defaultValue = {'analyzedFlag': False}
    # tk.Variables for Module, used in initialization only, won't clear value
    tkVariable = {'status': (tk.StringVar,''), 'mod': (tk.StringVar,),
                  'rate': (PercentVar,20.0), 'input_dir': (tk.StringVar,),
                  'percent': (PercentVar,15.0), 'topN': (tk.StringVar,'10')}
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['mod','rate','input_dir','percent'],'w','paramChanged')] 
    # requisites dict, used in decorator self.check_requisites
    requisites = {'analyzedFlag': (True, "Not analyzed yet! click Analyze button first")}
    
    # --------- GUI outputPage Contents --------
    subtitles = [None, "File Upload", "Settings",
                 "Sensitivity Results for Analysis & Display / Save"]
    texts = [[''],['File'],['',"Percentage Change\nin Inputs","Top 'N' Impacts"],['','']]
    def __init__(self,frame_format,root,**testkwargs):
        Module_GUI.__init__(self,frame_format,root)
        
        # construct module frames
        self._construct_module_frames(frame_format,**testkwargs)
    
    def paramChanged(self,*args):
        "tk.Variable.trace function to flag paramFlag as True"
        self.analyzedFlag = False
    
    #############################################
    # GUI frames functions
    def _construct_module_frames(self,frame_format,testMode=False):
        #########################################
        # outputPage
        title,subtitles,texts = self.__name__,self.subtitles,self.texts
        msg = "Please select the module of your interest in the box below."
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {}
        framewidth,padx,pady = 450,10,6
        # construct page, frames, titleframe, status text
        modPage = lambda : (self.__name__,self.mod.get())
        self.outputPage,_,center_f,bottom_f = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,msg=msg,helpPage=modPage,testMode=testMode)
        # create subtitle&first column of each row as tk.Lable, and seperator
        ns = self.gridframe_constructer(center_f,subtitles,texts,colN=4,separator=True,
                                        framewidth=framewidth,padx=padx,pady=pady)
        
        # Module Selection & File Upload | outputPage
        modsAllowed = list(SensitivityMods)
        ttk.Combobox(center_f,textvariable=self.mod,state="readonly",values=modsAllowed
                    ).grid(row=0, column=1, padx=padx, pady=pady, sticky='w')
        # Input File Selection
        ttk.Entry(center_f, textvariable=self.input_dir,width=40
                 ).grid(row=ns[1]+1, column=1, columnspan=2, padx=padx, pady=pady)
        ttk.Button(center_f, text="Browse",
                   command=lambda: self.button_selectFile(self.input_dir)
                  ).grid(row=ns[1]+1, column=3, padx=padx, pady=pady)
        
        # Settings | outputPage
        # Discount Rate frame in 1st row, conditional visible
        rate_t = tk.Label(center_f, text='Annual Discount Rate')
        rate_f = tk.Frame(center_f)
        ttk.Entry(rate_f,textvariable=self.rate,width=6).pack(side="left")
        tk.Label(rate_f, text="%").pack(side="left")
        def tracefunc(*arg):
            if self.mod.get() == "Net Present Value":
                rate_f.grid(row=ns[2]+1,column=1,padx=34,sticky='w')
                rate_t.grid(row=ns[2]+1,column=0,pady=pady,sticky='w')
            else:
                rate_f.grid_forget(); rate_t.grid_forget()
        self.mod.trace('w',tracefunc)
        self.mod.set(modsAllowed[0])
        # Percentage setting in 2nd row
        percent_f = tk.Frame(center_f,**border)
        percent_f.grid(row=ns[2]+2,column=1,padx=padx,sticky='w')
        tk.Label(percent_f, text="+/-").pack(side="left")
        ttk.Entry(percent_f,textvariable=self.percent,width=6).pack(side="left")
        tk.Label(percent_f, text="%").pack(side="left")
        # Top N Impact in 3rd row
        topNs = ['5','10','15','20','10%','25%','50%','75%','100%']
        ttk.Combobox(center_f,textvariable=self.topN,state="readonly",values=topNs,
                     width=10).grid(row=ns[2]+3, column=1, columnspan=2, padx=padx, sticky='w')
        
        # Sensitivity Results Display | outputPage
        # Sensitivity Analysis in 1st row
        ttk.Button(center_f, text="Analyze", command=self.button_Analyze, width=25
                  ).grid(row=ns[3]+1,column=0,columnspan=4,padx=padx,pady=pady)
        # Data Export in 2nd row
        ttk.Button(center_f,text="Display / Save", command=self.button_Export, width=25
                  ).grid(row=ns[3]+2,column=0,columnspan=4,padx=padx,pady=pady)
        
        # Swithing Page Button | outputPage
        ttk.Button(bottom_f, text="Back", command=lambda: self.button_Back(
            'outputPage')).pack(side="right",padx=10)
    
    #############################################
    # GUI button functions | outputPage
    @Module_GUI.showerror
    # @Module_GUI.show_status(timeit=True) # Error encountered, code after plot won't run
    def button_Analyze(self):
        # process file directory and initialize DataProcess object
        input_dir,module = self.input_dir.get(),self.mod.get()
        assert input_dir, module+" file directory not specified!"
        self.DPobj = Sensitivity(input_dir, module)
        
        # process & pass user input from GUI
        p = self.percent.get()
        self.DPobj.setup_userInputs([-p,p],self.rate.get())
        self.DPobj.analyze()
        
        # process topN and plot tornado
        topN = self.topN.get()
        topN = int(topN[:-1])/100 if '%' in topN else int(topN)
        self.analyzedFlag = True
        self.DPobj.plot_Tornado(topN=topN)

    @Module_GUI.showerror
    @Module_GUI.check_requisites('analyzedFlag')
    @Module_GUI.show_status(timeit=True)
    def button_Export(self):
        self.DPobj.export()


class NPV_GUI(Module_GUI):
    """
    NPV_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Project Net Present Value' # Module name, for display purpose in frames
    frames = ['filePage','outputPage'] # frames contained in Module
    DataProcessClass = NPV # Data Process Class linked, used in button_Next
    statsFmt = lambda _,x: f"-$ {-x:,.0f}" if x<0 else f"$ {x:,.0f}" # fmt in statsTable
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'Net Present Value' # fileName displayed in filePage
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {'filePage':[], 'outputPage':['simulationFlag','paramFlag','DPobj']}
    # default params value, used in initializationa and clear value
    defaultValue = {'simulationFlag': False, 'paramFlag': False}
    # tk.Variables for Module, used in initialization only, won't clear value
    tkVariable = {'status': (tk.StringVar,''), 'input_dir': (tk.StringVar,),
                  'rates': (tk.StringVar,'15,20,25'), 'nIter': (PositiveIntVar,10000),
                  'CFId': (IdVar,), 'rateId': (IdVar,)}
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['nIter','rates'],'w','paramChanged'), (['CFId'],'w','traceCFId')]
    # requisites dict, used in decorator self.check_requisites
    requisites = {'simulationFlag': (True, "Simulation not runned yet!"),
                  'paramFlag': (False, "Discount Rates or Simulation setting "+\
                                "is changed.\nclick Run Simulation again!")}
    
    # --------- GUI outputPage Contents --------
    subtitles = ["Discount Rates & Simulation Settings", 
                 "Cash Flows Distribution Display", "Net Present Value Display", 
                 "Simulated Values - Display / Save"]
    texts = [['Annual Discount Rates','Number of Iterations'], ['Year: Cash Flow'],
             ['Net Present Value','NPV Analysis'],['']]
    figTypes = ['Tornado Diagram','Correlation Coeff.']
    def __init__(self,frame_format,root,**testkwargs):
        Module_GUI.__init__(self,frame_format,root)
        
        # using default filePage frame
        self._construct_filePage_frames(frame_format)
        # construct module frames
        self._construct_module_frames(frame_format,**testkwargs)
    
    def paramChanged(self,*args):
        "tk.Variable.trace function to flag paramFlag as True"
        self.paramFlag = True
    
    def traceCFId(self,*args):
        "update distribution text and disable stats/plot button"
        dist = self.DPobj.CFs[self.CFId.get()].dist.__name__
        self.CFId.label.set(dist) # add distribution name to label widge
        state = 'disabled' if dist in ('No Distribution') else 'normal'
        self.CFId.widges[-1]['state'] = state
    
    #############################################
    # GUI frames functions
    def _construct_module_frames(self,frame_format,testMode=False):
        #########################################
        # outputPage
        title,subtitles,texts = self.__name__,self.subtitles,self.texts
        # msg = "Input 'Number of Iterations' and click on 'Run Simulation'."
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {}
        framewidth,padx,pady = 500,5,5
        # construct page, frames, titleframe, status text
        self.outputPage,_,center_f,bottom_f = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,helpPage='outputPage',testMode=testMode)
        # create subtitle&first column of each row as tk.Lable, and seperator
        ns = self.gridframe_constructer(center_f,subtitles,texts,colN=5,separator=True,
                                        framewidth=framewidth,padx=padx,pady=pady)
        
        # Discount Rates & Simulation Settings | outputPage
        # Discount Rates in 1st row 
        rate_f = tk.Frame(center_f,**border)
        rate_f.grid(row=1,column=2,columnspan=2)
        entry = ttk.Entry(rate_f,textvariable=self.rates,width=11,foreground='grey')
        entry.pack(side="left")
        # change fg color when focus in
        changeRatefg = lambda _: entry.config(foreground='black')
        entry.bind("<FocusIn>", changeRatefg)
        tk.Label(rate_f, text="%").pack(side="left")
        # Simulation setting in 2nd row
        self.nIter.Entry(center_f,width=7).grid(row=2,column=1,columnspan=2)
        ttk.Button(center_f, text="Run Simulation", command=self.button_RunSimulation
                  ).grid(row=2,column=2,columnspan=2)#,sticky='w')
        
        # Cash Flows Distribution Display | outputPage
        tk.Label(center_f,text='').grid(row=ns[1]+1,column=1,padx=50) # freeze col width
        self.CFId.Combobox(center_f,width=18
                          ).grid(row=ns[1]+1,column=0,columnspan=2,sticky='e')
        self.CFId.Label(center_f).grid(row=ns[1]+1,column=2,sticky='ew',padx=padx)
        self.CFId.Button(center_f,text="Statistics",command=self.button_CFStats
                        ).grid(row=ns[1]+1,column=3,padx=padx)
        self.CFId.Button(center_f,text="Distribution",command=self.button_CFPlot
                        ).grid(row=ns[1]+1,column=4,padx=padx)
        
        # StatisticsFigure Display | outputPage
        # Project Net Present Value in 1st row
        ttk.Button(center_f, text="Statistics", command=self.button_NPVStats
                  ).grid(row=ns[2]+1,column=1,sticky='e')
        ttk.Button(center_f, text="PDF Plot", command=self.button_PDFCDFPlot
                  ).grid(row=ns[2]+1,column=2,columnspan=2)
        ttk.Button(center_f, text="CDF Plot", command=lambda: self.button_PDFCDFPlot(
            "CDF")).grid(row=ns[2]+1,column=4,sticky='w')
        # Project NPV Analysis in 2nd row
        self.rateId.Combobox(center_f,width=9).grid(row=ns[2]+2,column=1,sticky='e')
        ttk.Button(center_f, text="Tornado Diagram", command=self.button_PerfPlot
                  ).grid(row=ns[2]+2,column=2,columnspan=2)
        
        # Data Export | outputPage
        ttk.Button(center_f,text="Display / Save", command=self.button_Browse, width=30
                  ).grid(row=ns[3]+1,column=0,columnspan=5)
        
        # Swithing Page Button | outputPage
        ttk.Button(bottom_f, text="Back", command=lambda: self.button_Back(
            'outputPage','filePage')).pack(side="right",padx=10)
    
    #############################################
    # GUI button functions | filePage
    @Module_GUI.showerror
    def button_Next(self):
        # fetch file directory
        filedir = self.input_dir.get()
        assert filedir,self.i_fileName+" file directory not specified!"
        # initialzied schedule cost obj and raise next page
        self.DPobj = self.DataProcessClass(filedir)
        # dynamically update dropdown list
        content = [(f"Year {c.year}: {c.name}",i) for i,c in enumerate(self.DPobj.CFs)]
        self.CFId.add_ComboboxContent(text2id=content)
        self.outputPage.tkraise()
        
    #############################################
    # GUI button functions | outputPage
    @Module_GUI.showerror
    @Module_GUI.show_status(timeit=True,msg=("Completed!","Simulation is done."))
    def button_RunSimulation(self):
        # process parameters: pass discount rates
        try:
            rates = [float(r)/100 for r in self.rates.get().split(',')]
        except:
            raise ValueError("Invalid Entry 'Discount Rates'!\n"\
                             "expect float numbers separated by ','")
        self.DPobj.setup_DiscountFactors(rates)
        content = [(f"{r:.2%}",i) for i,r in enumerate(self.DPobj.rates)]
        self.rateId.add_ComboboxContent(text2id=content)
        # run simulation & calculate simulated cost results
        self.runtime_updatestatus("simulating random variables...")
        self.DPobj.simulate(self.nIter.get())
        self.runtime_updatestatus("calculating simulated results...")
        self.DPobj.calculate_simresults()
        # rewrite Flags and update entry
        self.simulationFlag,self.paramFlag = True,False
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_CFStats(self):
        CFId = self.CFId.get(); CF = self.DPobj.CFs[CFId]
        title = f"{CF.disp}; {CF.dist.__name__} Distribution"
        get_statstable(self.DPobj.cashflows[:,CFId],title,self.statsFmt)
        
    def button_CFPlot(self):
        CF = self.DPobj.CFs[self.CFId.get()]
        xlabel = f"Cash Flow, {self.DPobj.unit}"
        title = f"{CF.disp}\n{CF.dist.__name__} Distribution"
        plt_pdfcdf(CF.dist,xlabel=xlabel,title=title,xfmt=self.DPobj.fmt)
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_NPVStats(self):
        header = ['Discount Rate']+[f"{x:.2%}" for x in self.DPobj.rates]
        get_statstable(self.DPobj.NPVs,self.DPobj.title,self.statsFmt,header)
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_PDFCDFPlot(self,figtype='PDF'):
        self.DPobj.plot_PDFCDF(figtype=figtype)
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_PerfPlot(self):
        self.DPobj.plot_performance(rateId=self.rateId.get())
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    @Module_GUI.show_status(timeit=True)
    def button_Browse(self):
        self.DPobj.browse()


class Cost_GUI(Module_GUI):
    """
    Cost_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Project Cost' # Module name, for display purpose in frames
    frames = ['filePage','outputPage'] # frames contained in Module
    DataProcessClass = Cost # Data Process Class linked, used in button_Next
    impact = Cost.riskImpacts # Risk Impact Applied for Module
    impactattr, impactvar = [Cost.riskImpacts.lower()+s for s in ['ImpactDist','Impact']]
    statsFmt = lambda _,x: f"-$ {-x:,.0f}" if x<0 else f"$ {x:,.0f}" # fmt in statsTable
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'Project Cost' # fileName displayed in filePage
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {'filePage':[], 'outputPage':['simulationFlag','paramFlag','DPobj']}
    # default params value, used in initializationa and clear value
    defaultValue = {'simulationFlag': False, 'paramFlag': False}
    # tk.Variables for Module, used in initialization only, won't clear value
    tkVariable = {'status': (tk.StringVar,''), 'input_dir': (tk.StringVar,),
                  'nIter': (PositiveIntVar,10000), 'taskId': (IdVar,), 'riskId': (IdVar,),
                  'withRisk': (tk.BooleanVar,True), 'withRisk_e': (tk.BooleanVar,True),
                  'figType': (tk.StringVar,'Tornado Diagram')}
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['nIter'],'w','paramChanged'),
                 (['taskId'],'w','traceTaskId'),
                 (['riskId'],'w','traceRiskId')]
    # requisites dict, used in decorator self.check_requisites
    requisites = {'simulationFlag': (True, "Simulation not runned yet!"),
                  'paramFlag': (False, "Simulation setting is changed.\n"+\
                                "click Run Simulation again!")}
    
    # --------- GUI outputPage Contents --------
    subtitles = ["Simulation Settings", "Task & Risk Event Distribution Display",
                 "Project Cost Display", "Simulated Values - Display / Save"]
    texts = [['Number of Iterations'], ['ID: Task','ID: Risk Event'],
             ['','Project Cost','Project Cost Analysis',
              'Project Cost With\nAND Without Risk Events'],['','']]
    figTypes = ['Tornado Diagram','Correlation Coefficients']
    def __init__(self,frame_format,root,**testkwargs):
        Module_GUI.__init__(self,frame_format,root)
        self.omitFilePage = testkwargs.pop('omitFilePage',False)
        # using default filePage frame
        if not self.omitFilePage: self._construct_filePage_frames(frame_format)
        # construct module frames
        self._construct_module_frames(frame_format,**testkwargs)
    
    def paramChanged(self,*args):
        "tk.Variable.trace function to flag paramFlag as True"
        self.paramFlag = True
    
    def traceTaskId(self,*args):
        "update distribution text and disable stats/plot button"
        dist = self.DPobj.tasks[self.taskId.get()].dist
        self.taskId.label.set(dist.__name__) # add distribution name to label widge
        self.taskId.widges[-1]['state'] = 'normal' if hasattr(dist,'ppf') else 'disabled'
        
    def traceRiskId(self,*args):
        "update distribution text and disable stats/plot button"
        dist = getattr(self.DPobj.risks[self.riskId.get()],self.impactattr)
        distname = "No Impact" if dist is None else dist.__name__
        self.riskId.label.set(distname)  # add distribution name to label widge
        self.riskId.widges[-1]['state'] = 'normal' if hasattr(dist,'ppf') else 'disabled'
        self.riskId.widges[-2]['state'] = 'disabled' if dist is None else 'normal'
        
    #############################################
    # GUI frames functions
    def _construct_module_frames(self,frame_format,testMode=False):
        #########################################
        # outputPage
        title,subtitles,texts = self.__name__,self.subtitles,self.texts
        # msg = "Input 'Number of Iterations' and click on 'Run Simulation'."
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {}
        framewidth,padx,pady = 550,10,4
        # construct page, frames, titleframe, status text
        self.outputPage,_,center_f,bottom_f = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,helpPage='outputPage',testMode=testMode)
        # create subtitle&first column of each row as tk.Lable, and seperator
        ns = self.gridframe_constructer(center_f,subtitles,texts,colN=5,separator=True,
                                        framewidth=framewidth,padx=padx,pady=pady)
        
        # Simulation setting in 1st row | outputPage
        self.nIter.Entry(center_f,width=7).grid(row=1,column=1,columnspan=2)
        ttk.Button(center_f, text="Run Simulation", command=self.button_RunSimulation
                  ).grid(row=1,column=2,columnspan=2)#,sticky='w')
        
        # Task & Risk Distribution Display | outputPage
        # Task/Risk Distribution Display in 1st/2nd row
        tk.Label(center_f,text='').grid(row=ns[1]+1,column=1,padx=60) # freeze col1 width
        for n,Idtype in enumerate(["Task","Risk"],ns[1]+1):
            Id = self.__dict__[Idtype.lower()+'Id']
            Id.Combobox(center_f).grid(row=n,column=0,columnspan=2,sticky='e')
            Id.Label(center_f).grid(row=n,column=2,sticky='ew',padx=2)
            btn_stats = getattr(self,f'button_{Idtype}Stats')
            Id.Button(center_f,text="Statistics",command=btn_stats).grid(row=n,column=3)
            btn_plt = getattr(self,f'button_{Idtype}Plot')
            Id.Button(center_f,text="Distribution",command=btn_plt).grid(row=n,column=4)
        
        # StatisticsFigure Display | outputPage
        # Choose Data Set in 1st row
        n = ns[2]; texts = ['With Risk Events','Without Risk Events']
        ListVar.Radiobuttons(center_f,texts,self.withRisk,[True,False],padx=padx,
                             **border).grid(row=n+1,column=1,columnspan=4,sticky='w')
        # Project Total Cost in 2nd row
        ttk.Button(center_f, text="Statistics", command=self.button_ProjectStats
                  ).grid(row=n+2,column=1,columnspan=2)
        ttk.Button(center_f, text="PDF & CDF Plots", command=self.button_PDFCDFPlot
                  ).grid(row=n+2,column=2,columnspan=2)
        # Project Cost Performance in 3rd row
        ListVar.Radiobuttons(center_f,self.figTypes,self.figType,method='grid',padx=padx,
                             colN=2).grid(row=n+3,column=1,columnspan=3,sticky='w')
        ttk.Button(center_f, text="Plot", command=self.button_PerfPlot
                  ).grid(row=n+3,column=4)
        # Project Total Cost with VS without Risk Events Plot in 4th row
        ttk.Button(center_f, text="PDF Plot", command=lambda:self.button_PDFCDFPlot(
            "PDF")).grid(row=n+4,column=1,columnspan=2)
        ttk.Button(center_f, text="CDF Plot", command=lambda:self.button_PDFCDFPlot(
            "CDF"),width=15).grid(row=n+4,column=2,columnspan=2)
        
        # Data Export | outputPage
        # Choose Data Set in 1st row
        n = ns[3]; texts = ['With Risk Events','Without Risk Events']
        ListVar.Radiobuttons(center_f,texts,self.withRisk_e,[True,False],padx=padx,
                             **border).grid(row=n+1,column=1,columnspan=4,sticky='w')
        # Choose Variables in 2nd row
        ttk.Button(center_f,text="Display / Save", command=self.button_Browse, width=30
                  ).grid(row=n+2,column=0,columnspan=5)
        
        # Swithing Page Button | outputPage
        if not self.omitFilePage:
            ttk.Button(bottom_f, text="Back", command=lambda: self.button_Back(
                'outputPage','filePage')).pack(side="right",padx=10)
    
    #############################################
    # GUI button functions | filePage
    @Module_GUI.showerror
    def button_Next(self):
        if not self.omitFilePage:
            # fetch file directory
            filedir = self.input_dir.get()
            assert filedir,self.i_fileName+" file directory not specified!"
            # initialzied schedule cost obj and raise next page
            self.DPobj = self.DataProcessClass(filedir)
        # dynamically update dropdown list
        content = [(f"{s.id}: {s.name}",i) for i,s in enumerate(self.DPobj.tasks)]
        self.taskId.add_ComboboxContent(text2id=content)
        content = [(f"{s.id}: {s.name}",i) for i,s in enumerate(self.DPobj.risks)]
        self.riskId.add_ComboboxContent(text2id=content)
        self.outputPage.tkraise()
        
    #############################################
    # GUI button functions | outputPage
    @Module_GUI.showerror
    @Module_GUI.show_status(timeit=True,msg=("Completed!","Simulation is done."))
    def button_RunSimulation(self):
        # run simulation & calculate simulated cost results
        self.runtime_updatestatus("simulating random variables...")
        self.DPobj.simulate(self.nIter.get())
        self.runtime_updatestatus("calculating simulated results...")
        self.DPobj.calculate_simresults()
        # rewrite Flags and update entry
        self.simulationFlag,self.paramFlag = True,False
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_TaskStats(self):
        taskId = self.taskId.get(); task = self.DPobj.tasks[taskId]
        title = f"Task {task.id}: {task.name}; Distribution: {task.dist.__name__}"
        get_statstable(self.DPobj.task_norisk[:,taskId],title,self.statsFmt)

    def button_TaskPlot(self):
        task = self.DPobj.tasks[self.taskId.get()]
        xlabel = f"{self.DPobj.title(None,'Task')}, {self.DPobj.unit}"
        title = f"Task {task.id}: {task.name}\n{task.dist.__name__} Distribution"
        plt_pdfcdf(task.dist,xlabel=xlabel,title=title,xfmt=self.DPobj.fmt)
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_RiskStats(self):
        riskId = self.riskId.get(); risk = self.DPobj.risks[riskId]
        title = f"Risk {risk.id}: {risk.name} Frequency: {risk.freqDist.__name__}; " \
                f"{self.impact} Impact: {getattr(risk,self.impactattr).__name__}"
        get_statstable(getattr(self.DPobj,self.impactvar)[riskId],title,self.statsFmt)
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_RiskPlot(self):
        riskId = self.riskId.get(); risk = self.DPobj.risks[riskId]
        taskIdImpacted = getattr(risk,self.impact.lower()+'TaskId')
        freqDistName = risk.freqDist.__name__
        impactDist = getattr(risk,self.impactattr)
        title = f"Risk {risk.id}: {risk.name} Task Impacted: {taskIdImpacted}\n" \
                f"Frequency: {freqDistName}; {self.impact} Impact: {impactDist.__name__}"
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(9.6,4.8)); fig.suptitle(title)
        plt_hist(getattr(self.DPobj,self.impactvar)[riskId],ax=ax1,
                 xlabel="Risk Impact, "+self.DPobj.unit,xfmt=self.DPobj.fmt)
        if freqDistName == "Poisson":
            data = self.DPobj.riskFreqs[riskId]; xlabel = "Risk Frequency (# of Events)"
            maxFreq = data.max(); bins = [(i-0.5)/3 for i in range(maxFreq*3+2)]
            plt_hist(data,ax=ax2,bins=bins,color='tab:green',xlabel=xlabel,showfig=True)
        else:
            xlabel = impactDist.__name__
            plt_pdf(impactDist,ax=ax2,xlabel=xlabel,xfmt=self.DPobj.fmt,showfig=True)
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_ProjectStats(self):
        withrisk = self.withRisk.get()
        totalCost = self.DPobj.get('totalCost',withrisk)
        title = self.DPobj.title(withrisk)
        get_statstable(totalCost,title,self.statsFmt)
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_PDFCDFPlot(self,figtype='PDF&CDF'):
        self.DPobj.plot_PDFCDF(figtype=figtype,withrisk=self.withRisk.get())
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_PerfPlot(self):
        self.DPobj.plot_performance(self.figType.get(), self.withRisk.get())
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    @Module_GUI.show_status(timeit=True)
    def button_Browse(self):
        self.DPobj.browse(self.withRisk_e.get())


class Schedule_GUI(Cost_GUI):
    """
    Schedule_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Project Schedule'
    frames = ['filePage','outputPage']
    DataProcessClass = Schedule # Data Process Class linked, used in button_Next
    impact = Schedule.riskImpacts # Risk Impact Applied for Module
    impactattr,impactvar = [Schedule.riskImpacts.lower()+s 
                            for s in ['ImpactDist','Impact']]
    statsFmt = "{x:,.1f} days" # fmt in statsTable
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'Project Schedule' # fileName displayed in filePage
    
    # ----------- Module GUI configs -----------
    # inherit params_byFrame,defaultValue,tkVariable,var2Trace,requisites
    
    # --------- GUI outputPage Contents --------
    subtitles = ["Simulation Settings", "Task & Risk Event Distribution Display",
                 "Project Schedule Display", "Simulated Values - Display / Save"]
    texts = [['Number of Iterations'], ['ID: Task','ID: Risk Event'],
             ['','Project Duration','Project Duration\nAnalysis',
              'Project Duration With\nAND Without Risk Events'],['','']]
    figTypes = ['Tornado Diagram','Correlation Coefficients',
                'Critical Index','Schedule Impact Indicators']
    #############################################
    # GUI frames functions | frame page inherited from Cost_GUI
    
    #############################################
    # GUI button functions | filePage: button_Next inherited from Cost_GUI
    # inherit buttons: Next
    
    #############################################
    # GUI button functions | outputPage
    # inherit buttons: RunSimulation,TaskStats,TaskPlot,RiskStats,RiskPlot,
    #                  PDFCDFPlot,PerfPlot,Browse
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_ProjectStats(self):
        withrisk = self.withRisk.get()
        title = self.DPobj.title(withrisk)
        get_statstable(self.DPobj.get('totalDuration',withrisk),title,self.statsFmt)


class ScheduleCost_GUI(Module_GUI):
    """
    ScheduleCost_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Schedule & Cost Integration'
    frames = ['filePage','outputPage']
    statsFmt = lambda _,x: f"-$ {-x:,.0f}" if x<0 else f"$ {x:,.0f}" # fmt in statsTable
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'Schedule & Cost'
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {'filePage':[], 'outputPage':['simulationFlag','paramFlag','SCobj']}
    # default params value, used in initializationa and clear value
    defaultValue = {'simulationFlag': False, 'paramFlag': False}
    # tk.Variables for Module, used in initialization only, won't clear value
    tkVariable = {'status': (tk.StringVar,''), 'input_dir': (tk.StringVar,),
                  'nIter': (PositiveIntVar,10000), 'frequency': (tk.StringVar,'Monthly'),
                  'custom_freq': (tk.DoubleVar,21), 'withRisk': (tk.BooleanVar,True),
                  'periodN': (tk.IntVar,''), 'withRisk_e': (tk.BooleanVar,True),
                  'stats': (StatsVar,['Minimum','Maximum','Mean','Percentiles:20,80']),
                  'vars': (ListVar,['Duration','Risk Events','Cost','Cost Statistics'])}
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['nIter','frequency','custom_freq'],'w','paramChanged'),
                 (['withRisk'],'w','updatePeriodN'),
                 (['withRisk_e'],'w','disableRiskImpactBrowse')]
    # requisites dict, used in decorator self.check_requisites
    requisites = {'simulationFlag': (True, "Simulation not runned yet!"),
                  'paramFlag': (False, "Simulation/Integration setting is changed."+\
                                "\nclick Run Simulation again!")}
    # --------- GUI outputPage Contents --------
    subtitles = ["Integration & Simulation Settings", "Schedule & Cost Only Results",
                 "Project Cumulative Cost Display", "Simulated Values - Display / Save"]
    texts=[['Integration Frequency','Number of Iterations'],[' '],
           ['','Month/Week/\nCustom Period #',
            'Data to be Displayed\nin Integration Curves'],
           ['','Simulated Variables:','']]
    def __init__(self,frame_format,root,**testkwargs):
        Module_GUI.__init__(self,frame_format,root)
        
        # using default filePage frame
        self._construct_filePage_frames(frame_format)
        # construct module frames
        self._construct_module_frames(frame_format,**testkwargs)
    
    def paramChanged(self,*args):
        "tk.Variable.trace function to flag paramFlag as True"
        self.paramFlag = True
    
    def updatePeriodN(self,*args):
        "update self.periodN value"
        if not self.simulationFlag: return
        withrisk = self.withRisk.get()
        cumCost = self.SCobj.cumCost_risk if withrisk else self.SCobj.cumCost_norisk
        self.periodN.set(cumCost.shape[1])
        
    def disableRiskImpactBrowse(self,*args):
        "disable Risk Impact variable for Browse by default"
        if self.withRisk_e.get():
            self.vars.BoolVars[1].set(True)
            self.vars.buttons[1].config(state='normal')
        else:
            self.vars.BoolVars[1].set(False)
            self.vars.buttons[1].config(state='disabled')
        
    #############################################
    # GUI frames functions
    def _construct_module_frames(self,frame_format,testMode=False):
        #########################################
        # outputPage
        title,subtitles,texts = self.__name__,self.subtitles,self.texts
        # msg = "Input 'Number of Iterations' and click on 'Run Simulation'."
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {}
        framewidth,padx,pady = 550,10,4
        # construct page, frames, titleframe, status text
        self.outputPage,_,center_f,bottom_f = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,helpPage='outputPage',testMode=testMode)
        # create subtitle&first column of each row as tk.Lable, and seperator
        ns = self.gridframe_constructer(center_f,subtitles,texts,colN=4,separator=True,
                                        framewidth=framewidth,padx=padx,pady=pady)
        
        # Simulation&Integration Setting | outputPage
        # Integration Frequency setting in 1st row
        texts = ['Monthly','Weekly','Custom Frequency:']
        freq_f = ListVar.Radiobuttons(center_f,texts,self.frequency,padx=padx,**border)
        freq_f.grid(row=1,column=1,columnspan=3,sticky='w')
        entry = ttk.Entry(freq_f,textvariable=self.custom_freq,width=3,
                          show=' ',state='disabled')
        entry.pack(side="left")
        self.frequency.trace('w',lambda *args: entry.config(show='',state='normal') 
                             if self.frequency.get()=='Custom Frequency:' 
                             else entry.config(show=' ',state='disabled'))
        tk.Label(freq_f, text="days/period").pack(side="left")
        # Simulation setting in 2nd row
        self.nIter.Entry(center_f, width=7).grid(row=2,column=1,sticky='w',padx=padx)
        ttk.Button(center_f, text="Run Simulation", command=self.button_RunSimulation
                  ).grid(row=2,column=2,sticky='w')
        
        # Schedule Cost Only Display | outputPage
        ttk.Button(center_f,text="Schedule Only",command=lambda: self.button_ModuleOnly(
            "Schedule")).grid(row=ns[1]+1,column=1,sticky='w',padx=padx)
        ttk.Button(center_f,text="Cost Only",command=lambda: self.button_ModuleOnly(
            "Cost")).grid(row=ns[1]+1,column=2,sticky='e',padx=padx)
        
        # StatisticsFigure Display | outputPage
        # Choose Data Set in 1st row
        n = ns[2]; texts = ['With Risk Events','Without Risk Events']
        ListVar.Radiobuttons(center_f,texts,self.withRisk,[True,False],padx=padx,
                             **border).grid(row=n+1,column=1,columnspan=3,sticky='w')
        # Statistics in 2nd row
        ttk.Entry(center_f, textvariable=self.periodN, width=4
                 ).grid(row=n+2,column=1,sticky='w',padx=padx)
        ttk.Button(center_f, text="Statistics", command=self.button_StatsbyPeriod
                  ).grid(row=n+2,column=2,sticky='w')
        ttk.Button(center_f, text="PDF & CDF", command=lambda: \
                   self.button_StatsbyPeriod('Figure')).grid(row=n+2,column=3)
        # choose stats and Plot in 3rd row
        texts = ['Min','Max','Mean','Percentiles']
        self.stats.CheckButtons(center_f,texts=texts,**border).grid(
            row=n+3,column=1,columnspan=2,sticky='w')
        ttk.Button(center_f, text="Plot", command=self.button_cumCostStatsPlot
                  ).grid(row=n+3,column=3,padx=padx)
        
        # Data Browse | outputPage
        # Choose Data Set in 1st row
        n = ns[3]; texts = ['With Risk Events','Without Risk Events']
        ListVar.Radiobuttons(center_f,texts,self.withRisk_e,[True,False],padx=padx,
                             **border).grid(row=n+1,column=1,columnspan=3,sticky='w')
        # Choose Variables in 2nd row
        self.vars.CheckButtons(center_f,**border).grid(
            row=n+2,column=1,columnspan=3,sticky='w')
        # Choose Export Directory in 3rd row
        ttk.Button(center_f,text="Display / Save", command=self.button_Browse, width=30
                  ).grid(row=n+3,column=0,columnspan=4, padx=padx)
                  
        # Swithing Page Button | outputPage
        ttk.Button(bottom_f, text="Back", command=lambda: self.button_Back(
            'outputPage','filePage')).pack(side="right",padx=padx)
    
    #############################################
    # GUI button functions | filePage
    @Module_GUI.showerror
    def button_Next(self):
        # fetch file directory
        filedir = self.input_dir.get()
        assert filedir,self.i_fileName+" file directory not specified!"
        # initialzied schedule cost obj and raise next page
        self.SCobj = ScheduleCost(filedir)
        self.outputPage.tkraise()
        
    #############################################
    # GUI button functions | outputPage
    @Module_GUI.showerror
    @Module_GUI.show_status(timeit=True,msg=("Completed!","Simulation is done."))
    def button_RunSimulation(self):
        # process parameters: pass frequency&customizedDaysPerPeriod
        self.SCobj.setup_integrationParams(self.frequency.get(),self.custom_freq.get())
        # run simulation & calculate simulated cost results
        assert self.nIter.get()>0, "Number of Iterations should be positive number!"
        self.runtime_updatestatus("simulating random variables...")
        self.SCobj.simulate(self.nIter.get())
        self.runtime_updatestatus("calculating simulated costs...")
        self.SCobj.calculate_simresults()
        # rewrite Flags and update entry
        self.simulationFlag,self.paramFlag = True,False
        self.periodN.set(self.SCobj.get('cumCost',self.withRisk.get()).shape[1])
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_ModuleOnly(self,mod='Schedule'):
        # setup toplevel window and frame container
        window = tk.Toplevel()
        window.resizable(width=True, height=True)
        window.title(mod+' Only Results') # title at window border
        container = tk.Frame(window)
        container.pack(side="top", fill="both", expand=True, ipadx=75)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        # init UIobj and remove simulation setting section & Back button
        UIobj = globals()[mod+'_GUI'](container,self.root,omitFilePage=True)
        center_f = [f for f in UIobj.outputPage.pack_slaves() 
                    if f.pack_info()['side']=="top" and f.pack_info()['fill']=="y"][-1]
        _ = [w.grid_forget() for row in range(3) for w in center_f.grid_slaves(row=row)]
        # modify UIobj.DPobj and enable simulationFlag
        UIobj.simulationFlag = True
        UIobj.DPobj = self.SCobj.copy(to=mod) # copy variables to new DPobj
        UIobj.button_Next() # insert content to dropdown list
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_StatsbyPeriod(self,out='StatsTable'):
        withrisk = self.withRisk.get()
        cumCost = self.SCobj.get('cumCost',withrisk)
        maxPeriodN,periodN = cumCost.shape[1],self.periodN.get()
        assert periodN>0 and periodN<=maxPeriodN, \
            f"Period # entered exceed boundary(1-{maxPeriodN})!"
        # linking to tree table / PDFCDF Plot
        title = f"Project Cost (with{'' if withrisk else 'out'} Risk Events) " \
                f"as of {self.SCobj.periodUnit} #: {periodN}"
        data = cumCost[:,periodN-1]
        if out == 'Figure': 
            plt_pdfcdf(data,title=title,xlabel='Project Cost',xfmt=self.SCobj.fmt)
        elif out == 'StatsTable':
            get_statstable(data, title, self.statsFmt)
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    def button_cumCostStatsPlot(self):
        statsNames = self.stats.get()
        assert statsNames, "Select at least one Statistics to Plot"
        self.SCobj.plot_cumCostStats(statsNames, self.withRisk.get())
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag','paramFlag')
    @Module_GUI.show_status(timeit=True)
    def button_Browse(self):
        sheets = self.vars.get()
        assert sheets, "Select at least one Variable to Browse Data"
        self.SCobj.browse(sheets,self.stats.get(),self.withRisk_e.get())


class DecisionTree_GUI(Module_GUI):
    """
    ScheduleCost_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Decision Tree'
    frames = ['filePage','outputPage']
    statsFmt = lambda _,x: f"-$ {-x:,.0f}" if x<0 else f"$ {x:,.0f}" # fmt in statsTable
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'Decision Tree'
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {
        'filePage': [],
        'outputPage': ['checkparam','simulationFlag','SCobj']}
    # default params value, used in initializationa and clear value
    defaultValue = {'checkparam':False,
                    'simulationFlag': False}
    # tk.Variables for Module, used in initialization only, won't auto clear value
    tkVariable = {'input_dir': (tk.StringVar, ''), 
                  'nIter':(PositiveIntVar, 1000), 'distId': (IdVar,),
                  'withSimul': (tk.BooleanVar,False),
                  'MaxValue':(tk.BooleanVar,True)}
                  
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['distId'],'w','traceDistId')]
   
    # requisites dict, used in decorator self.check_requisites
    requisites = {   
        'simulationFlag': (True, "Simulation not runned")}
    def __init__(self,frame_format,root,**testkwargs):
        Module_GUI.__init__(self,frame_format,root)  
        # using default filePage frame
        self._construct_filePage_frames(frame_format)
        # construct module frames
        self._construct_module_frames(frame_format,**testkwargs)
    
    def traceDistId(self,*args):
        "update distribution text and disable stats/plot button"
        dist = self.SCobj.distribution[self.distId.get()-1].__name__
        self.distId.label.set(dist) # add distribution name to label widge
        state = 'disabled' if dist in ('Project Duration','No Distribution') else 'normal'
        self.distId.widges[-1]['state'] = state   
#    def tkraise(self):
#        self.outputPage.tkraise()
        
    #############################################
    # GUI frames functions
    def _construct_module_frames(self,frame_format,**testkwargs):
        #########################################
        # outputPage
        title = self.__name__
        #msg = "Check the distributions of the individual project to confirm that they are correct.\nThen input 'Number of Iterations' and click on 'Run Simulation'."
        subtitles = ["Simulation Settings",
                     "Decision Node Distribution Display ",
                     "Decision Tree Results Display",
                     "Simulated Values - Display / Save"]
        texts=[['Number of Iterations'],['ID: Decision Node'],['','','','',''],['']]
#        
        testMode = False
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {} 
        frameWidth,padx,pady,width = 550,9,5,11
        
        # construct page, frames, titleframe, status text
        self.outputPage,_,center_f,bottom_f = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,testMode=testMode)
        # create subtitle&first column of each row as tk.Lable, and seperator
        ns = self.gridframe_constructer(center_f,subtitles,texts,colN=5,separator=True)        
        
                # Simulation setting in 1nd row
        ttk.Entry(center_f, textvariable=self.nIter,width=width
                 ).grid(row=1,column=1,columnspan=2)
        ttk.Button(center_f, text="Run Simulation",command=self.button_RunSimulation).grid(row=1, column=3, columnspan=2)
        # Decision Node Checking & Simulation Settings| outputPage
        
        # Decision Node No. setting in 1st row
        # Task Distribution Display in 1st row
        n = ns[1]
        tk.Label(center_f,text='').grid(row=n+1,column=1,padx=65)
        self.distId.Combobox(center_f,width=15).grid(row=n+1,column=0,columnspan=2,sticky='e')
        self.distId.Label(center_f).grid(row=n+1,column=2,sticky='ew',padx=2)
        self.distId.Button(center_f,text="Statistics",command=self.button_TaskStats
                          ).grid(row=n+1,column=3)#,sticky='w',padx=padx)
        self.distId.Button(center_f,text="Distribution",command=self.button_TaskPlot
                          ).grid(row=n+1,column=4)#,sticky='w',padx=padx)
        

        # StatisticsFigure Display | outputPage
        # Choose Data Set in 1st row
        n = ns[2]; texts = ['Stochastic','Deterministic']
        ListVar.Radiobuttons(center_f,texts,self.withSimul,[True,False],padx=2,**border
                            ).grid(row=n+1,column=0,columnspan=2)#,sticky='e')
        text = ['Maximum Value','Minimum Value']
        ListVar.Radiobuttons(center_f,text,self.MaxValue,[True,False],padx=2,**border
                            ).grid(row=n+1,column=2,columnspan=3)
        snb=ttk.Button(center_f, text="Decision Tree", 
                   command=lambda:self.button_TreeGraph('w',self.MaxValue.get()),state='normal',width=20)
        snb.grid(row=n+2,column=0,columnspan=2)#,sticky='e',padx=20)
#        
        swb=ttk.Button(center_f, text="Decision Tree Without Sunk Cost", width=30,
                   command=lambda:self.button_TreeGraph('n',self.MaxValue.get()),state='normal')
        swb.grid(row=n+2, column=2, columnspan=3)#,sticky="w")
#       
        pdf=ttk.Button(center_f, text="PDF & CDF Plots", 
                   command=lambda:self.SCobj.pdfcdf_button(self.expectvalue,title="Expected Value",xlabel="Expected Value"),state='disabled',width=20)
        pdf.grid(row=n+3,column=0,columnspan=2)#,sticky='e',padx=20)
#        
        sat=ttk.Button(center_f, text="Expected Value Statistics", width=30,
                   command=lambda:get_statstable(self.expectvalue, "Expected Value",fmt=self.statsFmt),state='disabled')
        sat.grid(row=n+3, column=2, columnspan=3)#,sticky="w")
        def switch_state(*arg):
            withSimul= self.withSimul.get()
            if withSimul ==True and self.simulationFlag==True:
                [b.config(state='normal') for b in [snb,swb,pdf,sat]]
            elif withSimul ==True and self.simulationFlag==False:
                [b.config(state='disabled') for b in [snb,swb,pdf,sat]]
            elif withSimul ==False and self.simulationFlag==True:
                [b.config(state='normal') for b in [snb,swb]]
                [b.config(state='disabled') for b in [pdf,sat]]
            else:
                [b.config(state='normal') for b in [snb,swb]]
                [b.config(state='disabled') for b in [pdf,sat]]
           
        self.withSimul.trace('w',switch_state)
        
        
         # Data Export | outputPage
        # Choose Data Set in 1st row
        n = ns[3]; 
#        
        ttk.Button(center_f,text="Display / Save", command=self.button_Export, width=30
                  ).grid(row=n+1,column=0,columnspan=5)
                  
        # Swithing Page Button | outputPage
        ttk.Button(bottom_f, text="Back", command=lambda: self.button_Back(
            'outputPage','filePage')).pack(side="right",padx=padx)
        
    #############################################
    # GUI button functions | filePage
    @Module_GUI.showerror
    def button_Next(self):
        # fetch file directory
        filedir = self.input_dir.get()
        if not filedir:
            raise ValueError(self.i_fileNames[0]+" file path not specified!")
        # initialzied schedule cost obj and raise next page
        self.SCobj = DecisionTree(filedir)
        # dynamically update dropdown list
        content = [(f"{d.id}: {d.name}",d.id) for d in self.SCobj.decision]
        self.distId.add_ComboboxContent(text2id=content)
        self.outputPage.tkraise()
        
    #############################################
    # GUI button functions | outputPage
    @Module_GUI.showerror
    def button_errortmp(self):
        raise ValueError("raise temp error!")
        
    @Module_GUI.showerror
    def button_RunSimulation(self):
        self.SCobj.simulate(self.nIter.get())
        self.expectvalue=self.SCobj.simul_calculate()
        self.simulationFlag = True
        #self.withSimul=True
        messagebox.showinfo("Hey", "Simulation is done.")
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag')
    def button_TaskStats(self):
        
        title = self.SCobj.name[self.distId.get()-1]+'; Distribution: '+self.SCobj.distribution[self.distId.get()-1].__name__
        get_statstable(self.SCobj.simcost_value[self.distId.get()-1], title,fmt=self.statsFmt)
    
    def button_TaskPlot(self):
        self.SCobj.pdf_button(self.distId.get())
        
    @Module_GUI.showerror
    def button_TreeGraph(self,opera,minflag):
        if self.withSimul.get() == True:
            simu='w'
            self.SCobj.graphprepare(simu,opera,minflag)
        else: 
            simu='n'
            self.SCobj.graphprepare(simu,opera,minflag)
    
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag')
    def button_Export(self):
        
        self.SCobj.export(self.expectvalue)


class Optimization_GUI(Module_GUI):
    """
    Optimazision_GUI class for interface file that inherated from module file 
    """
    # ------------- Module configs -------------
    __name__ = 'Portfolio Optimization'
    frames = ['filePage','outputPage']
    
    # ----------- Module GUI I/O file ----------
    i_fileName = 'Portfolio'
    
    # ----------- Module GUI configs -----------
    # params list for each tk.Frame, used in clear params
    params_byFrame = {'filePage': [],'outputPage': ['simulationFlag','constraintList']}
    # default params value, used in initializationa and clear value
    defaultValue = {'simulationFlag': False,
                    'constrains': []} # list of user defined constriant
    # tk.Variables for Module, used in initialization only, won't auto clear value
    tkVariable = {'input_dir': (tk.StringVar, ''), 
                  'nIter':(PositiveIntVar, 1000), 'distId': (IdVar,),
                  'withSimul': (tk.BooleanVar,False),'ConstVar':(tk.BooleanVar,False),'consts': (tk.StringVar,''),
                  'investc':(PositiveIntVar,), 'resourcec': (PositiveIntVar,), 
                  'varentry1':(tk.StringVar,''),'varentry2':(tk.StringVar,'')}
                  
    # tk Variables to trace, will call functions once value is write/read/undefine
    var2Trace = [(['distId'],'w','traceDistId')]#,(['ConstVar'],'w','button_disable')]
   
    # requisites dict, used in decorator self.check_requisites
    requisites = {   
        'simulationFlag': (True, "Simulation not runned")}
    tableFmt = Optimization.fmt
    def __init__(self,frame_format,root,**testkwargs):
        Module_GUI.__init__(self,frame_format,root)  
        # using default filePage frame
        self._construct_filePage_frames(frame_format)
        # construct module frames
        self._construct_module_frames1(frame_format,**testkwargs)
        # construct module frames
        #self._construct_module_frames2(frame_format,**testkwargs)
    
    def traceDistId(self,*args):
        "update distribution text and disable stats/plot button"
        dist = self.SCobj.projects[self.distId.get()].npvDist.__name__
        self.distId.label.set(dist) # add distribution name to label widge
        state = 'disabled' if dist in ('Project Duration','No Distribution') else 'normal'
        self.distId.widges[-1]['state'] = state   
    
    #############################################
    # GUI frames functions
    def _construct_module_frames1(self,frame_format,**testkwargs):
        #########################################
        # outputPage
        title = "Portfolio Optimization"
        subtitles = ["Constraints & Simulation Settings",
                     "Project Distribution Display ",
                     "Portfolio Optimization Results Display",
                     "Simulated Values - Display / Save"]
        texts=[['Constraints:','','','',''],['Project ID: '],['','','Optimum Portfolio NPV'],['']]
        testMode = False
        border = {'bd':1,'highlightbackground':'black','highlightthickness':1
                 } if testMode else {} 
        frameWidth,padx,pady,width = 550,9,5,11
        
        # construct page, frames, titleframe, status text
        self.outputPage,_,center_f,bottom_f = self.pageframe_constructer(
            frame_format,Nframes=3,title=title,testMode=testMode)
        # create subtitle&first column of each row as tk.Lable, and seperator
        ns = self.gridframe_constructer(center_f,subtitles,texts,colN=5,separator=True)        
        
        tk.Label(center_f,text='Investment:').grid(row=1,column=1)
        ttk.Entry(center_f,textvariable=self.investc,width=width).grid(row=1,column=2)
        tk.Label(center_f,text='Resources:').grid(row=1,column=3)
        ttk.Entry(center_f,textvariable=self.resourcec,width=width).grid(row=1,column=4)
        
        #Set other Resource on Second line
        texts = ['Fixed','Either, or','If, then']
        constr_f = ListVar.Radiobuttons(center_f,texts,self.consts,padx=padx,pady=pady,**border)
        constr_f.grid(row=2,column=2,columnspan=3,sticky='w')
        tk.Label(center_f,text='Entry:').grid(row=3,column=1)
        self.tbEntry1=tk.Entry(center_f,textvariable=self.varentry1,width=width,state='disabled')
        self.tbEntry1.bind('<Return>',lambda _: self.add_symbol())
        self.tbEntry1.grid(row=3, column=2)#, sticky='w')
        self.consts.trace('w',lambda *args: self.tbEntry1.config(state='disabled') if self.consts.get()=='' 
                             else self.tbEntry1.config(state='normal'))
        
        self.btnAddSymbal=ttk.Button(center_f, text="Add", command=self.add_symbol,width=width)
        self.btnAddSymbal.grid(row=3, column=3)#, sticky='w')
        self.btnDeleteSymbol=ttk.Button(center_f,text="Remove",command=self.delete_symbol,width=width)
        self.btnDeleteSymbol.grid(row=3,column=4)#,sticky='w')
        
        scrollbar_V=tk.Scrollbar(center_f)
        scrollbar_H=tk.Scrollbar(center_f,orient='horizontal')
        scrollbar_V.grid(row=2,column=0,rowspan=2,sticky='nes')
        scrollbar_H.grid(row=4,column=0,sticky='nesw')
      
        # listbox wideget
        self.listbox1=tk.Listbox(center_f,width=14,height=4,selectmode='extended',
                                 yscrollcommand=scrollbar_V.set,xscrollcommand=scrollbar_H.set)
        self.listbox1.bind('<<ListboxSelect>>',self.select_item)
        self.listbox1.bind('<Delete>',lambda _: self.delete_symbol())
        self.listbox1.grid(row=2,column=0,rowspan=3,sticky='nesw',padx=4)
        scrollbar_V.config(command=self.listbox1.yview)
        scrollbar_H.config(command=self.listbox1.xview)
        tk.Label(center_f,text='Number of Iterations:').grid(row=5,column=0)
        ttk.Entry(center_f, textvariable=self.nIter,width=width
                 ).grid(row=5,column=1)
        ttk.Button(center_f, text="Run Simulation",command=self.button_RunSimulation).grid(row=5, column=2, columnspan=2)
        
        # Portfolio Checking & Simulation Settings| outputPage
        # Portfolio ID setting in 1st row
        # Task Distribution Display in 1st row
        n = ns[1]
        tk.Label(center_f,text='').grid(row=n+1,column=1,padx=65)
        self.distId.Combobox(center_f,width=15).grid(row=n+1,column=0,columnspan=2,sticky='e')
        self.distId.Label(center_f).grid(row=n+1,column=2,sticky='ew',padx=2)
        self.distId.Button(center_f,text="Statistics", width=width,command= self.button_TaskStats ).grid(row=n+1,column=3)#,sticky='w',padx=padx)
        self.distId.Button(center_f,text="Distribution", width=width,command= self.button_TaskPlot ).grid(row=n+1,column=4)#,sticky='w',padx=padx)
        
        # StatisticsFigure Display | outputPage
        # Choose Data Set in 1st row and set constraint
        n = ns[2]; 
        # Add Optimazition Result
        texts = ['Stochastic','Deterministic']
        ListVar.Radiobuttons(center_f,texts,self.withSimul,[True,False],padx=20,**border).grid(row=n+1,column=1,columnspan=3)
        resultbuton=ttk.Button(center_f, text="Optimization Results", command=self.button_Result,state='normal')
        resultbuton.grid(row=n+2,column=1,columnspan=2)
        efbutton=ttk.Button(center_f,text="Efficient Frontier",command=self.button_curve,state='normal')
        efbutton.grid(row=n+2,column=3,columnspan=1)
        pcbutton=ttk.Button(center_f, text="PDF & CDF Plots",command=self.pdf_cdfbutton,state="diabled")
        pcbutton.grid(row=n+3,column=1)
        statisbutton=ttk.Button(center_f, text="Statistics",command=self.stats_button,state="diabled")
        statisbutton.grid(row=n+3,column=2)
        #block PDF&CDF and Statistics for optimum NPV when click Deterministic
        def switch_state(*arg):
            if self.withSimul.get() ==False and self.simulationFlag==False:
                resultbuton.config(state='normal') 
                efbutton.config(state='normal')
                pcbutton.config(state='disabled')
                statisbutton.config(state='disabled')
            elif self.withSimul.get() ==True and self.simulationFlag==False:
                resultbuton.config(state='disabled') 
                efbutton.config(state='disabled')
                pcbutton.config(state='disabled')
                statisbutton.config(state='disabled')
            elif self.withSimul.get() ==False and self.simulationFlag==True: 
                resultbuton.config(state='normal')
                efbutton.config(state='normal')
                pcbutton.config(state='disabled')
                statisbutton.config(state='disabled')
            elif self.withSimul.get() ==True and self.simulationFlag==True: 
                resultbuton.config(state='normal') 
                efbutton.config(state='normal') 
                pcbutton.config(state='normal')
                statisbutton.config(state='normal')
        self.withSimul.trace('w',switch_state)
        # Data Export | outputPage
        # Choose Data Set in 1st row
        n = ns[3]
        ttk.Button(center_f,text="Display / Save", command=self.button_Export, width=30
                  ).grid(row=n+1,column=0,columnspan=5)
        # Swithing Page Button | outputPage
        ttk.Button(bottom_f, text="Back", command=lambda: self.button_Back('outputPage','filePage')).pack(side="right",padx=padx)
    
    def select_item(self,event):
        pass
        
    @Module_GUI.showerror
    def add_symbol(self):
        # read and parse user defined projects & mode
        inputs = self.varentry1.get().strip()
        if not inputs: return # break if no entry
        errormsg = "Invalid Entry!\nExpect projects seperated by ',', "\
                   f"but got '{inputs}' instead.\n"
        projects = inputs.split(",") # projects
        for p in projects:
            if p not in self.SCobj.ids:
                raise ValueError(errormsg+f"Project '{p}' not found!")
        mode = self.consts.get()
        
        if mode == 'Fixed':
            self.constrains.append((0,projects)) # add to constrains list
            self.listbox1.insert('end','Fixed '+','.join(projects)) # display in listbox
        elif mode == 'Either, or':
            self.constrains.append((1,projects)) # add to constrains list
            self.listbox1.insert('end','Either '+' or '.join(projects)) # display in listbox
        elif mode == 'If, then':
            assert len(projects)==2, errormsg+"Only accept 2 projects for 'If,then'"
            self.constrains.append((2,projects)) # add to constrains list
            self.listbox1.insert('end',f'If {projects[0]} then {projects[1]}') # display in listbox
        else:
            raise ValueError(f"Incorrect Mode: {mode}")
        self.tbEntry1.delete(0,'end') # clear entry
    
    @Module_GUI.showerror
    def delete_symbol(self):
        "delete user select item in listbox"
        current_selection = self.listbox1.curselection()
        # modify listbox and constrain list
        _ = [(self.listbox1.delete(i), self.constrains.pop(i)) for i in 
             sorted(current_selection, reverse=True)]
        
    def input_data(self):
        "analyze user defined constraints and optimize if needed"
        tmp = [[] for _ in range(3)] # tmp list for process constrain list
        _ = [tmp[i].append(projects) for i,projects in self.constrains]
        
        capacity = [self.investc.get(),self.resourcec.get()]
        fixed = sorted(set(sum(tmp[0],[])))
        
        self.SCobj.setup_Constraints(capacity,fixed,tmp[1],tmp[2])
        
    def result_table(self,Simulflag,title,fmt="$ {x:,.0f}",header=None):
        """
        Display Optimized Results table to user with args table title and data diplay format
        """
        # Create 4 colums of results
        result=[['Projects Selected','NPV','Investment','Number of Resources']]
        if Simulflag==True:
            npv=int(self.SCobj.totalNPVs_stoch)
            invest=self.SCobj.totalWeight_stoch[0]
            resource=self.SCobj.totalWeight_stoch[1] if len(self.SCobj.capacity)==2 else ''
            result_index=self.SCobj.port_stoch 
            title='Stochastic '+title
        else:
            npv=int(self.SCobj.totalNPVs_deter)
            invest=self.SCobj.totalWeight_deter[0]
            resource=self.SCobj.totalWeight_deter[1] if len(self.SCobj.capacity)==2 else ''
            result_index=self.SCobj.port_deter
            title='Deterministic '+title
        for i in (result_index):
            result +=[[self.SCobj.names[i],self.tableFmt(self.SCobj.projNPV_deter[i]),\
                      self.tableFmt(self.SCobj.constraints[i][0]),self.SCobj.constraints[i][1] if len(self.SCobj.capacity)==2 else '']]
        result +=[['Optimized Portfolio:',self.tableFmt(npv),self.tableFmt(invest),resource]]
        tree_result, window_result=make_tkTreeview(result, title, show='tree',  anchor='center')
        return tree_result, window_result
    
    def constraints_table(self,window_result,title,header=None):
        """
        Display Constraints enterned in Interface
        """
        capacity,fixed,eitheror,ifthen = self.SCobj.get_Constraints()
        row1 = ["Portfolio Investment:", self.tableFmt(capacity[0]),'','']
        if len(capacity)>1: row1[2:] = ['Portfolio Resources:', ','.join(map(str,capacity[1:]))]
        table = [row1,['Fixed','Either, or','If, then','']]
        maxLen = max(1,len(eitheror),len(ifthen))
        fixed = [",".join(fixed)]+['']*(maxLen-1)
        eitheror = [",".join(eo) for eo in eitheror]+['']*(maxLen-len(eitheror))
        ifthen = [",".join(it) for it in ifthen]+['']*(maxLen-len(ifthen))
        table += list(zip(fixed,eitheror,ifthen))
        
        make_tkTreeview(table, title, show='tree',  anchor='center',window=window_result)
        
    @Module_GUI.showerror
    def button_Next(self):
        # fetch file directory
        filedir = self.input_dir.get()
        assert filedir,self.i_fileName+" file directory not specified!"
        # initialzied schedule cost obj and raise next page
        self.SCobj = Optimization(filedir)
        # dynamically update dropdown list
        content = [(f"{s.id}: {s.name}",i) for i,s in enumerate(self.SCobj.projects)]
        self.distId.add_ComboboxContent(text2id=content)
        self.outputPage.tkraise()

    @Module_GUI.showerror
    def button_RunSimulation(self):
        self.input_data()
        self.SCobj.simulate(self.nIter.get())
        self.simulationFlag = True
        messagebox.showinfo("Hey", "Simulation is done.")
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag')
    def button_TaskStats(self):
        task = self.SCobj.projects[self.distId.get()]
        title = f"Project {task.id}: {task.name}; Distribution: {task.npvDist.__name__}"
        get_statstable(self.SCobj.NPVs_stoch[self.distId.get()],title,fmt=self.tableFmt)
        
    def button_TaskPlot(self):
        task = self.SCobj.projects[self.distId.get()]
        xlabel = "Profits"
        title = f"Project {task.id}: {task.name}\n{task.npvDist.__name__} Distribution"
        plt_pdfcdf(task.npvDist,xlabel=xlabel,title=title,xfmt=self.SCobj.fmt)
        
    def button_Result(self):
        self.input_data()
        _, window_result=self.result_table(self.withSimul.get(),'Results',fmt="$ {x:,.0f}",header=None)
        self.constraints_table(window_result,'Constraints',header=None)
    def button_curve(self):
        self.input_data()
        self.SCobj.plot_step(self.withSimul.get()) 
        
    @Module_GUI.showerror
    def pdf_cdfbutton(self):
        self.SCobj.plot_pdfcdf()
    @Module_GUI.showerror
    def stats_button(self):
        get_statstable(self.SCobj.portNPVs,"Optimum Portfolio NPV",fmt=self.tableFmt)
        
    @Module_GUI.showerror
    @Module_GUI.check_requisites('simulationFlag')
    def button_Export(self):
        self.SCobj.export() 


if __name__ == "__main__":
    hawk = App()
    hawk.mainloop()
