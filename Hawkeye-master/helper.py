#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font as tkFont

from configs import *

__all__ = ['dirDoc','dirTheme','tkScaler','XMLTheme_parser','helpClass']


dirDoc = HOME+"\\help_docs\\documentations.xml"
dirTheme = HOME+"\\help_docs\\theme.xml"


class tkScaler(tk.DoubleVar):
    base = 1.0
    delta = 0.1
    ranges = [.2,5]
    def __init__(self,master=None,value=None):
        self._callbacks = []
        super().__init__(master,value or self.base)
        self.trace('w',self.callback)
        # self.bind()
        
    def register(self,func,*defaults):
        '''
        register func to callback when scaled, func take scaled value as arg
        Example: func = lambda x: font.config[size=x]    
        '''
        try:
            for r in self.ranges:
                func(*[round(r*default) for default in defaults])
            func(*defaults)
            self._callbacks.append((func,defaults))
        except Exception as e:
                print(f"{e}, tkScaler register failed!")
        
    def register_fonts(self,fonts):
        if isinstance(fonts,tkFont):
            self.register(lambda x:fonts.config(size=x),fonts.cget("size"))
        elif isinstance(fonts,(list,tuple)):
            _ = [self.register_fonts(font) for font in fonts]
        else:
            raise TypeError(f"Unexpected arg 'fonts' type '{type(fonts)}'")
            
    def callback(self,*args):
        _scale = self.get()
        # call funcs in _callbacks to pass new scaled values
        for func,defaults in self._callbacks:
            func(*[round(_scale*default) for default in defaults])
    
    def Scroll(self,event):
        "cmd for Hotkey Ctrl+Scroll Up/Down, results in zoom in/out"
        if event.delta>0: # Ctrl + Scroll Up, zoom in
            _value = self.get() + self.delta
            if _value <= self.ranges[-1]: self.set(_value)
        elif event.delta<0: # Ctrl + Scroll Down, zoom out
            _value = self.get() - self.delta
            if _value >= self.ranges[0]: self.set(_value)
    
    def bind(self,widge=None):
        "setup hotkeys to zoom in/out, as Ctrl+Scroll Up/Down"
        widge = widge or self._root
        widge.bind("<Control-MouseWheel>", self.Scroll)
        
    def Reset(self):
        self.set(self.base)
    
    def tkFrame(self):
        pass


class XMLTheme_parser(object):
    layoutTagMap = {'marginH':'padx', 'marginV':'pady'}
    styleTagMap = { # map Style node attributes key in XML to new name, if applied
        'fgColor':'foreground', 'bgColor':'background', 'align':'justify',
        'spaceBefore':'spacing1', 'spaceAfter':'spacing3', 'spaceLine':'spacing2',
        'indentLeftline1':'lmargin1', 'indentLeftOthers':'lmargin2', 
        'indentRight':'rmargin'}
    styleTagOmit = ('name','Font','Bullet','prepend','append')
    def __init__(self,XMLdir):
        self.parseFlag = False
        self._parse_XML(XMLdir)
        self.Layout,self.Styles = {},{} # set default attributes
        if self.parseFlag:
            self._parse_Layout()
            self._parse_Fonts()
            self._parse_Bullets()
            self._parse_Styles()
    
    def tagMapper(self,attrib,mapper,omit=None):
        "map dict attributes key to new name, if not omitted"
        return {mapper.get(k,k):v for k,v in attrib.items() if k not in omit} \
             if omit else {mapper.get(k,k):v for k,v in attrib.items()}
    
    def _parse_XML(self,XMLdir):
        "parse XML theme file, use default system theme if failed"
        try:
            self.root = ElementTree.parse(XMLdir).getroot()
            self.parseFlag = True # 
        except Exception as e:
            print(f"{e}, parse XML theme file failed, using system default theme!")
    
    def _parse_Layout(self):
        "parse GeneralLayout node as window display layout, used when init Text widge"
        node = self.root.find("GeneralLayout")
        self.Layout = self.tagMapper(node.attrib, self.layoutTagMap)
        
    def _parse_Fonts(self):
        "parse user specified fonts, used in creating customized Styles"
        self.defaultFont = tkFont(font="TkDefaultFont") # system default font
        self.Fonts = {'default': self.defaultFont}
        for node in self.root.findall("Fonts/Font"):
            attrib = node.attrib
            try:
                self.Fonts[attrib['name']] = tkFont(font=tuple(attrib.values())[1:])
            except Exception as e:
                print(f"{e}, create font '{attrib['name']}' failed, using default font!")
    
    def _parse_Bullets(self):
        self.Bullets = {node.attrib['name']: node.attrib 
                        for node in self.root.findall("Bullets/Bullet")}
        
    def _parse_Styles(self):
        "parse user specified Styles, used in set up tag configs in Text widge"
        for node in self.root.findall("Styles/Style"):
            attrs = node.attrib
            name,font,bullet,prepend,append = [attrs[k] for k in self.styleTagOmit]
            # prepare kwargs passed to text.tag_config
            kwargs = self.tagMapper(attrs,self.styleTagMap,self.styleTagOmit)
            kwargs['font'] = self.Fonts.get(font,self.defaultFont)
            # override kwargs for bullets setting
            Bullet = self.Bullets.get(bullet)
            if bullet:
                kwargs['tabs'] = kwargs['lmargin1'] # tabs = Line 1 indent
                kwargs['lmargin1'] = Bullet['indent'] # Line 1 indent = bullet indent
                prepend = Bullet['symbol'] + '\t' + prepend # \tab for indent placement
            self.Styles[name] = (prepend,append,kwargs)


class helpClass(object):
    def __init__(self):
        self._parse_XMLdoc(dirDoc)
        self._parse_theme(dirTheme)
        self._init_UI()
        self.Text = self._setup_theme() # validate / setup theme in window
    
    def _parse_XMLdoc(self,XMLdir):
        "parse help button documentation XML file"
        self.parsed = False
        try:
            self.doc = ElementTree.parse(XMLdir).getroot()
            self.parsed = True
        except Exception as e:
            print(f"{e}, parse XML doc file failed, help button disabled!")
            
    def _parse_theme(self,XMLdir):
        self.theme = XMLTheme_parser(XMLdir)
    
    def _init_UI(self):
        "initialize tk.Toplevel, add Text widge and applied theme to it"
        self.window = tk.Toplevel()
        self.window.geometry("%dx%d" %(800,800))
        self.window.resizable(height = None, width = None)
        self.window.title("HawkEye Help") # title at window border
        self.window.withdraw() # hide window, call widge.deiconify() to unhide
        self.window.protocol('WM_DELETE_WINDOW', self.window.withdraw)
        
        # init tkScaler binded with window, register fonts to follow scale change
        self.scaler = tkScaler()
        self.scaler.register_fonts(list(self.theme.Fonts.values()))
        self.scaler.bind(self.window)
        
    def _setup_theme(self, frame=None):
        "create text widge and validate / applied theme to Text"
        frame = frame or self.window
        # setup vertical scroll / Text widge
        yScroll = ttk.Scrollbar(frame, orient='vertical')
        yScroll.pack(side='right',fill='y')
        Text = tk.Text(frame,bg=frame.cget('background'),
                       font=self.theme.Fonts['default'],yscrollcommand=yScroll.set)
        Text.pack(expand=True,fill='both')
        try:
            Text.config(**self.theme.Layout)
        except Exception as e:
            print(f"{e}, create Text Widge failed, using default General Layout!")
        yScroll.config(command=Text.yview) # allow drag scroll command
        
        # setup Styles in theme
        for name,(_,_,kwargs) in self.theme.Styles.items():
            try:
                Text.tag_config(name,**kwargs)
            except Exception as e:
                print(f"{e}, Style '{name}' setup failed, using default Style!")
        return Text
    
    def get_node(self, module, page):
        "try fetch page node in XML doc file, raise error if node not found"
        assert self.parsed, "parse XML doc file failed, help button disabled!"
        node = self.doc.find(f"./Module[@id='{module}']./Page[@name='{page}']")
        if node is not None: return node
        raise ValueError(f"Module '{module}' Page '{page}' not found in XML doc file")
    
    def append_content(self,node):
        "add single node text to text widge"
        tagName = node.tag
        prepend,append,_ = self.theme.Styles.get(tagName,('','\n',''))
        content = (prepend+node.text+append).encode().decode('unicode_escape')
        self.Text.insert('end',content,tagName)
        
    def show_doc(self, modPage=None, pageNode=None): 
        "fetch help window content for selected module/page, pop up tk.Toplevel"
        if not isinstance(pageNode,ElementTree.Element):
            # get corresponding content node for given Module & Page
            if hasattr(modPage,'__call__'): # callable function/class method
                modPage = modPage()
            assert isinstance(modPage,(list,tuple)) and len(modPage)==2, \
                "Unexpected arg 'modPage', either list/tuple/function as Module&Page"
            pageNode = self.get_node(*modPage) # fetch content
        
        # switch to writable mode, clear all current content
        self.Text.config(state='normal') 
        self.Text.delete(1.0,'end')
         # add new content to Text widge
        _ = [self.append_content(node) for node in pageNode]
        # switch back to read only mode, and unhide window
        self.Text.config(state='disabled')
        self.window.deiconify()


if __name__ == "__main__":
    root = tk.Tk()
    helper = helpClass()
    for mod in helper.doc:
        modName = mod.get('id')
        frame = tk.LabelFrame(root,text=modName)
        frame.pack(side='top',padx=20,pady=5)
        for page in mod:
            pageName = page.get('name')
            ttk.Button(frame,text=pageName,command=lambda a=modName,b=pageName: \
                       helper.show_doc((a,b))).pack(side='left',padx=10,pady=5)
    root.mainloop()