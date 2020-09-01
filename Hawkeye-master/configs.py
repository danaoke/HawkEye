#!/usr/bin/env python
# coding: utf-8

import sys,os

__all__ = ['VERSION','HOME','OS','UUID','TITLE_FONT','HEADR_FONT','SUBHEAD_FONT',
           'MESSG_FONT','STATUS_FONT','TABLE_FONT']


VERSION = "3.0.0"

# hawkeye code home path for import and PyInstaller
if getattr(sys, 'frozen', False):
    HOME = sys._MEIPASS
elif __file__:
    HOME = os.path.dirname(__file__)

# operation system
OStmp = sys.platform.lower()
if OStmp.startswith('win'):
    OS = "Windows"
elif OStmp.startswith('darwin'):
    OS = "Mac OS"
else:
    raise ImportError(f"Platform {sys.platform} is not supported!")

# computer UUID for software validation
if OS == "Windows":
    cmd = 'wmic csproduct get uuid'
elif OS == "Mac OS":
    cmd = "system_profiler SPHardwareDataType | awk '/UUID/ { print $3; }'"
UUID = os.popen(cmd).read().split()[-1]


# ------------- GUI general configs-------------
# if OS == "Windows":
TITLE_FONT = ("Times New Roman", 24)
HEADR_FONT = ("Times New Roman", 16)
SUBHEAD_FONT = ("Times New Roman", 10, "bold")
MESSG_FONT = ("Arial", 10)
STATUS_FONT = ("Courier New",9)
TABLE_FONT = {'title':('Calibri',20,'bold'),'header':('Calibri',15,'bold'),
              'body':('Calibri',15)}
