#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 4.21
#  in conjunction with Tcl version 8.6
#    Mar 13, 2019 10:07:02 AM IST  platform: Linux
#    Mar 13, 2019 02:28:10 PM IST  platform: Linux
#    Mar 13, 2019 02:37:57 PM IST  platform: Linux

import sys
import second_win

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def No():
    destroy_window()

def Yes():
    destroy_window()
    root = tk.Tk()
    root.withdraw()
    second_win.create_second_win_Yes(root)
    

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import first_win
    first_win.vp_start_gui()



