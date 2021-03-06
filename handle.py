# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:17:01 2017

@author: Jelena
"""
import Tkinter as tk
import sys, os, ttk
from PIL import ImageTk, Image

def visual(self, name):
       
    window = tk.Tk()
    window.minsize(width=800, height=450)
    window.title("Photomath")
    window.configure(background='White')
    
    size = 800, 300
    path = os.getcwd() + name
    slika = Image.open(path)
    slika.thumbnail(size, Image.ANTIALIAS)
    img = ImageTk.PhotoImage(slika)
    panel = ttk.Label(window, image=img)
    panel.grid(row=0, column=0, rowspan=3,columnspan=3)
    
    txt = ttk.Label(window, text="REZULTAT:",background="white", font=('Helvetica', '20'))
    txt.grid(row=4, column=0)
    lhd, rhs = name.split("\\", 1)
    from PhotoMath import proces
    ttt = proces(self,rhs)
    #ttt = 45
    tx = ttk.Label(window, text=ttt,background="white", font=('Helvetica', '20'))
    tx.grid(row=4, column=1)
    
    s = ttk.Style()
    s.configure('TButton', background="white", font=('Helvetica', '20'))
    b = ttk.Button(window, text='Quit', style='TButton',command=sys.exit)
    b.grid(row=4, column=2)
    window.mainloop()
        