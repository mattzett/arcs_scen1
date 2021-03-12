#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:37:54 2021

Perform some numerical operations needed for scenario 1 estimation

@author: zettergm
"""


# imports
import numpy as np


# two dimensional divergence
def div2D(Ux,Uy,x,y):
    [dUxdx,_]=np.gradient(Ux,x,y)
    [_,dUydy]=np.gradient(Uy,x,y)
    return dUxdx+dUydy