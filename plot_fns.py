#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:00:35 2021

Contains plotting functions for script testing scenario 1 data processing

@author: zettergm
"""

import matplotlib.pyplot as plt


def plotSigmaP_debug(mlon,mlat,mlonp,mlatp,Spar,Eperp,dissipation,int_ohmic_ref, \
                     SigmaP_ref,SigmaP_refi,magE2):
    # plot input quantities
    plt.subplots(1,3,dpi=100)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(mlon,mlat,Spar)
    plt.title("$S_\parallel$")
    plt.colorbar()
    
    plt.subplot(1,3,2)
    plt.pcolormesh(mlon,mlat,Eperp[:,:,1])
    plt.title("$E_2$")
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.pcolormesh(mlon,mlat,Eperp[:,:,2])
    plt.title("$E_3$")
    plt.colorbar()
    plt.show()
    
    
    # plot
    plt.subplots(2,2,dpi=100)
    
    plt.subplot(2,2,1)
    plt.pcolormesh(mlon,mlat,dissipation) # note lack of transposition on interpolated quantities
    plt.title("Ohmic disspitation")
    plt.colorbar()
    
    plt.subplot(2,2,2)
    plt.pcolormesh(mlon,mlat,int_ohmic_ref) # note lack of transposition on interpolated quantities
    plt.title("Ohmic disspitation (2)")
    plt.colorbar()
    
    plt.subplot(2,2,3)
    plt.pcolormesh(mlonp,mlatp,SigmaP_ref)
    plt.title("Reference Pedersen")
    plt.colorbar()
    
    plt.subplot(2,2,4)
    plt.pcolormesh(mlon,mlat,SigmaP_refi)
    plt.title("Reference Pedersen (interp.)")
    plt.colorbar()
    plt.show()
    
    
    # more debug plots
    plt.subplots(1,2,dpi=100)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(mlon,mlat,magE2)
    plt.title("$E^2$")
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.pcolormesh(mlon,mlat,Spar/magE2)
    plt.title("$S/E^2$")
    plt.colorbar()
    plt.show()