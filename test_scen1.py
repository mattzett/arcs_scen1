#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 06:02:51 2021

Load GEMINI output corresponding to synthetic Poynting flux, current density (parallel),
 and electric field (perp.).  Attempt to process the data into conductances

@author: zettergm
"""

# imports
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import scipy.interpolate as sciint


# Load synthetic data maps and organize data
filename="/Users/zettergm/Dropbox (Personal)/shared_simulations/arcs/scen1.mat"
data=spio.loadmat(filename)
E=np.asarray(data["E"],dtype="float64")
Jpar=np.asarray(data["Jpar"],dtype="float64")
Spar=np.asarray(data["Spar"],dtype="float64")
mlon=np.asarray(data["mlon"],dtype="float64")
mlon=mlon.squeeze()
mlat=np.asarray(data["mlat"],dtype="float64")
mlat=mlat.squeeze()
SigmaP_ref=np.asarray(data["SIGP"],dtype="float64")
SigmaH_ref=np.asarray(data["SIGH"],dtype="float64")
mlonp=np.asarray(data["mlonp"],dtype="float64")
mlonp=mlonp.squeeze()
mlatp=np.asarray(data["mlatp"],dtype="float64")
mlatp=mlatp.squeeze()
int_ohmic_ref=np.asarray(data["int_ohmic"]) # this computed via integration of 3D dissipation
ohmic_ref=np.asarray(data["ohmici"])


# Try to convert Spar to conductance, using steady-state integrated Poynting thm.
Eperp=E.squeeze()
magE2=np.sum(Eperp**2,axis=2)
SigmaP=-Spar/magE2


# Recompute Ohmic dissipation (field-integrated) as a test
[MLON,MLAT]=np.meshgrid(mlon,mlat)
SigmaP_refi=sciint.interpn((mlonp,mlatp),SigmaP_ref,(MLON,MLAT))
dissipation=SigmaP_refi*magE2


# plot input quantities
fig=plt.figure(num=1,dpi=300)
plt.subplots(1,3)

plt.subplot(1,3,1)
plt.pcolormesh(mlon,mlat,np.transpose(Spar))
plt.colorbar()

plt.subplot(1,3,2)
plt.pcolormesh(mlon,mlat,Eperp[:,:,1])
plt.colorbar()

plt.subplot(1,3,3)
plt.pcolormesh(mlon,mlat,Eperp[:,:,2])
plt.colorbar()
plt.show()


# plot
plt.figure(num=2,dpi=300)
plt.subplots(2,2)

plt.subplot(2,2,1)
plt.pcolormesh(mlon,mlat,dissipation) # note lack of transposition on interpolated quantities
plt.colorbar()

plt.subplot(2,2,2)
plt.pcolormesh(mlon,mlat,np.transpose(int_ohmic_ref)) # note lack of transposition on interpolated quantities
plt.colorbar()

plt.subplot(2,2,3)
plt.pcolormesh(mlonp,mlatp,np.transpose(SigmaP_ref))
plt.colorbar()

plt.subplot(2,2,4)
plt.pcolormesh(mlon,mlat,SigmaP_refi)
plt.colorbar()
plt.show()


# plot some comparisons
plt.figure(num=3,dpi=300)
plt.subplots(1,2)

plt.subplot(1,2,1)
plt.pcolormesh(mlon,mlat,np.transpose(SigmaP))
plt.colorbar()

plt.subplot(1,2,2)
plt.pcolormesh(mlonp,mlatp,np.transpose(SigmaP_ref))
plt.colorbar()
plt.show()


