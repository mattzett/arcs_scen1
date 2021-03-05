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
from plot_fns import plotSigmaP_debug

# setup
plt.close("all")
flagSigP_debug=False


# Load synthetic data maps and organize data, permute/transpose arrays as lat,lon for plotting
#  squeeze 1D arrays for plotting as well
filename="/Users/zettergm/Dropbox (Personal)/shared_simulations/arcs/scen1.mat"
data=spio.loadmat(filename)
E=np.asarray(data["E"],dtype="float64")
Jpar=np.transpose(np.asarray(data["Jpar"],dtype="float64"))
Spar=np.transpose(np.asarray(data["Spar"],dtype="float64"))
mlon=np.asarray(data["mlon"],dtype="float64")
mlon=mlon.squeeze()
mlat=np.asarray(data["mlat"],dtype="float64")
mlat=mlat.squeeze()
SigmaP_ref=np.transpose(np.asarray(data["SIGP"],dtype="float64"))
SigmaH_ref=np.transpose(np.asarray(data["SIGH"],dtype="float64"))
mlonp=np.asarray(data["mlonp"],dtype="float64")
mlonp=mlonp.squeeze()
mlatp=np.asarray(data["mlatp"],dtype="float64")
mlatp=mlatp.squeeze()
int_ohmic_ref=np.transpose(np.asarray(data["int_ohmic"])) # this computed via integration of 3D dissipation
ohmic_ref=np.transpose(np.asarray(data["ohmici"]))


# Try to convert Spar to conductance, using steady-state integrated Poynting thm.
Eperp=E.squeeze()
magE2=np.sum(Eperp**2,axis=2)
SigmaP=-Spar/magE2


# Recompute Ohmic dissipation (field-integrated) as a test
[MLON,MLAT]=np.meshgrid(mlon,mlat)
SigmaP_refi=sciint.interpn((mlonp,mlatp),np.transpose(SigmaP_ref),(MLON,MLAT)) # needs to be permuted as lon,lat
dissipation=SigmaP_refi*magE2


# plot some comparisons
#plt.figure(dpi=300)
plt.subplots(1,2,dpi=100)

plt.subplot(1,2,1)
plt.pcolormesh(mlon,mlat,SigmaP)
plt.title("Estimated Pedersen")
plt.colorbar()

plt.subplot(1,2,2)
plt.pcolormesh(mlonp,mlatp,SigmaP_ref)
plt.title("Reference Pedersen")
plt.colorbar()
plt.show()


if flagSigP_debug:
    plotSigmaP_debug(mlon,mlat,mlonp,mlatp,Spar,Eperp,dissipation,int_ohmic_ref,
                     SigmaP_ref,SigmaP_refi,magE2)


