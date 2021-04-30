#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 07:59:49 2021

@author: zettergm
"""

# imports
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.sparse, scipy.sparse.linalg
from scen1_numerical import laplacepieces2D,mag2xy,linear_scen1,div2D,grad2D

# Load synthetic data maps and organize data, permute/transpose arrays as lat,lon for plotting
#  squeeze 1D arrays for plotting as well
#  We presume all of the data are organized as (z),x,y upon input
filename="/Users/zettergm/Dropbox (Personal)/shared/shared_simulations/arcs/scen1.mat"
data=spio.loadmat(filename)
E=np.asarray(data["E"],dtype="float64")      # do not use directly in calculations due to r,theta,phi basis.
Ex=np.squeeze(E[0,:,:,1]); Ey=np.squeeze(E[0,:,:,2]);
Jpar=np.asarray(data["Jpar"],dtype="float64")                  # already indexed x,y
Spar=np.asarray(data["Spar"],dtype="float64")                  # indexed x,y
mlon=np.asarray(data["mlon"],dtype="float64")
mlon=mlon.squeeze()
mlat=np.asarray(data["mlat"],dtype="float64")
mlat=mlat.squeeze()
SigmaP_ref=np.asarray(data["SIGP"],dtype="float64")            # indexed as x,y already
SigmaH_ref=np.abs(np.asarray(data["SIGH"],dtype="float64"))    # indexed as x,y already; convert to positive Hall conductance
mlonp=np.asarray(data["mlonp"],dtype="float64")
mlonp=mlonp.squeeze()
mlatp=np.asarray(data["mlatp"],dtype="float64")
mlatp=mlatp.squeeze()
int_ohmic_ref=np.asarray(data["int_ohmic"])       # this computed via integration of 3D dissipation; indexed x,y
ohmic_ref=np.asarray(data["ohmici"])

# map magnetic coordinates to local Cartesian to facilitate differencing and "fitting"
[x,y]=mag2xy(mlon,mlat)
lx=x.size; ly=y.size;

# compute E x bhat; take bhat to be in the minus z-direction (assumes northern hemis.)
Erotx=-Ey
Eroty=Ex

# various terms in continuity equation
[A,b,UL,UR,LL,LR,LxH,LyH,divE]=linear_scen1(x,y,Ex,Ey,Erotx,Eroty,Jpar,Spar)

# interpolations
[xp,yp]=mag2xy(mlonp,mlatp)
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaP_ref.transpose())    # transpose to y,x
SigmaP_refi=(interpolant(x,y)).transpose()                              # transpose back to x,y
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaH_ref.transpose())
SigmaH_refi=(interpolant(x,y)).transpose()
[gradSigPx,gradSigPy]=grad2D(SigmaP_refi,x,y)
gradSigHproj=Jpar+gradSigPx*Ex+gradSigPy*Ey+SigmaP_refi*divE     # Hall term from current continuity
