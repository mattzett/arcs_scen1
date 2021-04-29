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
import scipy.interpolate, scipy.sparse, scipy.sparse.linalg
#from plot_fns import plotSigmaP_debug
from scen1_numerical import laplacepieces2D,mag2xy,linear_scen1,div2D,grad2D

# setup
plt.close("all")
flagSigP_debug=False
flagdebug=True

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

# interpolate reference data into observation grid
[xp,yp]=mag2xy(mlonp,mlatp)
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaP_ref.transpose())    # transpose to y,x
SigmaP_refi=(interpolant(x,y)).transpose()                              # transpose back to x,y
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaH_ref.transpose())
SigmaH_refi=(interpolant(x,y)).transpose()

# add noise to "measurements"
noisefrac=0
Jpar=Jpar+noisefrac*max(Jpar.flatten())*np.random.randn(lx,ly)
Spar=Spar+noisefrac*max(Spar.flatten())*np.random.randn(lx,ly)

# Convert Spar to conductance, using steady-state integrated Poynting thm.
magE2=Ex**2+Ey**2
magE=np.sqrt(magE2)
SigmaP=-Spar/magE2
SigmaPvec=SigmaP.flatten(order="F")

# compute E x bhat; take bhat to be in the minus z-direction (assumes northern hemis.)
Erotx=-Ey
Eroty=Ex

# flatten data vectors
jvec=Jpar.flatten(order="F")
svec=Spar.flatten(order="F")

# Now try to estimate the Hall conductance using current continuity.
[A,b,UL,UR,LL,LR,LxH,LyH,divE]=linear_scen1(x,y,Ex,Ey,Erotx,Eroty,Jpar,Spar)

# Compute some derivatives that we need
divE=div2D(Ex,Ey,x,y)
[gradSigPx,gradSigPy]=grad2D(SigmaP,x,y)
LSigP=LxH@SigmaPvec + LyH@SigmaPvec
LSigP=LSigP/SigmaPvec
I=scipy.sparse.eye(lx*ly,lx*ly)
linterm=I.tocsr()     # because we need to do elementwise modifications
for k in range(0,lx*ly):
    linterm[k,k]=LSigP[k]

# Formulate the Hall estimation problem
b=jvec/SigmaPvec+divE.flatten(order="F")+ \
    gradSigPx.flatten(order="F")/SigmaPvec*Ex.flatten(order="F") + \
    gradSigPy.flatten(order="F")/SigmaPvec*Ey.flatten(order="F")    
A=UR+linterm

# Regularize
regparm1=1e-9
#regparm2=5e-15
regparm2=1e-13
#refrat=1
refrat=1.5
scale=np.ones((lx,ly))
[L2x,L2y]=laplacepieces2D(x,y,scale,scale)
regkern1=L2x+L2y                             # curvature
regkern2=scipy.sparse.eye(lx*ly,lx*ly)     # distance from Pedersen
Aprime=( A.transpose()@A + regparm1*regkern1.transpose()@regkern1 + regparm2*regkern2.transpose()@regkern2)
bprime=A.transpose()@b + regparm2*( regkern2.transpose()@regkern2 )@np.ones((lx*ly))*refrat

# solve
sigHrat=scipy.sparse.linalg.spsolve(Aprime,bprime,use_umfpack=True)
sigHrat=np.reshape(sigHrat,[lx,ly],order="F")
SigmaH=SigmaP*sigHrat

# plots
plt.subplots(2,3)
plt.subplot(2,3,1)
plt.pcolormesh(x,y,SigmaP_refi.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("$\Sigma_P$ ground truth")    
plt.colorbar()
plt.clim(0,60)

plt.subplot(2,3,2)
plt.pcolormesh(x,y,SigmaH_refi.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("$\Sigma_H$ ground truth")    
plt.colorbar()
plt.clim(0,60)

plt.subplot(2,3,3)
plt.pcolormesh(x,y,SigmaH_refi.transpose()/SigmaP_refi.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("$\Sigma_H / \Sigma_P$ ground truth")    
plt.colorbar()
#plt.clim(0,2)

plt.subplot(2,3,4)
plt.pcolormesh(x,y,SigmaP.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("estimated $\Sigma_P$")
plt.colorbar()
plt.clim(0,60)

plt.subplot(2,3,5)
plt.pcolormesh(x,y,SigmaH.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("estimated $\Sigma_H$")
plt.colorbar()
plt.clim(0,60)

plt.subplot(2,3,6)
plt.pcolormesh(x,y,sigHrat.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("estimated $\Sigma_H / \Sigma_P$")    
plt.colorbar()
#plt.clim(0,2)

plt.show(block=False)


