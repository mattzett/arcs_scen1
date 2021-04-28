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
from scen1_numerical import div2D,grad2D,FDmat2D,laplacepieces2D,mag2xy,linear_scen1

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

# Now try to estimate the Hall conductance using current continuity...  We could
#  formulate this as an estimation problem which the two conductances were estimated
#  subject to the approximate constraints dictated by the conservation laws.  
#  1) try finite difference decomposition (non-parametric)
#  2) basis expansion version if conditioning is poor (it seems to be)
[A,b,UL,UR,LL,LR,LxH,LyH,divE]=linear_scen1(x,y,Ex,Ey,Erotx,Eroty,Jpar,Spar)

# Tikhonov curvature regularization; this is our reference best case from the initial study
regparm=1e-6
scale=np.ones((lx,ly))
[L2x,L2y]=laplacepieces2D(x,y,scale,scale)
regkern=scipy.sparse.block_diag((L2x+L2y,L2x+L2y),format="csr")
bprime=A.transpose()@b
Aprime=(A.transpose()@A + regparm*regkern.transpose()@regkern)     # note there was an error here in the original code
sigsreg2=scipy.sparse.linalg.spsolve(Aprime,bprime,use_umfpack=True)
sigPreg2=np.reshape(sigsreg2[0:lx*ly],[lx,ly],order="F")
sigHreg2=np.reshape(sigsreg2[lx*ly:],[lx,ly],order="F")

# take the Pedersen conductivity as given and solve the Hall problem (with curv. regularization)
regparm=1e-9
scale=np.ones((lx,ly))
[L2x,L2y]=laplacepieces2D(x,y,scale,scale)
regkern=L2x+L2y
A2=UR
b2=jvec-UL@SigmaPvec
b2prime=A2.transpose()@b2
A2prime=(A2.transpose()@A2 + regparm*regkern.transpose()@regkern)
sigHregsep=scipy.sparse.linalg.spsolve(A2prime,b2prime,use_umfpack=True)
sigHregsep=np.reshape(sigHregsep,[lx,ly],order="F")

# # take the Pedersen conductivity as given and solve the Hall problem (with curv. 
# #  regularization and favor solutions similar to Pedersen conductance)
# regparm=1e-14
# scale=np.ones((lx,ly))
# [L2x,L2y]=laplacepieces2D(x,y,scale,scale)
# regkern1=L2x+L2y                             # curvature
# regkern2=I=scipy.sparse.eye(lx*ly,lx*ly)     # distance from Pedersen
# A3=UR
# A3prime=(A3.transpose()@A3 + regparm*regkern1 + regparm*regkern2)
# b3=jvec-UL@SigmaPvec+regparm*regkern2*SigmaPvec
# b3prime=A2.transpose()@b3

# make plots
plt.subplots(1,2,dpi=100)
plt.subplot(1,2,1)
plt.pcolormesh(x,y,sigHregsep.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Sequential estimation, curvature regularized $\Sigma_H$")    
plt.colorbar()
plt.clim(0,60)

plt.subplot(1,2,2)
plt.pcolormesh(x,y,sigHreg2.transpose())
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Full Operator, curvature regularized $\Sigma_H$")    
plt.colorbar()
plt.clim(0,60)
plt.show(block=False)