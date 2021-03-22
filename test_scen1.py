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
from plot_fns import plotSigmaP_debug
from scen1_numerical import div2D,FDmat2D

# setup
plt.close("all")
flagSigP_debug=False
flagdebug=True
Re=6370e3


# Load synthetic data maps and organize data, permute/transpose arrays as lat,lon for plotting
#  squeeze 1D arrays for plotting as well
filename="/Users/zettergm/Dropbox (Personal)/shared/shared_simulations/arcs/scen1.mat"
data=spio.loadmat(filename)
E=np.asarray(data["E"],dtype="float64")
Ex=np.squeeze(E[0,:,:,0]); Ey=np.squeeze(E[0,:,:,0]);
Jpar=np.asarray(data["Jpar"],dtype="float64")
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


# plot some comparisons to verify correctness
if flagdebug:
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


# map magnetic coordinates to Cartesian to facilitate differencing and "fitting"
theta=np.pi/2-np.deg2rad(mlat)
meantheta=np.average(theta)
phi=np.deg2rad(mlon)
meanphi=np.average(phi)
southdist=Re*(theta-meantheta)
y=np.flip(southdist,axis=0)
x=Re*np.sin(meantheta)*(phi-meanphi)
lx=x.size; ly=y.size;


# compute unit vectors for E,Exb basis, if needed?  Assume b is in the minus z-direction
Erotx=-Ey
Eroty=Ex


# Now try to estimate the Hall conductance using current continuity...  We could
#  formulate this as an estimation problem which the two conductances were estimated
#  subject to the approximate constraints dictated by the conservation laws.  
#  1) try finite difference decomposition (non-parametric)
#  2) basis expansion version if conditioning is poor
# Use python modules/functions for FDEs
[LxEx,LyEy]=FDmat2D(x,y,Ex,Ey)
I=scipy.sparse.eye(lx*ly,lx*ly)
IdivE=I.tocsr()     # because we need to do elementwise mods later...
divE=div2D(Ex,Ey,x,y)
for ix in range(0,lx):
    for iy in range(0,ly):
        k=iy*lx+ix
        IdivE[k,k]=divE[ix,iy]
magE=np.sqrt(magE2)
UL=IdivE + LxEx + LyEy

LL=I
for ix in range(0,lx):
    for iy in range(0,ly):
        k=iy*lx+ix
        IdivE[k,k]=magE2[ix,iy]

[LxH,LyH]=FDmat2D(x,y,Erotx,Eroty)
UR=LxH+LyH
LR=I*0     # my lazy way of generate a null matrix of the correct size
Uhstack=scipy.sparse.hstack([UL,UR])
Lhstack=scipy.sparse.hstack([LL,LR])
A=scipy.sparse.vstack([Lhstack,Uhstack])
b=np.concatenate((Jpar.flatten(order="F"),Spar.flatten(order="F")),axis=0)

sigs=scipy.sparse.linalg.spsolve(A.tocsr(),b)    # what backend is this using? can we force umfpack?
sigPnoreg=np.reshape(sigs[0:lx*ly],[lx,ly])
sigHnoreg=np.reshape(sigs[lx*ly:],[lx,ly])


# regularization of the problem (Tikhonov)
regparm=0.05
bprime=A.transpose()@b
Aprime=(A.transpose()@A + regparm*scipy.sparse.eye(2*lx*ly,2*lx*ly))
xreg=scipy.sparse.linalg.spsolve(Aprime,bprime)
sigPreg=np.reshape(xreg[0:lx*ly],[lx,ly])
sigHreg=np.reshape(xreg[lx*ly:],[lx,ly])


# Alternatively we can algebraicaly compute the gradient of Hall conductance given
#  Pedersen conductance.  Then can execute a line integral to get the Hall term.
#  We do need to choose a location with very low Pedersen conductance for our reference
#  Hall conductance location.  The issue is that this only gives the the projection along
#  the ExB direction so this may not be a suitable option!!!
[gradSigPx,gradSigPy]=np.gradient(SigmaP,x,y)
divE=div2D(Eperp[:,:,0],Eperp[:,:,1],x,y)
gradSigHproj=Jpar-SigmaP*divE+gradSigPx*Eperp[:,:,0]+gradSigPy*Eperp[:,:,1]



# check some of the calculations, gradients, divergences
if flagdebug:
    plt.subplots(1,3)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,gradSigPx)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,gradSigPy)
    plt.xlabel("x (km)")
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,divE)
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.show()
    
    
if flagdebug:
    plt.figure()
    plt.pcolormesh(x,y,gradSigHproj)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    
    
if flagdebug:
    plt.figure()
    


# do some extra debug plots?
if flagSigP_debug:
    # Recompute Ohmic dissipation (field-integrated) as a test
    [MLON,MLAT]=np.meshgrid(mlon,mlat)
    SigmaP_refi=scipy.interpolate.interpn((mlonp,mlatp),np.transpose(SigmaP_ref),(MLON,MLAT)) # needs to be permuted as lon,lat
    dissipation=SigmaP_refi*magE2
    plotSigmaP_debug(mlon,mlat,mlonp,mlatp,Spar,Eperp,dissipation,int_ohmic_ref, \
                     SigmaP_ref,SigmaP_refi,magE2)


