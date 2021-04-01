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
Ex=np.squeeze(E[0,:,:,0]); Ey=np.squeeze(E[0,:,:,1]);
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

LL=I.tocsr()
for ix in range(0,lx):
    for iy in range(0,ly):
        k=iy*lx+ix
        LL[k,k]=-magE2[ix,iy]

[LxH,LyH]=FDmat2D(x,y,Erotx,Eroty)
UR=-LxH-LyH
LR=I*0     # my lazy way of generate a null matrix of the correct size


# determine a scaling for current density and Poynting flux problems (separately) 
#  and apply to matrix elms.  The scaling does seem to be necessary to achieve a 
#  decent inversion since the parameters/equations are heterogeneous (different units)
#scaleS=np.max(abs(Spar))
#scaleJ=np.max(abs(Jpar))
scaleJ=1; scaleS=1;
Uhstack=scipy.sparse.hstack([UL,UR])
Lhstack=scipy.sparse.hstack([LL,LR])
A=scipy.sparse.vstack([Uhstack/scaleJ,Lhstack/scaleS])
jvec=Jpar.flatten(order="F")
svec=Spar.flatten(order="F")
b=np.concatenate((jvec/scaleJ,svec/scaleS),axis=0)    # make sure to use column-major ordering
sigs=scipy.sparse.linalg.spsolve(A.tocsr(),b,use_umfpack=True)    # what backend is this using? can we force umfpack?
sigPnoreg=np.reshape(sigs[0:lx*ly],[lx,ly])
sigHnoreg=np.reshape(sigs[lx*ly:],[lx,ly])


# regularization of the problem (Tikhonov)
regparm=1e-28
regkern=scipy.sparse.eye(2*lx*ly,2*lx*ly)
# regkern=regkern.tocsr(copy=True)
# for k in range(0,2*lx*ly):
#     if (k>=lx*ly):
#         regkern[k,k]=1/10
#     else:
#         regkern[k,k]=1
bprime=A.transpose()@b
Aprime=(A.transpose()@A + regparm*regkern)
sigsreg=scipy.sparse.linalg.spsolve(Aprime,bprime,use_umfpack=True)
sigPreg=np.reshape(sigsreg[0:lx*ly],[lx,ly])
sigHreg=np.reshape(sigsreg[lx*ly:],[lx,ly])


# test various subcomponents of inverse problem
#  first a sanity check on the Poynting thm. (lower left block of full matrix)
ALL=LL;
bLL=svec;
xLL=scipy.sparse.linalg.spsolve(ALL,bLL)
sigPLL=np.reshape(xLL,[lx,ly])
sigPLL=sigPLL.transpose()


# try to use FD matrices to do a gradient to check that operators are being formed correctly
thetap=np.pi/2-np.deg2rad(mlatp)
meanthetap=np.average(thetap)
phip=np.deg2rad(mlonp)
meanphip=np.average(phip)
southdistp=Re*(thetap-meanthetap)
yp=np.flip(southdistp,axis=0)
xp=Re*np.sin(meanthetap)*(phip-meanphip)
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaP_ref)
SigmaP_refi=interpolant(x,y)
SigPvec=SigmaP_refi.flatten(order="F")
[Lx,Ly]=FDmat2D(x,y,np.ones(Ex.shape),np.ones(Ey.shape))
gradSigPxvec=Lx@SigPvec
gradSigPxmat=np.reshape(gradSigPxvec,[lx,ly],order="F")
gradSigPyvec=Ly@SigPvec
gradSigPymat=np.reshape(gradSigPyvec,[lx,ly],order="F")


# next try a system with no Hall current divergence (this is already nearly the case for our test example)
AUL=UL
IUL=scipy.sparse.eye(lx*ly,lx*ly)
AULprime=(AUL.transpose()@AUL+1e-21*IUL)
bUL=-1*jvec     # seems to be a sign convention issue here???
bULprime=AUL.transpose()@bUL
#xUL=scipy.sparse.linalg.spsolve(AUL,bUL)
xUL=scipy.sparse.linalg.spsolve(AULprime,bULprime)
sigPUL=np.reshape(xUL,[lx,ly],order="F")


# just current continuity with Pedersen terms requires regularization what about adding in the Poynting thm. to the inversion???
AULLL=scipy.sparse.vstack([UL,LL])   # overdetermined system
bULLLprime=AULLL.transpose()@b
AULLLprime=(AULLL.transpose()@AULLL)    # don't regularize since overdetermined, simple Moore-Penrose approach
xULLL=scipy.sparse.linalg.spsolve(AULLLprime,bULLLprime)
sigPULLL=np.reshape(xULLL,[lx,ly],order="F")


# now try to recover the current density from matrix-computed conductivity gradients as a check
jvectest=-1*UL@SigPvec     #possibly a sign convention issue here???
jvectestmat=np.reshape(jvectest,[lx,ly],order="F")


# compute the project of the Hall conductance gradient using matrix operators
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaH_ref)
SigmaH_refi=interpolant(x,y)
SigHvec=SigmaH_refi.flatten(order="F")
gradSigHprojvec=(LxH+LyH)@SigHvec
gradSigHprojmat=np.reshape(gradSigHprojvec,[lx,ly],order="F")


# Alternatively we can algebraicaly compute the gradient of Hall conductance given
#  Pedersen conductance.  Then can execute a line integral to get the Hall term.
#  We do need to choose a location with very low Pedersen conductance for our reference
#  Hall conductance location.  The issue is that this only gives the the projection along
#  the ExB direction so this may not be a suitable option!!!
[gradSigPx,gradSigPy]=np.gradient(SigmaP,x,y)
divE=div2D(Eperp[:,:,0],Eperp[:,:,1],x,y)
#gradSigHproj=Jpar-SigmaP*divE+gradSigPx*Eperp[:,:,0]+gradSigPy*Eperp[:,:,1]
gradSigHproj=SigmaP*divE+gradSigPx*Eperp[:,:,0]+gradSigPy*Eperp[:,:,1]-Jpar    # Hall term from current continuity
[gradSigHx,gradSigHy]=np.gradient(SigmaH_refi,x,y)
gradSigHprojFD=Erotx*gradSigHx+Eroty*gradSigHy


# check some of the calculations, gradients, divergences
if flagdebug:
    plt.subplots(1,3,dpi=100)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(mlon,mlat,SigmaP)
    plt.title("Estimated Pedersen")
    plt.colorbar()
    
    plt.subplot(1,3,2)
    plt.pcolormesh(mlonp,mlatp,SigmaP_ref)
    plt.title("Reference Pedersen")
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.pcolormesh(mlonp,mlatp,SigmaH_ref)
    plt.title("Reference Hall")
    plt.colorbar()
    plt.show(block=False)
    
if flagdebug:
    plt.subplots(1,3)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,gradSigPx)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Numerical $\\nabla \Sigma_P \cdot \mathbf{e}_x$")
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,gradSigPy)
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.title("Numerical $\\nabla \Sigma_P \cdot \mathbf{e}_y$")
   
    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,divE)
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.title("Numerical $\\nabla \cdot \mathbf{E}$")
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,2,dpi=100)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,gradSigPxmat)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Matrix $\\nabla \Sigma_P \cdot \mathbf{e}_x$")
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,gradSigPymat)
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.title("Matrix $\\nabla \Sigma_P \cdot \mathbf{e}_y$")
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,2)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,jvectestmat)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("$J_\parallel$ computed from matrix operator")
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,Jpar)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("$J_\parallel$ from model")    
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,3)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,gradSigHproj)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Projection of ${\\nabla \Sigma_H}$ into ExB direction (CC)")
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,gradSigHprojmat)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Projection of ${\\nabla \Sigma_H}$ into ExB direction (matrix)")

    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,gradSigHprojFD)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Projection of ${\\nabla \Sigma_H}$ into ExB direction (FD)")    
    plt.show(block=False)
    
if flagdebug:
    plt.subplots(1,2)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,sigPreg.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Full Operator Regularized $\Sigma_P$")
    plt.colorbar()    
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,sigHreg.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Full Operator Regularized $\Sigma_H$")    
    plt.colorbar()
    plt.show(block=False)
    
if flagdebug:
    plt.subplots(1,2)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,sigPLL)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("$\Sigma_P$ via Poynting")
    plt.colorbar()    
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,sigPUL)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("$\Sigma_P$ via current continuity")    
    plt.colorbar()    
    plt.show(block=False)

if flagdebug:
    plt.figure(dpi=100)
    plt.pcolormesh(x,y,sigPULLL)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("$\Sigma_P$ via current continuity and Poynting combined")    
    plt.colorbar()    
    plt.show(block=False)    


# do some extra debug plots?
if flagSigP_debug:
    # Recompute Ohmic dissipation (field-integrated) as a test
    [MLON,MLAT]=np.meshgrid(mlon,mlat)
    SigmaP_refi=scipy.interpolate.interpn((mlonp,mlatp),np.transpose(SigmaP_ref),(MLON,MLAT)) # needs to be permuted as lon,lat
    dissipation=SigmaP_refi*magE2
    plotSigmaP_debug(mlon,mlat,mlonp,mlatp,Spar,Eperp,dissipation,int_ohmic_ref, \
                     SigmaP_ref,SigmaP_refi,magE2)


