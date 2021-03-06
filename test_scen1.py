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
filename="/Users/zettergm/Pegasusr4i/Dropbox (Personal)/shared/shared_simulations/arcs/scen1.mat"
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


# Try to convert Spar to conductance, using steady-state integrated Poynting thm.
magE2=Ex**2+Ey**2
magE=np.sqrt(magE2)
SigmaP=-Spar/magE2


# compute E x bhat;  Take bhat to be in the minus z-direction (assumes northern hemis.)
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

# regularization of the problem ("regular" Tikhonov)
regparm=1e-14
#regparm=1e-9
regkern=scipy.sparse.eye(2*lx*ly,2*lx*ly)
bprime=A.transpose()@b
Aprime=(A.transpose()@A + regparm*regkern)
sigsreg=scipy.sparse.linalg.spsolve(Aprime,bprime,use_umfpack=True)
sigPreg=np.reshape(sigsreg[0:lx*ly],[lx,ly],order="F")
sigHreg=np.reshape(sigsreg[lx*ly:],[lx,ly],order="F")

# Tikhonov curvature regularization
regparm=1e-14
#regparm=1e-9
scale=np.ones((lx,ly))
[L2x,L2y]=laplacepieces2D(x,y,scale,scale)
regkern=scipy.sparse.block_diag((L2x+L2y,L2x+L2y),format="csr")
bprime=A.transpose()@b
Aprime=(A.transpose()@A + regparm*regkern)
sigsreg2=scipy.sparse.linalg.spsolve(Aprime,bprime,use_umfpack=True)
sigPreg2=np.reshape(sigsreg2[0:lx*ly],[lx,ly],order="F")
sigHreg2=np.reshape(sigsreg2[lx*ly:],[lx,ly],order="F")

# test various subcomponents of inverse problem
#  first a sanity check on the Poynting thm. (lower left block of full matrix)
ALL=LL;
bLL=svec;
xLL=scipy.sparse.linalg.spsolve(ALL,bLL)
sigPLL=np.reshape(xLL,[lx,ly])
sigPLL=sigPLL.transpose()

# try to use FD matrices to do a gradient to check that operators are being formed correctly
#thetap=np.pi/2-np.deg2rad(mlatp)
#meanthetap=np.average(thetap)
#phip=np.deg2rad(mlonp)
#meanphip=np.average(phip)
#southdistp=Re*(thetap-meanthetap)
#yp=np.flip(southdistp,axis=0)
#xp=Re*np.sin(meanthetap)*(phip-meanphip)
[xp,yp]=mag2xy(mlonp,mlatp)
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaP_ref.transpose())    # transpose to y,x
SigmaP_refi=(interpolant(x,y)).transpose()                              # transpose back to x,y
SigPvec=SigmaP_refi.flatten(order="F")
[Lx,Ly]=FDmat2D(x,y,np.ones(Ex.shape),np.ones(Ey.shape))
gradSigPxvec=Lx@SigPvec
gradSigPxmat=np.reshape(gradSigPxvec,[lx,ly],order="F")
gradSigPyvec=Ly@SigPvec
gradSigPymat=np.reshape(gradSigPyvec,[lx,ly],order="F")


# next try a system with no Hall current divergence (this is already nearly the case for our test example)
AUL=UL
IUL=scipy.sparse.eye(lx*ly,lx*ly)
regparm2=1e-16
AULprime=(AUL.transpose()@AUL+regparm2*IUL)
bUL=jvec
bULprime=AUL.transpose()@bUL
xUL=scipy.sparse.linalg.spsolve(AULprime,bULprime)
sigPUL=np.reshape(xUL,[lx,ly],order="F")


# just current continuity with Pedersen terms requires regularization what about adding in the Poynting thm. to the inversion???
AULLL=scipy.sparse.vstack([UL,LL])   # overdetermined system
bULLLprime=AULLL.transpose()@b
AULLLprime=(AULLL.transpose()@AULLL)    # don't regularize since overdetermined, simple Moore-Penrose approach
xULLL=scipy.sparse.linalg.spsolve(AULLLprime,bULLLprime)
sigPULLL=np.reshape(xULLL,[lx,ly],order="F")


# now try to recover the current density from matrix-computed conductivity gradients as a check
#  note that we neglect hall currents for now since they are small
jvectest=UL@SigPvec
jvectestmat=np.reshape(jvectest,[lx,ly],order="F")


# compute the projection of the Hall conductance gradient using matrix operators
interpolant=scipy.interpolate.interp2d(xp,yp,SigmaH_ref.transpose())
SigmaH_refi=(interpolant(x,y)).transpose()
SigHvec=SigmaH_refi.flatten(order="F")
gradSigHprojvec=(LxH+LyH)@SigHvec
gradSigHprojmat=np.reshape(gradSigHprojvec,[lx,ly],order="F")


# recover current density from operator with the Hall terms
SigHvec=SigmaH_refi.flatten(order="F")
jvectest2=UL@SigPvec+UR@SigHvec
jvectest2mat=np.reshape(jvectest2,[lx,ly],order="F")


# Alternatively we can algebraicaly compute the gradient of Hall conductance given
#  Pedersen conductance.  Then can execute a line integral to get the Hall term.
#  We do need to choose a location with very low Pedersen conductance for our reference
#  Hall conductance location.  The issue is that this only gives the the projection along
#  the ExB direction so this may not be a suitable option!!!
[gradSigPx,gradSigPy]=grad2D(SigmaP,x,y)
gradSigHproj=Jpar+gradSigPx*Ex+gradSigPy*Ey+SigmaP*divE     # Hall term from current continuity


# Hall term computed from finite differences.
[gradSigHx,gradSigHy]=grad2D(SigmaH_refi,x,y)
gradSigHprojFD=Erotx*gradSigHx+Eroty*gradSigHy


# check some of the calculations, gradients, divergences
if flagdebug:
    plt.subplots(2,3,dpi=100)

    plt.subplot(2,3,1)
    plt.pcolormesh(x,y,-(divE*SigmaP_refi).transpose())
    plt.colorbar()
    plt.title("$-\Sigma_P ( \\nabla \cdot \mathbf{E} )$")
    plt.clim(-1.5e-5,1.5e-5)
    
    plt.subplot(2,3,2)
    plt.pcolormesh(x,y,(-gradSigPx*Ex-gradSigPy*Ey).transpose())
    plt.colorbar()
    plt.title("$-\\nabla \Sigma_P \cdot \mathbf{E}$")
    plt.clim(-1.5e-5,1.5e-5)

    plt.subplot(2,3,3)
    plt.pcolormesh(x,y,(Erotx*gradSigHx+Eroty*gradSigHy).transpose())
    plt.colorbar()
    plt.title("$\\nabla \Sigma_H \cdot ( \mathbf{E} \\times \hat{b} )$")
    plt.clim(-1.5e-5,1.5e-5)
    
    plt.subplot(2,3,4)
    plt.pcolormesh(x,y,(Erotx*gradSigHx+Eroty*gradSigHy \
                   -gradSigPx*Ex-gradSigPy*Ey \
                   -divE*SigmaP_refi).transpose() )
    plt.colorbar()
    plt.title("Current density (all terms)")
    plt.clim(-1.5e-5,1.5e-5)    
    
    plt.subplot(2,3,5)
    plt.pcolormesh(x,y,Jpar.transpose())
    plt.colorbar()
    plt.title("Current density (model)")
    plt.clim(-1.5e-5,1.5e-5)    
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,3,dpi=100)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(mlon,mlat,SigmaP.transpose())
    plt.title("Estimated Pedersen")
    plt.colorbar()
    plt.clim(0,38)      
    
    plt.subplot(1,3,2)
    plt.pcolormesh(mlonp,mlatp,SigmaP_ref.transpose())
    plt.title("Reference Pedersen")
    plt.colorbar()
    plt.clim(0,38)    

    plt.subplot(1,3,3)
    plt.pcolormesh(mlonp,mlatp,SigmaH_ref.transpose())
    plt.title("Reference Hall")
    plt.colorbar()
    plt.clim(0,60)    
    plt.show(block=False)
    
if flagdebug:
    plt.subplots(1,3)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,gradSigPx.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Numerical $\\nabla \Sigma_P \cdot \mathbf{e}_x$")
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,gradSigPy.transpose())
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.title("Numerical $\\nabla \Sigma_P \cdot \mathbf{e}_y$")
   
    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,divE.transpose())
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.title("Numerical $\\nabla \cdot \mathbf{E}$")
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,2,dpi=100)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,gradSigPxmat.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Matrix $\\nabla \Sigma_P \cdot \mathbf{e}_x$")
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,gradSigPymat.transpose())
    plt.xlabel("x (km)")
    plt.colorbar()
    plt.title("Matrix $\\nabla \Sigma_P \cdot \mathbf{e}_y$")
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,3)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,jvectestmat.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("$J_\parallel$ (matrix sans Hall)")
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,Jpar.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("$J_\parallel$ from model")
    
    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,jvectest2mat.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("$J_\parallel$ (matrix with Hall)")    
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,3)
    
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,gradSigHproj.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Projection of ${\\nabla \Sigma_H}$ (CC)")
    plt.clim(-3e-6,3e-6)
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,gradSigHprojmat.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Projection of ${\\nabla \Sigma_H}$ (matrix)")
    plt.clim(-3e-6,3e-6)

    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,gradSigHprojFD.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.colorbar()
    plt.title("Projection of ${\\nabla \Sigma_H}$ (FD)")    
    plt.show(block=False)
    plt.clim(-3e-6,3e-6)
    
if flagdebug:
    plt.subplots(1,2)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,sigPreg.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Full Operator, norm regularized:  $\Sigma_P$")
    plt.colorbar()    
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,sigHreg.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Full Operator, norm egularized:  $\Sigma_H$")    
    plt.colorbar()
    plt.show(block=False)

if flagdebug:
    plt.subplots(1,2)

    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,sigPreg2.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Full Operator, curvature regularized:  $\Sigma_P$")
    plt.colorbar()
    plt.clim(0,38)    
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,sigHreg2.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Full Operator, curvature regularized $\Sigma_H$")    
    plt.colorbar()
    plt.clim(0,60)
    plt.show(block=False)
    
if flagdebug:
    plt.subplots(1,2)
    
    plt.subplot(1,2,1)
    plt.pcolormesh(x,y,sigPLL.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("$\Sigma_P$ via Poynting")
    plt.colorbar()    
    
    plt.subplot(1,2,2)
    plt.pcolormesh(x,y,sigPUL.transpose())
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("$\Sigma_P$ via current continuity")    
    plt.colorbar()    
    plt.show(block=False)

if flagdebug:
    plt.figure(dpi=100)
    plt.pcolormesh(x,y,sigPULLL.transpose())
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
#    plotSigmaP_debug(mlon,mlat,mlonp,mlatp,Spar,Eperp,dissipation,int_ohmic_ref, \
#                     SigmaP_ref,SigmaP_refi,magE2)


