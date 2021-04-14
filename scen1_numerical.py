#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:37:54 2021

Perform some numerical operations needed for scenario 1 estimation

@author: zettergm
"""


#############################################################################
#### imports
import numpy as np
import scipy.sparse

#############################################################################
#### global vars
Re=6370e3

#############################################################################
#### procedures

# convert mlon,mlat->xy
def mag2xy(mlon,mlat):
    theta=np.pi/2-np.deg2rad(mlat)
    meantheta=np.average(theta)
    phi=np.deg2rad(mlon)
    meanphi=np.average(phi)
    southdist=Re*(theta-meantheta)
    y=np.flip(southdist,axis=0)
    x=Re*np.sin(meantheta)*(phi-meanphi)
    return [x,y]


# form a linear inverse problem
def linear_scen1(x,y,Ex,Ey,Erotx,Eroty,Jpar,Spar):
    lx=x.size; ly=y.size;
    magE2=Ex**2+Ey**2
    magE=np.sqrt(magE2)

    ## The pieces of the full operator being used
    # compute Ex d/dx and Ey d/dy as a matrix operator
    [LxEx,LyEy]=FDmat2D(x,y,Ex,Ey)
    
    # compute div(E) multiply as a matrix operator (diagonal)
    I=scipy.sparse.eye(lx*ly,lx*ly)
    IdivE=I.tocsr()     # because we need to do elementwise modifications
    divE=div2D(Ex,Ey,x,y)
    for ix in range(0,lx):
        for iy in range(0,ly):
            k=iy*lx+ix
            IdivE[k,k]=divE[ix,iy]
    
    # upper left block of system (operators for Pedersen conductance
    UL=IdivE + LxEx + LyEy
    UL=-UL    # sign convention, z is up
    
    # lower left block of system (multiplier of Pedersen)
    LL=I.tocsr()
    for ix in range(0,lx):
        for iy in range(0,ly):
            k=iy*lx+ix
            LL[k,k]=-magE2[ix,iy]
    
    # upper right block
    [LxH,LyH]=FDmat2D(x,y,Erotx,Eroty)
    UR=-LxH-LyH
    UR=-UR   # sign convenction, z is up
    
    # lower right block (nothing since Hall terms doesn't appear in Poynting Thm.)
    LR=I*0     # my lazy way of generate a null, sparse matrix of the correct size
    
    ## form the full operator
    scaleJ=1; scaleS=1;
    Uhstack=scipy.sparse.hstack([UL,UR])
    Lhstack=scipy.sparse.hstack([LL,LR])
    A=scipy.sparse.vstack([Uhstack/scaleJ,Lhstack/scaleS])
    jvec=Jpar.flatten(order="F")
    svec=Spar.flatten(order="F")
    b=np.concatenate((jvec/scaleJ,svec/scaleS),axis=0)    # make sure to use column-major ordering

    return[A,b,UL,UR,LL,LR,LxH,LyH,divE]


# compute second derivatives in matrix form for a laplacian operator
def laplacepieces2D(x,y,scalex,scaley):
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    lx=x.size; ly=y.size

    lent=3*(lx-2)*(ly)+2*ly     # three entries for each x interior point; 2*ly edges in x with two points each    
    ir=np.zeros(lent)
    ic=np.zeros(lent)
    L=np.zeros(lent)
    ient=0    
    for iy in range(0,ly):        
        for ix in range(0,lx):
            k=iy*lx+ix      # this represents the equation for the kth unknown, column major

            if ix==0:
                ir[ient]=k
                ic[ient]=k
                L[ient]=1
                ient=ient+1
            elif ix==lx-1:
                ir[ient]=k
                ic[ient]=k
                L[ient]=1
                ient=ient+1                
            else:    
                ir[ient]=k
                ic[ient]=k-1
                L[ient]=1/dx**2*scalex[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k
                L[ient]=-2/dx**2*scalex[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k+1
                L[ient]=1/dx**2*scalex[ix,iy]
                ient=ient+1                
    L2x=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )
    L2x=L2x.tocsr()
    #print("for x derivative filled:  ",ient," entries")
    
    lent=3*(lx)*(ly-2)+2*lx     # three entries for each y interior point; 2*lx edges in x with two points each
    ir=np.zeros(lent)
    ic=np.zeros(lent)
    L=np.zeros(lent)
    ient=0
    for iy in range(0,ly):
        for ix in range(0,lx):
            k=iy*lx+ix      # this represents the equation for the kth unknown

            if iy==0:
                ir[ient]=k
                ic[ient]=k
                L[ient]=1
                ient=ient+1
            elif iy==ly-1:
                ir[ient]=k
                ic[ient]=k
                L[ient]=1
                ient=ient+1                
            else: 
                ir[ient]=k
                ic[ient]=k-lx
                L[ient]=1/dy**2*scaley[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k
                L[ient]=-2/dy**2*scaley[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k+lx
                L[ient]=1/dy**2*scaley[ix,iy]
                ient=ient+1
    L2y=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )  
    L2y=L2y.tocsr()
    #print("for y derivative filled:  ",ient," entries")
    return [L2x,L2y]


# two dimensional divergence assuming i,j -> x,y. 
def div2D(Ux,Uy,x,y):
    [dUxdx,_]=np.gradient(Ux,x,y)
    [_,dUydy]=np.gradient(Uy,x,y)
    return dUxdx+dUydy


# two dimensional gradient assuming i,j -> x,y
def grad2D(U,x,y):
    [dUdx,dUdy]=np.gradient(U,x,y)
    return [dUdx,dUdy]


# construct a finite difference matrix for a single coordinate derivative over a 2D grid
#  the matrix is returned in csr format so it can be indexed or used for sparse solutions.
#  The returned matrix assumes column-major ordering.
def FDmat2D(x,y,scalex,scaley):
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    lx=x.size; ly=y.size
    lent=2*lx*ly     # two finite difference entries for each grid point
    
    ir=np.zeros(lent)
    ic=np.zeros(lent)
    L=np.zeros(lent)
    ient=0    
    for iy in range(0,ly):        
        for ix in range(0,lx):
            k=iy*lx+ix      # this represents the equation for the kth unknown, column major

            if ix==0:
                ir[ient]=k
                ic[ient]=k
                L[ient]=-1/dx*scalex[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k+1
                L[ient]=1/dx*scalex[ix,iy]
                ient=ient+1
            elif ix==lx-1:
                ir[ient]=k
                ic[ient]=k-1
                L[ient]=-1/dx*scalex[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k
                L[ient]=1/dx*scalex[ix,iy]
                ient=ient+1                
            else:    
                ir[ient]=k
                ic[ient]=k-1
                L[ient]=-1/2/dx*scalex[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k+1
                L[ient]=1/2/dx*scalex[ix,iy]
                ient=ient+1                
    Lx=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )
    Lx=Lx.tocsr()
    # print("for x derivative filled:  ",ient," entries")
    
    ir=np.zeros(lent)
    ic=np.zeros(lent)
    L=np.zeros(lent)
    ient=0
    for iy in range(0,ly):
        for ix in range(0,lx):
            k=iy*lx+ix      # this represents the equation for the kth unknown

            if iy==0:
                ir[ient]=k
                ic[ient]=k
                L[ient]=-1/dy*scaley[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k+lx
                L[ient]=1/dy*scaley[ix,iy]
                ient=ient+1                
            elif iy==ly-1:
                ir[ient]=k
                ic[ient]=k-lx
                L[ient]=-1/dy*scaley[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k
                L[ient]=1/dy*scaley[ix,iy]
                ient=ient+1                
            else: 
                ir[ient]=k
                ic[ient]=k-lx
                L[ient]=-1/2/dy*scaley[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k+lx
                L[ient]=1/2/dy*scaley[ix,iy]
                ient=ient+1
    Ly=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )  
    Ly=Ly.tocsr()
    # print("for y derivative filled:  ",ient," entries")
    
    return [Lx,Ly]


# This implements second derivatives over a 2D grid as a matrix operation
#def laplacepieces2D(x,y,scalex,scaley):
#    dx=x[1]-x[0]
#    dy=y[1]-y[0]
#    lx=x.size; ly=y.size
#
#    lent=3*(lx-2)*(ly)+2*2*ly     # three entries for each x interior point; 2*ly edges in x with two points each    
#    ir=np.zeros(lent)
#    ic=np.zeros(lent)
#    L=np.zeros(lent)
#    ient=0    
#    for iy in range(0,ly):        
#        for ix in range(0,lx):
#            k=iy*lx+ix      # this represents the equation for the kth unknown, column major
#
#            if ix==0:
#                ir[ient]=k
#                ic[ient]=k
#                L[ient]=-1/dx*scalex[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k+1
#                L[ient]=1/dx*scalex[ix,iy]
#                ient=ient+1
#            elif ix==lx-1:
#                ir[ient]=k
#                ic[ient]=k-1
#                L[ient]=-1/dx*scalex[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k
#                L[ient]=1/dx*scalex[ix,iy]
#                ient=ient+1                
#            else:    
#                ir[ient]=k
#                ic[ient]=k-1
#                L[ient]=1/dx**2*scalex[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k
#                L[ient]=-2/dx**2*scalex[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k+1
#                L[ient]=1/dx**2*scalex[ix,iy]
#                ient=ient+1                
#    L2x=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )
#    L2x=L2x.tocsr()
#    # print("for x derivative filled:  ",ient," entries")
#    
#    lent=3*(lx)*(ly-2)+2*2*lx     # three entries for each y interior point; 2*lx edges in x with two points each
#    ir=np.zeros(lent)
#    ic=np.zeros(lent)
#    L=np.zeros(lent)
#    ient=0
#    for iy in range(0,ly):
#        for ix in range(0,lx):
#            k=iy*lx+ix      # this represents the equation for the kth unknown
#
#            if iy==0:
#                ir[ient]=k
#                ic[ient]=k
#                L[ient]=-1/dy*scaley[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k+lx
#                L[ient]=1/dy*scaley[ix,iy]
#                ient=ient+1                
#            elif iy==ly-1:
#                ir[ient]=k
#                ic[ient]=k-lx
#                L[ient]=-1/dy*scaley[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k
#                L[ient]=1/dy*scaley[ix,iy]
#                ient=ient+1                
#            else: 
#                ir[ient]=k
#                ic[ient]=k-lx
#                L[ient]=1/dy**2*scaley[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k
#                L[ient]=-2/dy**2*scaley[ix,iy]
#                ient=ient+1
#                ir[ient]=k
#                ic[ient]=k+lx
#                L[ient]=1/dy**2*scaley[ix,iy]
#                ient=ient+1
#    L2y=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )  
#    L2y=L2y.tocsr()
#    # print("for y derivative filled:  ",ient," entries")
#    return [L2x,L2y]
