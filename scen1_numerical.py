#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:37:54 2021

Perform some numerical operations needed for scenario 1 estimation

@author: zettergm
"""


# imports
import numpy as np
import scipy.sparse


# This implements second derivatives over a 2D grid as a matrix operation
def laplacepieces2D(x,y,scalex,scaley):
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    lx=x.size; ly=y.size

    lent=3*(lx-2)*(ly)+2*2*ly     # three entries for each x interior point; 2*ly edges in x with two points each    
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
    # print("for x derivative filled:  ",ient," entries")
    
    lent=3*(lx)*(ly-2)+2*2*lx     # three entries for each y interior point; 2*lx edges in x with two points each
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
                L[ient]=1/dy**2*scaley[ix,iy]
                ient=ient+1
                ir[ient]=k
                ic[ient]=k
                L[ient]=-2/dy**2*scaley[ix,iy]
                ir[ient]=k
                ic[ient]=k+lx
                L[ient]=1/dy**2*scaley[ix,iy]
                ient=ient+1
    L2y=scipy.sparse.coo_matrix( (L,(ir,ic)), shape=(lx*ly,lx*ly) )  
    L2y=L2y.tocsr()
    # print("for y derivative filled:  ",ient," entries")
    return [L2x,L2y]

# two dimensional divergence assuming i,j -> x,y.  This mean that the input must be
#  transposes prior to being used in numpy.gradient - this is not at all clear from the
#  documentation...
#def div2D(Ux,Uy,x,y):
#    [dUxdx,_]=np.gradient(Ux.transpose(),x,y)
#    [_,dUydy]=np.gradient(Uy.transpose(),x,y)
#    return dUxdx.transpose()+dUydy.transpose()

def div2D(Ux,Uy,x,y):
    [dUxdx,_]=np.gradient(Ux,x,y)
    [_,dUydy]=np.gradient(Uy,x,y)
    return dUxdx+dUydy

# two dimensional gradient assuming i,j -> x,y
#def grad2D(U,x,y):
#    [dUdx,dUdy]=np.gradient(U.transpose(),x,y)
#    return [dUdx.transpose(),dUdy.transpose()]

def grad2D(U,x,y):
    [dUdx,dUdy]=np.gradient(U,x,y)
    return [dUdx,dUdy]

# # construct a finite difference matrix for a single coordinate derivative over a 1D grid
# def FDmat1D(x):
#     dx=x[1]-x[0]
#     lx=x.size
    
#     lent=2*lx
#     ir=np.zeros(lent)
#     ic=np.zeros(lent)
#     L=np.zeros(lent)
    
#     ir[0]=0
#     ic[0]=0
#     L[0]=-1/dx
#     ir[1]=0
#     ic[1]=1
#     L[1]=1/dx
#     ient=2
#     for ix in range(1,lx-1):
#         ir[ient]=ix
#         ic[ient]=ix-1
#         L[ient]=-1/dx
#         ient=ient+1
#         ir[ient]=ix
#         ic[ient]=ix+1
#         L[ient]=1/dx
#         ient=ient+1        
#     ir[2*lx-1]=lx
#     ic[2*lx-1]=lx-1
#     L[2*lx-1]=-1/dx
#     ir[2*lx]=lx
#     ic[2*lx]=lx
#     L[2*lx]=-1/dx
    
#     L=scipy.sparse.coo_matrix(ir,ic,L)
#     return L


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


