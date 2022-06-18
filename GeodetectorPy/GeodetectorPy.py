# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:10:33 2021
https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.ncf.html#scipy.stats.ncf
@author: pan
"""
import numpy as np
from scipy import stats

def factor_detector(y,x):
    '''
    The factor detector q-statistic measures the SSH of a variable Y,
    or the determinant power of a covariate X of Y.

    Parameters
    ----------
    y : ndarray
        explained variable.
    x : ndarray
        explanatory variable.

    Returns
    -------
    q : float
        q statistic.
    prob : float
        corresponding p value.

    '''
    # reshape y,x to 1 columns
    y,x = y.reshape(-1,1),x.reshape(-1,1)
    # get unique class value's of x
    g = np.unique(x)
    # statistics of population
    pop = [np.size(y), np.mean(y), np.var(y,ddof=0)]
    # statistics of samples
    sample=[]
    for i in range(np.size(g)):
        cond = x==g[i]
        pxs = np.size(y[cond])
        avg = np.mean(y[cond])
        var = np.var(y[cond],ddof=0)
        
        sample.append([i,pxs,avg,var])
    
    ### start to calculate geodetector q
    sample = np.array(sample)
    # the sum of product of quantities and variance of each stratification
    a = np.sum(sample[:,1]*sample[:,3])
    # the product of quantities and variance of popolation
    b = pop[0]*pop[2]
    # the value of geodetector q
    q = 1 - a/b
    
    ### test of q 
    N, L = pop[0], sample.shape[0]
    F = ((N-L)*q) / ((L-1)*(1-q))
    # the sum of square of each stratification
    p1 = np.sum(sample[:,2]*sample[:,2])
    p2 = np.square(np.sum(np.sqrt(sample[:,1])*sample[:,2]))
    λ = (p1 -p2/pop[0]) / pop[2]
    prob = stats.ncf.sf(F,L-1,N-L, λ)
    
    return q, prob

def interaction_detector(y,x1,x2):   
    '''
    The interaction detector reveals whether the risk factors X1 and X2 (and more X) 
    have an interactive influence on a disease Y.

    Parameters
    ----------
    y : ndarray
        explained variable.
    x1 : ndarray
        first explanatory variables.
    x2 : ndarray
        second explanatory variables.

    Returns
    -------
    q : float
        q statistic.
    prob : float
        corresponding p value.

    '''
    # reshape y,x1,x2 to 1 columns
    y,x1,x2 = y.reshape(-1,1),x1.reshape(-1,1),x2.reshape(-1,1)
    # horizontal stack x1 and x2
    combine = np.hstack((x1, x2))
    # get unique combination class value of x1 and x2 
    g = np.unique(combine,axis=0)
    # rows and columns of g
    row,col = g.shape
    
    # statistics of population
    pop = [np.size(y), np.mean(y), np.var(y,ddof=0)]
    # statistics of samples
    sample=[]
    for i in range(row):
        # combine == g[i] return a array with the same shape of combine
        # if combine[n,0] == g[i][0] then cond[n,0]=True
        # so when the sum of combine equal to 2 along horizontal axis, it means exact matching
        cond = np.sum(combine == g[i],axis=1) ==2
        pxs = np.size(y[cond])
        avg = np.mean(y[cond])
        var = np.var(y[cond],ddof=0)
        
        sample.append([i,pxs,avg,var])
        
    ### start to calculate geodetector q
    sample = np.array(sample)
    # the sum of product of quantities and variance of each stratification
    a = np.sum(sample[:,1]*sample[:,3])
    # the product of quantities and variance of popolation
    b = pop[0]*pop[2]
    # the value of geodetector q
    q = 1 - a/b
    
    ### test of q 
    N, L = pop[0], sample.shape[0]
    F = ((N-L)*q) / ((L-1)*(1-q))
    # the sum of square of each stratification
    p1 = np.sum(sample[:,2]*sample[:,2])
    p2 = np.square(np.sum(np.sqrt(sample[:,1])*sample[:,2]))
    λ = (p1 -p2/pop[0]) / pop[2]
    prob = stats.ncf.sf(F,L-1,N-L, λ)
    
    return q, prob

def risk_detector(y,x):
    '''
    The risk detector calculates the average values in each stratum of explanatory variable (X), 
    and presents if there exists difference between two strata.

    Parameters
    ----------
    y : ndarray
        explained variable.
    x : ndarray
        explanatory variable.

    Returns
    -------
    mean_sub_y : two columns ndarray 
        first column is types of explanatory variable x,
        second column is corresponding average explained variable y of each x types.
    sig_t_p : NxN ndarray
        the p value of t-test that presents if there exists difference between two strata.
        rows and columns stand for each x types

    '''
    # reshape y,x to 1 columns
    y,x = y.reshape(-1,1),x.reshape(-1,1)
    # get unique class value's of x
    g = np.unique(x)
    # statistics of samples
    sample=[]
    for i in range(np.size(g)):
        cond = x==g[i]
        pxs = np.size(y[cond])
        avg = np.mean(y[cond])
        var = np.var(y[cond],ddof=0)        
        sample.append([i,g[i],pxs,avg,var])
        
    ### start risk detector processes
    sample = np.array(sample)
    mean_sub_y = np.hstack((sample[:,1].reshape(-1,1),sample[:,3].reshape(-1,1)))
    sig_t_p = np.ones((np.size(g),np.size(g)))
    
    for row in range(np.size(g)):        
        for col in range(row+1,np.size(g)):
            a1 = sample[row,4]/sample[row,2]
            a2 = sample[col,4]/sample[col,2]
            t_ij = (sample[row,3]-sample[col,3]) / np.sqrt(a1 + a2) #t-statistics
            df_ij = (a1 + a2)/((a1**2)/(sample[row,2]-1) + (a2**2)/(sample[col,2]-1)) # degree of freedom
            prob_ij = stats.t.sf(np.abs(t_ij), df_ij)*2 #t-test            
            sig_t_p[row,col], sig_t_p[col,row] = prob_ij, prob_ij 
               
    return mean_sub_y, sig_t_p

def ecological_detector(y,X):
    '''
    The ecological detector tests whether there is significant differences between two risk factors X1 ~ X2.

    Parameters
    ----------
    y : ndarray
        explained variable.
    X : MxN ndarray
        each column stands for one explanatory variable.

    Returns
    -------
    sig_f_p : NxN ndarray
        the p value of F test that presents whether there is significant differences between risk factors X1, X2, X3, ...
        rows and columns stand for each risk factors.

    '''
    # reshape yto 1 columns
    y = y.reshape(-1,1)
    row,col = X.shape
    
    sig_f_p = np.ones((col, col))
    for i in range(col):
        for j in range(i+1,col):
            x1 = X[:,i].reshape(-1,1)
            x2 = X[:,j].reshape(-1,1)
            # get unique class value's of x1 and x2
            g1, g2 = np.unique(x1), np.unique(x2)
            
            SSWx1 = np.sum([np.var(y[(x1==g)],ddof=0) for g in g1])
            SSWx2 = np.sum([np.var(y[(x2==g)],ddof=0) for g in g2])
            
            # Nx1, Nx2 = np.size(g1), np.size(g2)
            Nx1, Nx2 = row, row
            F = (Nx1*(Nx2-1)*SSWx1)/(Nx2*(Nx1-1)*SSWx2)
            prob = stats.f.sf(F, Nx1-1, Nx2-1)
            
            sig_f_p[i,j], sig_f_p[j,i] = prob, prob
        
    return sig_f_p

if __name__=='__main__':
    
    # import pandas as pd
    # # ds = pd.read_csv('./GeoDetector_2015_Example(Toy Dataset).csv')
    # # ds = pd.read_csv('./GeoDetector_2015_Example(NDVI Dataset).csv')
    # ds = pd.read_csv('./GeoDetector_2018_Example(Disease Dataset).csv')
    
    # factor_detector(ds.incidence.values,ds.level.values)
    # interaction_detector(ds.incidence.values, ds.type.values, ds.region.values)
    # risk_detector(ds.incidence.values,ds.level.values)
    
    # X = np.hstack((ds.type.values.reshape(-1,1), ds.region.values.reshape(-1,1), ds.level.values.reshape(-1,1)))
    # ecological_detector(ds.incidence.values, X)
    
    print('geodetector python')
    
    
    
    
    
    
    
    
