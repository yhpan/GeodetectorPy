# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:10:33 2021
https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.ncf.html#scipy.stats.ncf
@author: pan
"""
import numpy as np
from osgeo import gdal_array
import time
from scipy.stats import ncf

# # 关于结果的检验
# from scipy.stats import ncf
# # N:539149723 L:57 q:0.4193889253456581 F:6954292.545475954 λ:341.05219264860443
# print(ncf.sf(6954292.545475954,539149723,57,341.05219264860443))

# interaction_detector
# risk_detector
# ecological_detector
lc = gdal_array.LoadFile('C:/Users/pan/Desktop/geodetector/smpledata2019EVILC/igbpLC_2019.tif')
evi = gdal_array.LoadFile('C:/Users/pan/Desktop/geodetector/smpledata2019EVILC/VNP13A1_EVI2019.tif') 
evi1 = gdal_array.LoadFile('C:/Users/pan/Desktop/geodetector/smpledata2019EVILC/VNP13A1_EVI2019.tif')
evi1 = evi1.reshape(4540*1645)
lc = lc.reshape(4540*1645).astype(np.uint8)
evi = evi.reshape(4540*1645)
evi[(-2000<evi) & (evi<=0)] =1 
evi[(0<evi) & (evi<=2000)] = 2
evi[(2000<evi) & (evi<=4000)] = 3
evi[(4000<evi) & (evi<=6000)] = 4
evi[(6000<evi) & (evi<=8000)] = 5
evi[(8000<evi) & (evi<=10000)] = 6
evi = evi.astype(np.uint8)
y,x1,x2 = evi1,evi,lc

def factor_detector(y,x,inval=0):
    # mask invalid pixels
    cond = (y!=inval)
    y,x = y[cond],x[cond]
    cond = (x!=inval)
    y,x = y[cond],x[cond]
    # reshape y,x1,x2 to 1 columns
    y,x = y.reshape(-1,1),x.reshape(-1,1)
    # get unique class value's combination of x1 and x2 
    g = np.unique(x)
    g = g.reshape(-1,1)
    # rows and columns of g
    row,col = g.shape
    # statistics of population
    pop = [np.size(y), np.mean(y), np.var(y,ddof=0)]
    # statistics of samples
    sample=[]
    for i in range(row):
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
    prob = ncf.sf(F,L-1,N-L, λ)
    
    return q, prob

def interaction_detector(y,x1,x2,inval=0):    
    # mask invalid pixels
    cond = (y!=inval)
    y,x1,x2 = y[cond],x1[cond],x2[cond]
    cond = (x1!=inval)
    y,x1,x2 = y[cond],x1[cond],x2[cond]
    cond = (x2!=inval)
    y,x1,x2 = y[cond],x1[cond],x2[cond]
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
    prob = ncf.sf(F,L-1,N-L, λ)
    
    return q, prob

combine = np.array(list(zip(lc,evi)))
e = np.unique(combine,axis=0)
# f = np.sum(combine == e[45],axis=1) ==2
# np.sum(evi1[f])
pop = [np.count_nonzero(evi1), np.mean(evi1), np.var(evi1,ddof=0)]
sample=[]
for i in range(57):
    f = np.sum(combine == e[i],axis=1) ==2
    pxs = np.count_nonzero(evi1[f])
    avg = np.mean(evi1[f])
    var = np.var(evi1[f],ddof=0)
    
    sample.append([i,pxs,avg,var])
### 开始计算地理探测器q值
sample = np.array(sample)
# 各层个数与方差乘积之和
a = np.sum(sample[:,1]*sample[:,3])
# 总体个数与方差乘积
b = pop[0]*pop[2]
# 地理探测器q值
q = 1 - a/b
### 对q值进行检验
N, L = pop[0], sample.shape[0]
F = ((N-L)*q) / ((L-1)*(1-q))
# 各层平均值平方和
p1 = np.sum(sample[:,2]*sample[:,2])
p2 = np.square(np.sum(np.sqrt(sample[:,1])*sample[:,2]))
λ = (p1 -p2/pop[0]) / pop[2]
# print('q: ',q,'F:',F,'λ: ',λ)    


# Y = np.random.rand(100).reshape(10,10)
# A = np.random.rand(100).reshape(4,5,5)
# a, s,p = np.unique(A, return_index=True, return_inverse=True)
# print(a)
# print(s)
# print(p)

#### dem计算的q
# # 计算总体的数据，个数、均值、方差
# ndvi = ndvi_arr[ndvi_arr>0]
# pop = [np.count_nonzero(ndvi), np.mean(ndvi), np.var(ndvi,ddof=0)]
# #保存类个数、均值、方差用于计算q并进行检验
# sample=[]
# for elev in range(100,5701,100): 
#     time1=time.time()
           
#     bin = (dem_arr > elev-50) & (dem_arr <= elev+50) #dem区间
#     ndvi_bin = ndvi_arr[bin]#提取bin内NDVI像元
#     ndvi_bin = ndvi_bin[ndvi_bin>0]
    
#     # 计算个数、均值、方差用于计算q并进行检验
#     # 总体(母体)标准差，参数ddof = 0
#     # 样本标准差，参数ddof = 1
    
#     pxs = np.count_nonzero(ndvi_bin)
#     avg = np.mean(ndvi_bin)
#     var = np.var(ndvi_bin,ddof=0)
    
#     sample.append([elev,pxs,avg,var])
#     time2=time.time()
#     print(elev,' costed: ',time2-time1)

# ### 开始计算地理探测器q值
# sample = np.array(sample)
# # 各层个数与方差乘积之和
# a = np.sum(sample[:,1]*sample[:,3])
# # 总体个数与方差乘积
# b = pop[0]*pop[2]
# # 地理探测器q值
# q = 1 - a/b
# ### 对q值进行检验
# N, L = pop[0], sample.shape[0]
# F = ((N-L)*q) / ((L-1)*(1-q))
# # 各层平均值平方和
# p1 = np.sum(sample[:,2]*sample[:,2])
# p2 = np.square(np.sum(np.sqrt(sample[:,1])*sample[:,2]))
# λ = (p1 -p2/pop[0]) / pop[2]
# print('q: ',q,'F:',F,'λ: ',λ)
# time_e=time.time()
# print('all costed: ', time_e-time_s)

# 因子交互
a = np.array([1,2,2,3,6,4,8,8,7,2])
b = np.array([2,1,2,3,5,5,7,8,8,2])
c = list(zip(a,b))
d = np.array(c)
e = np.unique(d,axis=0)
f = np.sum(d == e[0],axis=1) ==2
