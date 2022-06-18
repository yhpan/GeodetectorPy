import sys
sys.path.append('./GeodetectorPy')
import pandas as pd
import numpy as np
import GeodetectorPy

ds = pd.read_csv('./data/GeoDetector_2018_Example(Disease Dataset).csv')

q, prob = GeodetectorPy.factor_detector(ds.incidence.values,ds.level.values)
print('factor_detector of level\n','q: {} \n p-value: {}'.format(q, prob))

q, prob = GeodetectorPy.interaction_detector(ds.incidence.values, ds.type.values, ds.region.values)
print('interaction_detector of type & region\n','q: {} \n p-value: {}'.format(q, prob))

mean_sub_y, sig_t_p = GeodetectorPy.risk_detector(ds.incidence.values,ds.level.values)
print('risk_detector\n','mean\n',mean_sub_y)
print('sig_t_p\n',sig_t_p)

X = np.hstack((ds.type.values.reshape(-1,1), ds.region.values.reshape(-1,1), ds.level.values.reshape(-1,1)))
sig_f_p = GeodetectorPy.ecological_detector(ds.incidence.values, X)
print('ecological_detector\n',sig_f_p)
