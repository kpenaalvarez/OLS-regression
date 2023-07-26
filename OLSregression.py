# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:37:29 2023

@author: upyou
Ordinary least squaes using old from statsmodels.api predict
point spread using line, and difference in days off
the theory being more days off means a ateam is better rested
and better prepared.

y is the response variable; actual spread(homescore-visitscore)
x1 is the line
x2 is the difference in days off

y= b0 + b1*x1 + b2*x2


"""
import statsmodels.api as sm
import pandas as pd
import numpy as np

#reading data from the csv
data = pd.read_csv('NFL2022.csv')

#defining the variables

x1 = data['line'].tolist()
y = (data['hScore'] - data['vScore']).tolist()
x2 = (data['hDoff'] - data['vDoff']).tolist()

A = np.array([x1,x2]).T
A.shape
A = sm.add_constant(A)
A.shape

model = sm.OLS(y,A)
results = model.fit()

results.params
print(results.summary())


