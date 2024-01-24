#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:41:31 2024

@author: williamtaylor
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import model
from SALib.sample import saltelli
from SALib.analyze import sobol

# define a dictionary of the model inputs (ranges to be changed based on literature findings)
problem = {
    'num_vars':4,
    'names': ['Initial Concentration','Unit Discharge','Porosity','Retardation'],
    'bounds': [[1000,1500],
               [0.01, 1],
               [0.125,0.5],
               [0.5,4]]
}

# generate samples using the saltelli sampler (change to larger value for more accurate SA indices)
param_values = saltelli.sample(problem, 2**8)

Y = np.zeros([param_values.shape[0]]) # model outputs
t = 1000 #days, choose a fixed time to evaluate the model at
L = 25 # fixed length

# evaluate the model
for i, X in enumerate(param_values):
    Y[i] = model.bc3(t, X[0], X[1], X[2], X[3])
    
Si = sobol.analyze(problem, Y, print_to_console = True)

plt.figure(figsize = (14,6))
Si.plot()
plt.tight_layout()