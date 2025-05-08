#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:14:13 2024

@author: williamtaylor
"""
import sys
import os

# Get the project root (one level up from the figure_scripts folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to Python’s search path
sys.path.append(project_root)

from model import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

times = np.linspace(1,1000,1000)

theta = 0.25
rho_b = 1.5
dispersivity = 0.5
v = 0.5
lamb = 0
alpha = 1
kd = 4
Co = 1
ts = 5
L = 2
x = 2

early_idx, peak_idx, late_idx, C_array, times_adaptive = model.concentration_102_all_metrics_adaptive(times,
                                                                                                      theta,
                                                                                                      rho_b,
                                                                                                      dispersivity,
                                                                                                      lamb,
                                                                                                      alpha,
                                                                                                      kd, 
                                                                                                      Co, 
                                                                                                      v, 
                                                                                                      ts, 
                                                                                                      L, 
                                                                                                      x)

plt.plot(times_adaptive, C_array, c='blue', alpha=0.5, label='Concentration')
plt.scatter(times_adaptive[early_idx], C_array[early_idx], c='red', marker='^', label='Early arrival', zorder=4, s=100)
plt.scatter(times_adaptive[peak_idx], C_array[peak_idx], c='black', marker='*', label='Peak concentration', zorder=4, s=100)
plt.scatter(times_adaptive[late_idx], C_array[late_idx], c='green', marker='v', label='Late time tailing', zorder=4, s=100)
plt.legend()
plt.xlabel('Dimensionless Time')
plt.ylabel('Normalized Concentration')
#plt.savefig('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/example_btc.pdf', format='pdf', bbox_inches='tight')

plt.show()

