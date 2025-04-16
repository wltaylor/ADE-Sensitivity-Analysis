#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 08:47:49 2025

@author: williamtaylor
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model

times = np.linspace(0,20,1000)

L = 2
x = 2
ts = 0.25
v = 1
Co = 1
theta = 0.7
rho_b = 1.5
dispersivity = 4
lamb = 0
alpha = 1
kd = 1

concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,theta,rho_b,dispersivity,lamb,alpha,kd, Co=Co, v=v, ts=ts, L=L, x=x)

plt.plot(adaptive_times, concentrations, c='black')
plt.xlabel('Time')
plt.ylabel('Concentration (Co/C)')
plt.ylim(0,0.1)
plt.xlim(0,25)
plt.tight_layout()
plt.show()