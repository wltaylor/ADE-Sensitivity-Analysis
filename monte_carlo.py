#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:30:33 2024

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
import model
from SALib.sample import saltelli
import seaborn as sns
import matplotlib.patches as mpatches
from pandas.plotting import table
#%%
# create a group of parameters to sample from
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.001, 2], # D
               [0.001, 1], # v
               [0, 0.5], # lamb
               [0, 1], # alpha
               [0, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**7)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

# plotting
plt.figure(figsize=(10,8))
plt.style.use('ggplot')  # Applying ggplot style

plt.subplot(2,2,1)
sns.kdeplot(params_df['Pe'], c='black')
plt.axvline(1, c = 'red')
plt.xscale('log')
plt.title('Sampled Peclet #')

plt.subplot(2,2,2)
sns.kdeplot(params_df['Da'], c='black')
plt.axvline(1, c = 'red')
plt.xscale('log')
plt.title('Sampled Dahmkoler #')


plt.subplot(2,2,3)
peclet_counts = params_df['Pe'].apply(lambda x: 'Dispersion Dominated' if x < 1 else 'Advection Dominated').value_counts()
peclet_counts = peclet_counts.reindex(['Dispersion Dominated', 'Advection Dominated'])
plt.bar(peclet_counts.index, peclet_counts.values, color=['blue','red'])
plt.title('System count by Peclet #')

plt.subplot(2,2,4)
dahmkoler_counts = params_df['Da'].apply(lambda x: 'Transport Dominated' if x < 1 else 'Reaction Dominated').value_counts()
dahmkoler_counts = dahmkoler_counts.reindex(['Transport Dominated', 'Reaction Dominated'])
plt.bar(dahmkoler_counts.index, dahmkoler_counts.values, color=['orange','purple'])
plt.title('System count by Dahmkoler #')

plt.suptitle('Parameter Ranges', fontsize=18)
plt.tight_layout()

# column_labels = ['Parameter', 'Minimum', 'Maximum']
# column_widths = [0.5,0.2,0.2]
# cell_text = [
#     ['Theta', str(params_df['theta'].min()), str(params_df['theta'].max())],
#     ['Bulk Density', str(params_df['rho_b'].min()), str(params_df['rho_b'].max())],
#     ['Dispersion', str(params_df['D'].min()), str(params_df['D'].max())],
#     ['Pore Velocity', str(params_df['v'].min()), str(params_df['v'].max())],
#     ['Lambda', str(params_df['lamb'].min()), str(params_df['lamb'].max())],
#     ['Alpha', str(params_df['alpha'].min()), str(params_df['alpha'].max())],
#     ['Sorption coefficient', str(params_df['kd'].min()), str(params_df['kd'].max())],
#     ['Initial Concentration', str(params_df['Co'].min()), str(params_df['Co'].max())]
# ]

# plt.table(cellText=cell_text,
#           colLabels=column_labels,
#           colWidths=column_widths,
#           cellLoc='center',
#           loc='center right',
#           bbox=[1.1,0,0.5,2.5])
# #table.auto_set_font_size(False)
# #table.set_fontsize(8)
# #table.scale(1.2, 1.2)
plt.show()



print('Peclet range: ' +str(params_df['Pe'].min()) +' to ' + str(params_df['Pe'].max()))
print('Dahmkoler range: ' +str(params_df['Da'].min()) +' to ' + str(params_df['Da'].max()))

#%% create BTCs from sampled parameters

times = np.linspace(1, 50, 50)
btc_data = []
legend_labels_pe = []
legend_labels_da = []

for i in range(len(params_df)):
    theta = params_df.iloc[i, 0]
    rho_b = params_df.iloc[i, 1]
    D = params_df.iloc[i, 2]
    v = params_df.iloc[i, 3]
    lamb = params_df.iloc[i, 4]
    alpha = params_df.iloc[i, 5]
    kd = params_df.iloc[i, 6]
    Co = params_df.iloc[i, 7]

    # Calculate Peclet number and Dahmkoler number
    Pe = v * 2 / D
    if Pe > 1:
        Da = lamb * 2 / v
    else:
        Da = lamb * 2**2 / D
    # Generate BTC
    concs = np.array([model.concentration_102(times, theta, rho_b, D, v, lamb, alpha, kd, Co)])
    
    # Append concentration data to btc_data
    btc_data.append(concs)

    # Store legend labels
    if Pe > 1 and 'Advection Dominated' not in legend_labels_pe:
        legend_labels_pe.append('Advection Dominated')
    elif Pe <= 1 and 'Dispersion Dominated' not in legend_labels_pe:
        legend_labels_pe.append('Dispersion Dominated')


    if Da > 1 and 'Reaction Dominated' not in legend_labels_da:
        legend_labels_da.append('Reaction Dominated')
    elif Da <= 1 and 'Transport Dominated' not in legend_labels_da:
        legend_labels_da.append('Transport Dominated')
        
# Convert btc_data to numpy array and flatten
btc_data = np.array(btc_data).reshape(len(params_df), -1)

# Plotting BTCs
plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
for i, concs in enumerate(btc_data):
    # Assign color based on Peclet number
    Pe = params_df.iloc[i, 3] * 2 / params_df.iloc[i, 2]
    color = 'red' if Pe > 1 else 'blue'

    # Plot BTC
    plt.plot(times, concs, alpha=0.3, color=color)

legend_handles_pe = [mpatches.Patch(color='red', alpha=0.3), mpatches.Patch(color='blue', alpha=0.3)]
plt.legend(legend_handles_pe, ['Advection Dominated', 'Dispersion Dominated'])
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m)')
plt.title('Peclet #')

plt.subplot(1,2,2)

for i, concs in enumerate(btc_data):
    # Assign color based on Peclet number
    Pe = params_df.iloc[i, 3] * 2 / params_df.iloc[i, 2]
    if Pe > 1:
        Da = params_df.iloc[i,4] * 2 / params_df.iloc[i,3]
    else:
        Da = params_df.iloc[i,4] * 2**2 / params_df.iloc[i,2]
    color = 'purple' if Da > 1 else 'orange'

    # Plot BTC
    plt.plot(times, concs, alpha=0.3, color=color)

legend_handles_da = [mpatches.Patch(color='purple', alpha=0.3), mpatches.Patch(color='orange', alpha=0.3)]
plt.legend(legend_handles_da, ['Reaction Dominated', 'Transport Dominated'])

# Adding labels and title
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m)')
plt.title('Dahmkoler #', fontsize=12)

plt.subplots_adjust(top=0.85)
#plt.tight_layout()
plt.suptitle('Contaminant Breakthrough Curves', fontsize=18)

# Displaying the plot
plt.show()

#%%

# times = np.linspace(1,80,80)

# theta = 0.25
# rho_b = 1.5
# D = 0.05
# v = 0.01
# lamb = 0
# alpha = 1
# kd = 0
# Co = 1
# ts = 5
# L = 2
# x = 2

# print(v*2/D)

# concs_102 = model.concentration_102(times,theta,rho_b,D,v,lamb,alpha,kd,Co) # generates a list of mpf objects
# concs_102_float = [float(conc) for conc in concs_102] # convert to floats
# concs_102_array = np.array(concs_102_float) # convert to array

# plt.plot(times, concs_102_array, label = 'dispersion dominated')

# theta = 0.25
# rho_b = 1.5
# D = 0.01
# v = 1
# lamb = 0
# alpha = 0
# kd = 0
# Co = 1
# ts = 5
# L = 2
# x = 2

# print(v*2/D)

# concs_102 = model.concentration_102(times,theta,rho_b,D,v,lamb,alpha,kd,Co) # generates a list of mpf objects
# concs_102_float = [float(conc) for conc in concs_102] # convert to floats
# concs_102_array = np.array(concs_102_float) # convert to array
# plt.plot(times, concs_102_array, label = 'advection dominated')

# plt.ylim(0,1)
# plt.legend()
# plt.show()

