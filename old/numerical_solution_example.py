#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:34:17 2024

@author: williamtaylor
"""

import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator_analytical(t, A, omega, phi):
    return A * np.cos(omega * t + phi)

# Constants
m = 1.0  # mass
k = 1.0  # spring constant

# Compute angular frequency
omega = np.sqrt(k / m)

# Initial conditions
A = 1.0  # amplitude
phi = 0.0  # phase angle

# Time array
t = np.linspace(0, 10, 1000)

# Analytical solution
x_analytical = harmonic_oscillator_analytical(t, A, omega, phi)

def harmonic_oscillator_numerical(m, k, x0, v0, dt, num_steps):
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    
    x[0] = x0
    v[0] = v0
    
    for i in range(1, num_steps):
        v[i] = v[i-1] - (k / m) * x[i-1] * dt
        x[i] = x[i-1] + v[i] * dt
    
    return x

# Initial conditions
x0 = A  # initial displacement
v0 = 0.0  # initial velocity

# Time parameters
dt = 0.01  # time step
num_steps = int(10 / dt)  # number of time steps

# Numerical solution
x_numerical = harmonic_oscillator_numerical(m, k, x0, v0, dt, num_steps)


plt.figure(figsize=(8, 6))
plt.plot(t, x_analytical, label='Analytical Solution')
plt.plot(np.linspace(0, 10, num_steps), x_numerical, label='Numerical Solution')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Analytical vs Numerical Solution of Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0         # Length of the domain
T = 1.0         # Total time
Nx = 100        # Number of spatial grid points
Nt = 1000       # Number of time steps
D = 0.01        # Dispersion coefficient
v = 0.1         # Advection velocity
k = 0.01        # Reaction rate

# Spatial and temporal grid
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# Time step based on CFL condition
dt = min(0.9 * dx**2 / (2 * D), 0.9 * dx / v, T / Nt)
Nt = int(T / dt)

# Initialize concentration array
C = np.zeros((Nt, Nx))

# Initial condition (pulse injection)
C[0, int(0.2*Nx):int(0.4*Nx)] = 1.0

# Numerical solution using explicit finite difference method
for n in range(Nt-1):
    for i in range(1, Nx-1):
        C[n+1, i] = C[n, i] + D * dt / dx**2 * (C[n, i+1] - 2*C[n, i] + C[n, i-1]) \
                    - v * dt / (2*dx) * (C[n, i+1] - C[n, i-1]) + k * C[n, i] * dt

    # Boundary conditions
    C[n+1, 0] = C[n+1, 1]
    C[n+1, -1] = C[n+1, -2]

# Plot the concentration profile
plt.figure(figsize=(8, 6))
plt.imshow(C, aspect='auto', extent=[0, L, 0, T], cmap='jet', origin='lower')
plt.colorbar(label='Concentration')
plt.xlabel('Distance')
plt.ylabel('Time')
plt.title('Numerical Solution of Advection-Dispersion-Reaction Equation')
plt.show()

#%%

# Parameters
L = 1.0         # Length of the domain
T = 1000.0         # Total time
Nx = 100        # Number of spatial grid points
D = 0.01        # Dispersion coefficient
v = 0.1         # Advection velocity
k = 0.01        # Reaction rate

# Spatial and temporal grid
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# Time step based on CFL condition
dt = min(0.9 * dx**2 / (2 * D), 0.9 * dx / v, T / Nt)
Nt = int(T / dt)

# Initialize concentration array
C = np.zeros((Nt, Nx))

# Initial condition (pulse injection)
C[0, int(0.2*Nx):int(0.4*Nx)] = 1.0

# Numerical solution using explicit finite difference method
for n in range(Nt-1):
    for i in range(1, Nx-1):
        C[n+1, i] = C[n, i] + D * dt / dx**2 * (C[n, i+1] - 2*C[n, i] + C[n, i-1]) \
                    - v * dt / (2*dx) * (C[n, i+1] - C[n, i-1]) + k * C[n, i] * dt

    # Boundary conditions
    C[n+1, 0] = C[n+1, 1]
    C[n+1, -1] = C[n+1, -2]

# Select the point near the end of the domain for the breakthrough curve
observation_point = int(0.9 * Nx)

# Extract the concentration at the observation point over time
breakthrough_curve = C[:, observation_point]

# Plot the breakthrough curve
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, T, Nt), breakthrough_curve, label=f'Breakthrough Curve at x={x[observation_point]:.2f}')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Breakthrough Curve of Advection-Dispersion-Reaction Equation')
plt.legend()
plt.grid(True)
plt.show()