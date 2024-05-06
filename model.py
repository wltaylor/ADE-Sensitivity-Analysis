import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special

# define the analytical solution for concentration in a type 1 boundary condition environment
# base analytical solution
def bc1(t, Co, q, p, R):
    
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term + second_term)

    return C

# first arrival
# def first_arrival_bc1(t, Co, q, p, R):
    
#     time = np.arange(0,t,1)
#     concs = bc1(time, Co, q, p, R)
#     peak = np.max(concs)
#     arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
#     if arrival_indices.size > 0:
#         return time[arrival_indices[0]]
#     else:
#         return None

def first_arrival_bc1(t, Co, q, p, R): # t is an array of input times
    
    # check if t is a single number or an iterable
    if np.isscalar(t):
        t = np.array([t])
    
    concs = np.array([bc1(time, Co, q, p, R) for time in t])
    peak = np.max(concs)
    arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
    if arrival_indices.size > 0:
        return t[arrival_indices[0]]
    else:
        return np.nan


# peak concentration
def peak_conc_bc1(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc1(time, Co, q, p, R)
    peak = np.max(concs)
        
    return peak


# define the analytical solution for concentration in a type 2 boundary condition environment
def bc2(t, Co, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term - second_term)

    return C

# first arrival bc2
def first_arrival_bc2(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc2(time, Co, q, p, R)
    peak = np.max(concs)
    arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
    if arrival_indices.size > 0:
        return time[arrival_indices[0]]
    else:
        return None

# peak concentration bc2
def peak_conc_bc2(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc2(time, Co, q, p, R)
    peak = np.max(concs)
    peak_indices = np.where(concs >= 0.10 * peak)[0]
    
    if peak_indices.size > 0:
        return time[peak_indices[0]]
    else:
        return None

# define the analytical solution for concentration in a type 3 boundary condition environment
def bc3(t, Co, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t)/(2 * np.sqrt(Dl * t))) + np.sqrt((v**2 * t)/(np.pi*Dl)) * np.exp(-(L - v *t)**2/(4*Dl*t))
    second_term = 0.5*(1 + v*L/Dl + v**2*t/Dl) * np.exp(v*L/Dl) * special.erfc((L + v *t)/(2 * np.sqrt(Dl * t)))

    C = (Co/2) * (first_term - second_term)

    return C

# first arrival bc3
def first_arrival_bc3(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc3(time, Co, q, p, R)
    peak = np.max(concs)
    arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
    if arrival_indices.size > 0:
        return time[arrival_indices[0]]
    else:
        return None

# peak concentration bc3
def peak_conc_bc3(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc3(time, Co, q, p, R)
    peak = np.max(concs)
    peak_indices = np.where(concs >= 0.10 * peak)[0]
    
    if peak_indices.size > 0:
        return time[peak_indices[0]]
    else:
        return None

# note: bc3 differs slightly from Veronica's notes. Van Genuchten describes the second term with v**2*t/Dl, Veron had v**2*L/Dl
# also VG had special.erfc(L + v *t....etc) while V had special.erfc(L - v * t....etc) I was getting negative concentration values with V's version

# inputs
# Co = 1000
# L = 25  # meters
# t = np.arange(0, 10000, 1)
# q = 0.01
# p = 0.25
# R = 1.0

# test1 = bc1(t, Co, q, p, R)
# test2 = bc2(t, Co, q, p, R)
# test3 = bc3(t, Co, q, p, R)

# plt.plot(t, test1, label='bc1')
# plt.plot(t, test2, label='bc2')
# plt.plot(t, test3, label='bc3')
# plt.xlabel('Time')
# plt.ylabel('Concentration')
# plt.title('Concentration vs Time')
# plt.legend()
# plt.show()


#%% one dimensional first-type finite pulse BC (concentration Co for duration t)

def continuous_bc1(v,lamb,Dx,x,t,R,Ci):
    # v = velocity?
    # lamb(lambda) = first order rate constant
    # Dx = dispersion coefficient
    # x = position
    # t = time
    # R = retardation factor
    # Ci = initial concentration
    
    u = v*np.sqrt(1+(4*lamb*Dx/v**2))
    
    first_part = np.exp((v-u)*x/(2*Dx)) * special.erfc((R*x - u*t)/(2*np.sqrt(Dx*R*t))) + \
                 np.exp((v+u)*x/(2*Dx)) * special.erfc((R*x + u*t)/(2*np.sqrt(Dx*R*t)))
    
    second_part = special.erfc((R*x - v*t)/(2*np.sqrt(Dx*R*t))) + \
                  np.exp(v*x/Dx)*special.erfc((R*x + v*t)/(2*np.sqrt(Dx*R*t)))
    
    C = 0.5*Co*first_part - 0.5*Ci*np.exp(lamb*t/R)*second_part + Ci*np.exp(-lamb*t/R)
    
    return C, first_part, second_part



# v = 0.01
# lamb = 0.5
# Dx = 1.0**10-9
# t = np.arange(0,1000,1)
# R = 1


# test, first, second = continuous_bc1(v,lamb,Dx,2,t,R,0)
# plt.plot(test)



#%%
def concentration(x, t, C0, u, v, Dx, R, lt, Ci):
    term1 = (v - u) * x / (2 * Dx)
    term2 = special.erfc((R * x - u * t) / (2 * np.sqrt(Dx * R * t)))
    term3 = np.exp((v + u) * x / (2 * Dx)) * special.erfc((R * x + u * t) / (2 * np.sqrt(Dx * R * t)))
    term4 = np.exp(v * x / Dx) * special.erfc((R * x + v * t) / (2 * np.sqrt(Dx * R * t)))
    
    return 0.5 * C0 * (np.exp(term1) * term2 + term3) - 0.5 * Ci * np.exp(-lt * t / R) * term4 + Ci * np.exp(-lt * t / R)

# Example usage:
# C0, u, v, Dx, R, lt, Ci are constants that you will need to define based on your problem.
# x and t are the position and time for which you want to calculate the concentration.
# C_x_t = concentration(x=1, t=1, C0=1, u=1, v=1, Dx=1, R=1, lt=1, Ci=1)
# print(C_x_t)

#%% continuous injection Goltz page 37
def continuous(x, t, Co, v, R, lamb, Ci):
    
    diffusion = 1.0 * 10**-9 # diffusion constant
    alpha = 0.83 * np.log10(x)**2.414 # dispersivity
    Dx = (alpha * v + diffusion) / R # dispersion coefficient
    u = v*np.sqrt(1+(4*lamb*Dx)/v**2)

    term1 = np.exp((v - u) * x / (2 * Dx))
    term2 = special.erfc((R * x - u * t) / (2 * np.sqrt(Dx * R * t)))
    term3 = np.exp((v + u) * x / (2 * Dx))
    term4 = special.erfc((R * x + u * t) / (2 * np.sqrt(Dx * R * t)))
    
    term5 = special.erfc((R * x - v * t) / (2 * np.sqrt(Dx * R * t)))
    term6 = np.exp(v*x/Dx)
    term7 = special.erfc((R * x + v * t) / (2 * np.sqrt(Dx * R * t)))

    return 0.5*Co * (term1*term2+term3*term4) - 0.5*Ci*np.exp(-lamb*t/R)*(term5+term6*term7) + Ci*np.exp(-lamb*t/R)


# time = np.arange(0,30,1) # minutes, experiment time
# x = 2 # meters, column length
# Co = 1 # mg/m input concentration
# v = 0.1 # m/min
# R = 1 # retardation factor
# lamb = 0 # first order degradation constant
# Ci = 0 # initial concentration in system

# plt.plot(continuous(2, time, Co, v, R, lamb, Ci))
# plt.title('Concentration curve')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Concentration (mg/m)')
# plt.show()


#%% pulse injection Goltz page 39

def pulse(x, t_scalar, ts, Co, v, R, lamb, Ci):
    
    #diffusion = 1.0 * 10 ** -9  # diffusion constant
    diffusion = 0
    #alpha = 0.83 * np.log10(x) ** 2.414
    #alpha = 0.1
    #Dx = (alpha * v + diffusion) / R  # dispersion coefficient
    Dx = 1000
    u = v * np.sqrt(1 + (4 * lamb * Dx) / v ** 2)
    
    term1 = np.exp((v - u) * x / (2 * Dx))
    term2 = special.erfc((R * x - u * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))
    term3 = np.exp((v + u) * x / (2 * Dx))
    term4 = special.erfc((R * x + u * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))

    term5 = special.erfc((R * x - v * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))
    term6 = np.exp(v * x / Dx)
    term7 = special.erfc((R * x + v * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))
    
    # Ensure that we don't compute terms that involve negative square roots
    if t_scalar < ts:
        C = 0.5 * Co * (term1 * term2 + term3 * term4) - 0.5 * Ci * np.exp(-lamb * t_scalar / R) * (term5 + term6 * term7) + Ci * np.exp(-lamb * t_scalar / R)
        print('use first time at time: '+str(t_scalar))
    else:
        term8 = special.erfc((R * x - u * (t_scalar - ts)) / (2 * np.sqrt(Dx * R * (t_scalar - ts))))
        term9 = special.erfc((R * x + u * (t_scalar - ts)) / (2 * np.sqrt(Dx * R * (t_scalar - ts))))
        C = 0.5 * Co * (term1 * term2 + term3 * term4) - 0.5 * Ci * np.exp(-lamb * t_scalar / R) * (term5 + term6 * term7) + Ci * np.exp(-lamb * t_scalar / R) - \
            0.5 * Co * (term1 * term8 + term3 * term9)
        print('use second term at time: '+str(t_scalar))
    
    return C,Dx,u

# times = np.arange(0, 50, 1)  # minutes, experiment time
# test = np.zeros(len(times))
# testDx = np.zeros(len(times))
# testu = np.zeros(len(times))
# x = 0 # meters, column length
# Co = 1 # mg/m input concentration
# ts = 5 # min, pulse duration
# v = 0.1 # m/min
# R = 2 # retardation factor
# lamb = 0 # first order degradation constant
# Ci = 0 # initial concentration in system

# for i, time in enumerate(times):
#     print(time)
#     test[i],testDx[i],testu[i] = pulse(x, time, ts, Co, v, R, lamb, Ci)

# plt.plot(times, test)
# plt.title('Concentration Curve')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Concentration (mg/m)')
# plt.ylim(0,1)
# plt.show()

#%% Van Genuchten solution

from scipy.special import erfc

def A(x, t, v, D, R):
    if t > 0:
        term1 = erfc((R * x - v * t) / (2 * np.sqrt(D * R * t)))
        term2 = np.sqrt(v**2*t/(np.pi*D*R)) * np.exp(-(R*x - v*t)**2/(4*D*R*t))
        term3 = (1 + v*x/D + v**2*t/(D*R)) * np.exp(v*x/D) * erfc((R*x + v*t)/(2*np.sqrt(D*R*t)))
        return 0.5*term1 + term2 - 0.5*term3
    else:
        return 0
    
def B(x, t, v, D, R,gamma):
    if t > 0:
        term1 = t + 1/(2*v) * (1+ v*x/D + v**2*t/(D*R)) * erfc((R * x + v * t) / (2 * np.sqrt(D * R * t)))
        term2 = np.sqrt(t/(4*np.pi*D*R)) * (R*x + v*t + 2*D*R/v) * erfc((R * x - v * t) / (2 * np.sqrt(D * R * t)))
        term3 = (t/2 - (D*R)/(2*v**2)) * np.exp(v*x/D) * erfc((R * x + v * t) / (2 * np.sqrt(D * R * t)))
        return gamma / R * (term1 - term2 + term3)
    else:
        return 0
    

def pulse_concentration(x, t, Co, Ci, to, v, R, D, gamma):
    if t < to:
        C = Ci + (Co - Ci) * A(x, t, v, D, R) + B(x, t, v, D, R, gamma)
        return C
    if t == to:
        C = Ci + (Co - Ci) * A(x, t, v, D, R) + B(x, t, v, D, R, gamma)
        return C
    else:
        C = Ci + (Co - Ci) * A(x, t, v, D, R) + B(x, t, v, D, R, gamma) - Co * A(x, (t - to), v, D, R)
        return C

# Example usage:
# x = position where concentration is to be computed
# t = time at which concentration is to be computed
# Co = concentration of the pulse input
# Ci = initial concentration in the system
# to = duration of the pulse input
# v = velocity of the flow
# R = retardation factor
# D = dispersion coefficient

# Example parameters
# x = 2  # meters
# t = 5  # minutes
# Co = 1  # mg/m
# Ci = 0  # mg/m
# to = 5  # minutes
# v = 0.1  # m/min
# R = 1  # unitless
# D = 0.01  # m^2/min, for example
# gamma = 1

# # Compute concentration at x=2 m and t=10 min
# concentration = pulse_concentration(x, t, Co, Ci, to, v, R, D, gamma)
# print(concentration)

# times = np.arange(1, 50, 1)  # minutes, experiment time
# concentration = np.zeros(len(times))
# for i,time in enumerate(times):
#     concentration[i] = pulse_concentration(x, time, Co, Ci, to, v, R, D, gamma)

# plt.plot(times,concentration)

#%% model 102 goltz page 177

# def model_102(theta, rho_b, D, v, lamb, alpha, kd, Co, L, ts, s, x):
    
    
#     big_theta = s + lamb + (rho_b* alpha* kd * s)/(theta * (s + alpha))
    
#     r1 = 1/(2*D) * (v + np.sqrt((v**2 + 4*D*big_theta)))
#     r2 = 1/(2*D) * (v - np.sqrt((v**2 + 4*D*big_theta)))
    
#     term1 = (r2 * np.exp(r2*L + r1*x) - r1*np.exp(r1*L + r2*x))/(r2 * np.exp(r2*L) - r1*np.exp(r1*L))
    
#     C = 1/s * Co *(1 - np.exp(-ts*s)) * term1
    
#     return C, big_theta, r1, r2, term1

# theta = 0.25 # porous media porosity (unitless)
# rho_b = 1.5 # porous media bulk density (kg/m)
# D = 0.01 # dispersion (can't be zero)
# v = 0.1 # pore velocity (m/min)
# lamb = 0 # degradation first order rate constant
# alpha = 100
# kd = 0 # sorption distribution coefficient
# Co = 1 # initial concentration (mg/m)
# L = 2 # column length (m)
# ts = 5 # pulse duration (min)
# s = np.arange(0,50,1) # timesteps (min)
# x = 2 # temporal location (m) in this case, at the column outlet

# concentrations = np.zeros(len(s))
# big_theta_test = np.zeros(len(s))
# r1_test = np.zeros(len(s))
# r2_test = np.zeros(len(s))
# term1_test = np.zeros(len(s))

# for i,time in enumerate(s):
#     concentrations[i], big_theta_test[i],r1_test[i],r2_test[i],term1_test[i] = model_102(theta, rho_b, D, v, lamb, alpha, kd, Co, L, ts, time, x)
# plt.plot(s,concentrations)
# plt.show()


#%%

from mpmath import invertlaplace
import numpy as np
import matplotlib.pyplot as plt

from mpmath import mp, exp
mp.dps = 12

# def C_laplace(s, theta, rho_b, D, v, lamb, alpha, kd, Co, L, x):
#     big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    
#     # Here we use mp.sqrt and mp.exp from mpmath for multiprecision calculations
#     r1 = 1 / (2 * D) * (v + mp.sqrt(v ** 2 + 4 * D * big_theta))
#     r2 = 1 / (2 * D) * (v - mp.sqrt(v ** 2 + 4 * D * big_theta))
    
#     # Convert the arguments to mpmath types before calling exp
#     term1_numerator = r2 * exp(r2 * L + r1 * x) - r1 * exp(r1 * L + r2 * x)
#     term1_denominator = r2 * exp(r2 * L) - r1 * exp(r1 * L)
    
#     # Make sure the division is done using mpmath's div function for multiprecision calculations
#     term1 = mp.fdiv(term1_numerator, term1_denominator)
    
#     # Division should be handled by mpmath to maintain compatibility
#     C = mp.fdiv(Co, s) * (1 - exp(-ts * s)) * term1
    
#     return C
# # Define the parameters
# theta, rho_b, D, v, lamb, alpha, kd, Co, L, x = 0.25, 1.5, 0.0001, 0.1, 0, 1, 0, 1, 2, 2

# # Define a range of times for which you want the inverse transform
# times = np.linspace(1, 50, 100)

# # Perform the inverse Laplace transform numerically for each time value
# concentrations = [invertlaplace(lambda s: C_laplace(s, theta, rho_b, D, v, lamb, alpha, kd, Co, L, x), t, method='dehoog') for t in times]

# # Plot the result
# plt.plot(times, concentrations, label = 'v = 0.01 m/min')

# theta, rho_b, D, v, lamb, alpha, kd, Co, L, x = 0.25, 1.5, 0.05, 0.1, 0, 1, 0, 1, 2, 2
# concentrations2 = [invertlaplace(lambda s: C_laplace(s, theta, rho_b, D, v, lamb, alpha, kd, Co, L, x), t, method='dehoog') for t in times]

# plt.plot(times, concentrations2, label = 'v = 0.02 m/min')


# plt.xlabel('Time')
# plt.ylabel('Concentration')
# plt.title('Concentration vs. Time')
# plt.legend()
# plt.show()

# #%%

# velocities = np.linspace(0.01,0.5,10)

# for i,v in enumerate(velocities):
#     theta, rho_b, D, lamb, alpha, kd, Co, L, x = 0.25, 1.5, 0.0001, 0.25, 1, 0, 1, 2, 2
#     times = np.linspace(1, 50, 100)
#     concentrations = [invertlaplace(lambda s: C_laplace(s, theta, rho_b, D, v, lamb, alpha, kd, Co, L, x), t, method='dehoog') for t in times]
#     plt.plot(times, concentrations, label = 'v = '+str(v))
# plt.legend()
# plt.show()


def laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2):
    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    
    # Here we use mp.sqrt and mp.exp from mpmath for multiprecision calculations
    r1 = 1 / (2 * D) * (v + mp.sqrt(v ** 2 + 4 * D * big_theta))
    r2 = 1 / (2 * D) * (v - mp.sqrt(v ** 2 + 4 * D * big_theta))
    
    # Convert the arguments to mpmath types before calling exp
    term1_numerator = r2 * mp.exp(r2 * L + r1 * x) - r1 * mp.exp(r1 * L + r2 * x)
    term1_denominator = r2 * mp.exp(r2 * L) - r1 * mp.exp(r1 * L)
    
    # Make sure the division is done using mpmath's div function for multiprecision calculations
    term1 = mp.fdiv(term1_numerator, term1_denominator)
    
    # Division should be handled by mpmath to maintain compatibility
    C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
    return C

def concentration_102(t, theta, rho_b, D, v, lamb, alpha, kd, Co):
    concentration = [invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog') for time in t]
    print('transformed')
    return concentration

def laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2):
    
    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    delta = 1/(2*D) * mp.sqrt((v**2 + 4*D*big_theta))
    d = 2 * delta * L
    h = D/v
    sigma = v/(2*D)
    
    r1 = sigma + delta
    r2 = sigma - delta
    
    term1_numerator = r2 * mp.exp(r1 * x - d) - r1 * mp.exp(r2 * x)
    term1_denominator = r2 * (1 - h * r1) * mp.exp(-d) - (1 - h * r2)*r1
    
    term1 = mp.fdiv(term1_numerator, term1_denominator)
    
    C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
    return C

def concentration_106(t, theta, rho_b, D, v, lamb, alpha, kd, Co):
    concentration = [invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog') for time in t]
    print('transformed')
    return concentration




