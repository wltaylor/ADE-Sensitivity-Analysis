import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special

# define the analytical solution for concentration in a type 1 boundary condition environment
def bc1(t, Co, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term + second_term)

    return C

# define the analytical solution for concentration in a type 1 boundary condition environment
def bc2(t, Co, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term - second_term)

    return C

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

# note: bc3 differs slightly from Veronica's notes. Van Genuchten describes the second term with v**2*t/Dl, Veron had v**2*L/Dl
# also VG had special.erfc(L + v *t....etc) while V had special.erfc(L - v * t....etc) I was getting negative concentration values with V's version

# inputs
Co = 1000
L = 25  # meters
t = np.arange(0, 1000, 1)
q = 0.056
p = 0.25
R = 1.0

#test1 = bc1(Co, L, t, q, p, R)
#test2 = bc2(Co, L, t, q, p, R)
#test3 = bc3(Co, L, t, q, p, R)

#plt.plot(t, test1, label='bc1')
#plt.plot(t, test2, label='bc2')
#plt.plot(t, test3, label='bc3')
#plt.xlabel('Time')
#plt.ylabel('Concentration')
#plt.title('Concentration vs Time')
#plt.legend()
#plt.show()


