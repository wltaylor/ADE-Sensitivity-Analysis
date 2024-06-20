import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special

# define the analytical solution for concentration in a type 2 boundary condition environment
def bc2(Co, L, t, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term - second_term)

    return C

# inputs
Co = 1000
L = 25  # meters
t = np.arange(0, 1000, 1)
q = 0.056
p = 0.25
R = 1.0

test = bc2(Co, L, t, q, p, R)

plt.plot(t, test)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration vs Time')
plt.show()