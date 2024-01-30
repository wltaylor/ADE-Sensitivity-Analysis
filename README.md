# ADE-Sensitivity-Analysis
Sensitivity analysis of type 1, 2 and 3 boundary condition advection dispersion equations. This experiment uses the popular Sobol sensitivity analyis method to determine sensitivity indices for each parameter in the three forms of the ADE. Sensitivity indices are calculated using the SALib package (Herman et al 2017).


# Analytical Solutions
The analytical solution for boundary condition 1 (Ogata and Banks 1961) is given by:

![Equation](https://latex.codecogs.com/svg.image?{\color{White}C(t)=\frac{C_0}{2}\left[\text{erfc}\left(\frac{L-vt}{2\sqrt{D_l&space;t}}\right)&plus;\exp\left(\frac{vL}{D_l}\right)\text{erfc}\left(\frac{L&plus;vt}{2\sqrt{D_l&space;t}}\right)\right]})

The analytical solution for boundary condition 2 (Sauty 1980) is given by:

![Equation](https://latex.codecogs.com/svg.image?{\color{White}C(t)=\frac{C_0}{2}\left[\text{erfc}\left(\frac{L-vt}{2\sqrt{D_l&space;t}}\right)-\exp\left(\frac{vL}{D_l}\right)\text{erfc}\left(\frac{L&plus;vt}{2\sqrt{D_l&space;t}}\right)\right]})

The analytical solution for boundary condition 3 (Van Genuchten 1984) is given by:

![Equation](https://latex.codecogs.com/svg.image?{\color{White}C(t)=\frac{C_0}{2}\left[\text{erfc}\left(\frac{L-vt}{2\sqrt{D_l&space;t}}\right)&plus;\sqrt{\frac{v^2&space;t}{\pi&space;D_l}}\exp\left(-\frac{(L-vt)^2}{4D_l&space;t}\right)-\frac{1}{2}\left(1&plus;\frac{vL}{D_l}&plus;\frac{v^2t}{D_l}\right)\exp\left(\frac{vL}{D_l}\right)\text{erfc}\left(\frac{L&plus;vt}{2\sqrt{D_l&space;t}}\right)\right])

where:
- \( C(t) \) is the concentration at time \( t \).
- \( C_0 \) is the initial concentration.
- \( v \) is the velocity, calculated as \( \frac{q}{pR} \).
- \( D_l \) is the dispersion length, calculated as \( \frac{\alpha v + \text{diffusion}}{R} \).
- \( \alpha \) is a coefficient dependent on \( L \), calculated as \( 0.83 \times \log_{10}(L)^{2.414} \).
- \( \text{diffusion} \) is a constant, typically around \( 1.0 \times 10^{-9} \).
- \( L \), \( q \), \( p \), and \( R \) are parameters of the system.


# Packages


# License
