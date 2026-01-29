#! /usr/bin/python3 -i

"""
CEE 251L Duke University
Example of constrained least squares curve fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import sin
from numpy import cos
from numpy import exp

interactive = True        # Enable interactive mode for matplotlib

# Values of independent variables, t1..tM
t_init = 0
t_finl = 5  # *
delt_t = 0.10
t = np.arange( t_init, t_finl+delt_t, delt_t)  

# Pre-selected latent values for the "true" coefficient 
c_true = np.array([5, 0.1, -4, 2, -20, 0.1])  # *

p = 1.5  # an exponent value

n = len(c_true)  # number of coefficients
M = len(t)  # number of measured data points

# Basis matrix
B = np.column_stack([ np.ones(M), t, sin(pi * t), cos(pi * t), exp(-t), t**p ])  # *

y_true = B @ c_true  # "true" values of y

# Constrained fit of y_hat to the measured data values

tc1 = 0  # value of t where y must be y_true(tc1) # *
tc2 = 5  # value of t where y must be y_true_tc2) and dy/dt must be d/dt(y_true(tc2) # *

# The constraint matrix A
A = np.array([
    [1, tc1,  sin(pi*tc1),     cos(pi*tc1),  exp(-tc1),   tc1**p    ], # *
    [0, 1, pi*cos(pi*tc2), -pi*sin(pi*tc2), -exp(-tc2), p*tc2**(p-1)]  # *
])

b = A @ c_true # values of [ b(1), b(2), b(3) ]  

print(f'b = {b}')

y_meas = y_true + 10.0 * np.random.randn(M)  # y + simulated measurement error

# Solve the unconstrained fit of y_hat to the measured data values

c_fit_u = np.linalg.solve(B.T @ B, B.T @ y_meas)  # optimum curve fit coefficients (5)

y_hat_u = B @ c_fit_u  # the model y_hat fit to the data

# Solve the constrained fit of y_hat to the measured data values

# The KKT matrix
KKT = np.block([
    [ 2*B.T @ B ,    A.T          ],
    [       A   , np.zeros((2,2)) ]                                    # *
])

rhs = np.block([  2*B.T @ y_meas , b ])       # right hand side vector

c_fit_c_lambda = np.linalg.solve( KKT, rhs )  # solve the KKT system (7)

c_fit_c = c_fit_c_lambda[:n]

y_hat_c = B @ c_fit_c

errr_u = y_hat_u - y_true
errr_c = y_hat_c - y_true

RMSE_u = np.sqrt( errr_u @ errr_u / len(errr_u) )
RMSE_c = np.sqrt( errr_c @ errr_c / len(errr_c) )

# Display the numerical solutions 
np.set_printoptions(precision=3)
print('  true   unconstrained     constrained')
print(np.column_stack([c_true, c_fit_u, c_fit_c]))

print('\n')
print(f' RMS (unconstrained - true)  = {RMSE_u:.3f}')
print(f' RMS (  constrained - true)  = {RMSE_c:.3f}')

# Plot the results 

# Set up plot formatting
if interactive:       
    plt.ion() # interactive plotting mode: on

plt.figure(1, figsize=(8,6))
plt.clf()
plt.plot(t, y_true, '-m', linewidth=3, label='latent (unobservable truth)')
plt.plot(t, y_meas, 'ok', linewidth=0.2, markerfacecolor='none',markersize=5, label='measurements with error')
plt.plot(t, y_hat_u, '--b', linewidth=3, label='unconstrained fit')
plt.plot(t, y_hat_c, '-g', linewidth=3, label='constrained fit')
plt.xlabel('independent variable, $t$', fontsize=15)
plt.ylabel('dependent variable, $y$', fontsize=15)
plt.legend(loc='lower right', fontsize=12)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.draw()
plt.pause(0.001)

# Save the plot to constrained-least-squares.pdf
filename = f'constrained-least-squares-1.pdf'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"\n  Saved: {filename}")

# Enter to Exit
if interactive: 
    input("Press Enter to close all figures ... ")
    plt.close('all')
