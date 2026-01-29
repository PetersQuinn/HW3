#! /usr/bin/python3 -i
"""
CEE 251L Duke University
Problem 4: Constrained least squares curve fitting (adapted)
Fits y_hat(t;c) = c0 + c1 t^2 + c2 sin(2πt) + c3 cos(πt) + c4 exp(-(t^2))
with certainty constraints:
  y_hat(0) = -9,  y_hat'(0) = 25,  y_hat(10) = 56
"""

import numpy as np
import matplotlib.pyplot as plt

interactive = True  # Enable interactive mode for matplotlib

# -----------------------------
# Independent variable domain
# -----------------------------
t_init = 0.0
t_finl = 10.0
delt_t = 0.10
t = np.arange(t_init, t_finl + delt_t, delt_t)
M = len(t)

# -----------------------------
# "True" coefficients (given)
# -----------------------------
c_true = np.array([1.0, 0.5, 4.0, 5.0, -15.0])
n = len(c_true)

# -----------------------------
# Basis matrix B
# Columns: [1, t^2, sin(2πt), cos(πt), exp(-(t^2))]
# -----------------------------
B = np.column_stack(
    [
        np.ones(M),
        t**2,
        np.sin(2.0 * np.pi * t),
        np.cos(np.pi * t),
        np.exp(-(t**2)),
    ]
)

y_true = B @ c_true  # latent (unobservable) truth

# Simulated noisy measurements
sigma = 10.0
y_meas = y_true + sigma * np.random.randn(M)

# -----------------------------
# Unconstrained least squares
# -----------------------------
c_fit_u = np.linalg.solve(B.T @ B, B.T @ y_meas)
y_hat_u = B @ c_fit_u

# -----------------------------
# Constraints: A c = b
# y(0) = -9, y'(0) = 25, y(10) = 56
# -----------------------------
tc0 = 0.0
tc10 = 10.0

A = np.array(
    [
        # y(0) = c0 + c3 + c4
        [1.0, tc0**2, np.sin(2.0 * np.pi * tc0), np.cos(np.pi * tc0), np.exp(-(tc0**2))],
        # y'(t) = 2*c1*t + (2π)c2 cos(2πt) + (-π)c3 sin(πt) + (-2t)c4 exp(-t^2)
        [0.0, 2.0 * tc0, 2.0 * np.pi * np.cos(2.0 * np.pi * tc0), -np.pi * np.sin(np.pi * tc0), -2.0 * tc0 * np.exp(-(tc0**2))],
        # y(10)
        [1.0, tc10**2, np.sin(2.0 * np.pi * tc10), np.cos(np.pi * tc10), np.exp(-(tc10**2))],
    ],
    dtype=float,
)

b = np.array([-9.0, 25.0, 56.0], dtype=float)

# -----------------------------
# Constrained least squares via KKT
# -----------------------------
KKT = np.block([[2.0 * (B.T @ B), A.T], [A, np.zeros((3, 3))]])
rhs = np.concatenate([2.0 * (B.T @ y_meas), b])

sol = np.linalg.solve(KKT, rhs)
c_fit_c = sol[:n]
lam = sol[n:]

y_hat_c = B @ c_fit_c

# -----------------------------
# Errors / diagnostics
# -----------------------------
err_u = y_hat_u - y_true
err_c = y_hat_c - y_true

RMSE_u = np.sqrt(np.mean(err_u**2))
RMSE_c = np.sqrt(np.mean(err_c**2))

resid = A @ c_fit_c - b

# Display numerical solutions
np.set_printoptions(precision=3, suppress=True)
print("b =", b)
print("\n  true   unconstrained     constrained")
print(np.column_stack([c_true, c_fit_u, c_fit_c]))

print("\nRMS (unconstrained - true)  =", f"{RMSE_u:.3f}")
print("RMS (  constrained - true)  =", f"{RMSE_c:.3f}")
print("\nConstraint residual (A c - b) =", resid)

# -----------------------------
# Plot the results (Figure 1)
# -----------------------------
if interactive:
    plt.ion()

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.plot(t, y_true, linewidth=3, label="latent (unobservable truth)")
plt.plot(t, y_meas, "ok", markerfacecolor="none", markersize=5, label="measurements with error")
plt.plot(t, y_hat_u, "--", linewidth=3, label="unconstrained fit")
plt.plot(t, y_hat_c, linewidth=3, label="constrained fit")
plt.xlabel("independent variable, $t$", fontsize=15)
plt.ylabel("dependent variable, $y$", fontsize=15)
plt.legend(loc="best", fontsize=12)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.draw()
plt.pause(0.001)

# Save plot
pdf_name = "constrained-least-squares-1.pdf"
plt.savefig(pdf_name, bbox_inches="tight", dpi=300)
print(f"\nSaved: {pdf_name}")

# Enter to exit
if interactive:
    input("Press Enter to close all figures ... ")
    plt.close("all")