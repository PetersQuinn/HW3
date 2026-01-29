# qprog.py
# Purpose: Solve (e), (h), (i), (j) for Homework 3 Problem 3 (Quadratic Programming)
# Name: Quinn Peters

import numpy as np


def f(v: np.ndarray) -> float:
    """Objective function f(v1,v2,v3)."""
    v1, v2, v3 = v
    return (
        12 * v1**2
        + 35 * v2**2
        + 18 * v3**2
        - 12 * v1 * v2
        + 3 * v1 * v3
        - 18 * v2 * v3
        + 10 * v1
        - 20 * v2
        - 26 * v3
        + 151
    )


def solve_unconstrained(H: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(e) Solve Hv + c = 0 => Hv = -c."""
    return np.linalg.solve(H, -c)


def solve_kkt(H: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Solve KKT:
        [H  A^T][v]   [-c]
        [A   0 ][λ] = [ b]
    Returns (v, lambda).
    """
    K = np.block([[H, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(K, rhs)
    v = sol[: H.shape[0]]
    lam = sol[H.shape[0] :]
    return v, lam


def main():
    # ------------------------
    # Hard-encoded problem data
    # ------------------------
    H = np.array(
        [
            [24.0, -12.0, 3.0],
            [-12.0, 70.0, -18.0],
            [3.0, -18.0, 36.0],
        ]
    )
    c = np.array([10.0, -20.0, -26.0])

    A = np.array(
        [
            [-6.0, 4.0, -2.0],
            [7.0, -5.0, 3.0],
        ]
    )
    b = np.array([-8.0, 9.0])

    dg1 = 0.1
    dg2 = 0.1

    # ------------------------
    # (e) Unconstrained
    # ------------------------
    v_e = solve_unconstrained(H, c)
    f_e = f(v_e)

    print("(e) Unconstrained solution")
    print("    v* =", v_e)
    print("    f(v*) =", f_e)

    # ------------------------
    # (h) Binding constraints KKT
    # ------------------------
    v_h, lam_h = solve_kkt(H, c, A, b)
    f_h = f(v_h)
    resid_h = A @ v_h - b

    print("\n(h) KKT solution (both constraints binding)")
    print("    v* =", v_h)
    print("    lambda* =", lam_h)
    print("    Av - b =", resid_h)
    print("    f(v*) =", f_h)
    print("    lambdas positive? =", (lam_h > 0).tolist())

    # ------------------------
    # (i) Increase g1 by dg1 => use b - [dg1, 0]^T
    # ------------------------
    b_i = b - np.array([dg1, 0.0])
    v_i, lam_i = solve_kkt(H, c, A, b_i)
    f_i = f(v_i)

    df_dg1 = (f_i - f_h) / dg1
    avg_lam1 = 0.5 * (lam_h[0] + lam_i[0])

    print("\n(i) Modify b -> b - [dg1, 0]^T with dg1 = 0.1")
    print("    v* =", v_i)
    print("    lambda* =", lam_i)
    print("    Av - b_i =", A @ v_i - b_i)
    print("    f(v*) =", f_i)
    print("    Δf/Δg1 =", df_dg1)
    print("    avg(lambda1) =", avg_lam1)

    # ------------------------
    # (j) Increase g2 by dg2 => use b - [0, dg2]^T
    # ------------------------
    b_j = b - np.array([0.0, dg2])
    v_j, lam_j = solve_kkt(H, c, A, b_j)
    f_j = f(v_j)

    df_dg2 = (f_j - f_h) / dg2
    avg_lam2 = 0.5 * (lam_h[1] + lam_j[1])

    print("\n(j) Modify b -> b - [0, dg2]^T with dg2 = 0.1")
    print("    v* =", v_j)
    print("    lambda* =", lam_j)
    print("    Av - b_j =", A @ v_j - b_j)
    print("    f(v*) =", f_j)
    print("    Δf/Δg2 =", df_dg2)
    print("    avg(lambda2) =", avg_lam2)


if __name__ == "__main__":
    main()