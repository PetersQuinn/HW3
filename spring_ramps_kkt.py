# spring_ramps_kkt.py
# Purpose: Solve KKT equilibrium for two disks on ramps connected by a zero-free-length spring.
# Name: Quinn Peters

import numpy as np


def main():
    # Given numerical values
    m1 = 3.0
    m2 = 2.0
    a1 = -3.0
    a2 = 4.0
    k = 20.0
    g = 9.81

    # q = [x1, y1, x2, y2]^T
    # Π = 1/2 k[(x1-x2)^2 + (y1-y2)^2] + m1 g y1 + m2 g y2
    # => Π = 1/2 q^T H q + c^T q (up to an additive constant)
    H = k * np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    c = np.array([0.0, m1 * g, 0.0, m2 * g], dtype=float)

    # Constraints: y1 >= a1 x1 and y2 >= a2 x2
    # Write as Aq - b <= 0 with a1 x1 - y1 <= 0 and a2 x2 - y2 <= 0
    A = np.array(
        [
            [a1, -1.0, 0.0, 0.0],
            [0.0, 0.0, a2, -1.0],
        ],
        dtype=float,
    )
    b = np.array([0.0, 0.0], dtype=float)

    # KKT system: [H  A^T; A  0] [q; λ] = [-c; b]
    KKT = np.block([[H, A.T], [A, np.zeros((2, 2))]])
    rhs = np.concatenate([-c, b])

    sol = np.linalg.solve(KKT, rhs)
    q_star = sol[:4]
    lam_star = sol[4:]

    x1, y1, x2, y2 = q_star

    # Spring stretch and force magnitude
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    F = k * d

    # Constraint residual (binding => ~0)
    resid = A @ q_star - b

    np.set_printoptions(precision=8, suppress=True)
    print("(e) q* = [x1, y1, x2, y2]^T =", q_star)
    print("(e) lambda* =", lam_star)
    print("    Aq - b =", resid)

    print("\n(f) spring stretch d =", d)
    print("(f) spring force magnitude F = k d =", F)

    print("\nRank(H)   =", np.linalg.matrix_rank(H))
    print("Rank(KKT) =", np.linalg.matrix_rank(KKT))


if __name__ == "__main__":
    main()