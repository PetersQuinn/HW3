# hw3_dimensionality_simple.py
# Purpose: Compare ORS, NMS, SQP on f(v)=sum_{k=1..n} (1/k)(v_k-k)^2 for n=2,10,50
# Name: Quinn Peters

import sys
import time
import numpy as np
import contextlib
import io

MV_ROOT = r"C:\Users\quint\OneDrive\Desktop\Python\CEE\CEE251\multivarious-main\multivarious-main"
if MV_ROOT not in sys.path:
    sys.path.insert(0, MV_ROOT)

from multivarious.opt.ors import ors
from multivarious.opt.nms import nms
from multivarious.opt.sqp import sqp

print("imports ok")

np.set_printoptions(precision=6, suppress=True)

N_LIST = [2, 10, 50]
BASE_EVALS = 2000
EVALS_PER_N = 200


def run_quiet(callable_fn, *args, **kwargs):
    """Run a function while suppressing any prints it produces."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return callable_fn(*args, **kwargs)


for n in N_LIST:

    def func(v, consts=1.0):
        v = np.asarray(v, dtype=float).ravel()
        k = np.arange(1, n + 1, dtype=float)
        f = float(np.sum(((v - k) ** 2) / k))
        g = np.array([-1.0], dtype=float)  
        return f, g

    v_lb = (-2.0 * n) * np.ones(n)
    v_ub = (2.0 * n) * np.ones(n)
    v_init = np.zeros(n)

    max_evals = int(BASE_EVALS + EVALS_PER_N * n)
    options = np.array(
        [0, 1e-8, 1e-10, 1e-9, max_evals, 1.0, 2.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1e-8, 1e-2, 0],
        dtype=float,
    )

    v_true = np.arange(1, n + 1, dtype=float)
    f_true, _ = func(v_true)

    print("\n" + "=" * 90)
    print(f"n = {n}   (bounds: [{-2*n}, {2*n}] each)   max_evals = {max_evals}")
    print(f"True minimizer: f(v*) = {f_true:.3e}")

    # ---------------- ORS (quiet) ----------------
    t0 = time.time()
    v_opt, f_opt, g_opt, cvg_hst, fevals, iters = run_quiet(
        ors, func, v_init, v_lb, v_ub, options, 1.0
    )
    dt = time.time() - t0
    rms = np.linalg.norm(v_opt - v_true) / np.sqrt(n)

    print("\nORS (final):")
    print(f"  f_opt          = {f_opt:.3e}")
    print(f"  RMS ||v-v*||    = {rms:.3e}")
    print(f"  evals, iters    = {fevals}, {iters}")
    print(f"  time (s)        = {dt:.3f}")

    # ---------------- NMS (quiet) ----------------
    t0 = time.time()
    v_opt, f_opt, g_opt, cvg_hst, fevals, iters = run_quiet(
        nms, func, v_init, v_lb, v_ub, options, 1.0
    )
    dt = time.time() - t0
    rms = np.linalg.norm(v_opt - v_true) / np.sqrt(n)

    print("\nNMS (final):")
    print(f"  f_opt          = {f_opt:.3e}")
    print(f"  RMS ||v-v*||    = {rms:.3e}")
    print(f"  evals, iters    = {fevals}, {iters}")
    print(f"  time (s)        = {dt:.3f}")

    # ---------------- SQP (quiet) ----------------
    options_sqp = options.copy()
    options_sqp[0] = 1

    t0 = time.time()
    v_opt, f_opt, g_opt, cvg_hst, lambda_qp, HESS = run_quiet(
        sqp, func, v_init, v_lb, v_ub, options_sqp, 1.0
    )
    dt = time.time() - t0
    rms = np.linalg.norm(v_opt - v_true) / np.sqrt(n)

    # eval count is stored in cvg_hst row (-3), last column (func_count)
    fevals_sqp = int(cvg_hst[-3, -1]) if (cvg_hst is not None and np.size(cvg_hst) > 0) else None

    print("\nSQP (final):")
    print(f"  f_opt          = {f_opt:.3e}")
    print(f"  RMS ||v-v*||    = {rms:.3e}")
    print(f"  evals           = {fevals_sqp}")
    print(f"  time (s)        = {dt:.3f}")

print("\nDone.")