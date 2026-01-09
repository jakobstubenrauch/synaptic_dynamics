#    Copyright 2025 Jakob Stubenrauch, Naomi Auer, Richard Kempter, and
#    Benjamin Lindner
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the Affero GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the Affero GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# cython: language_level=3
# distutils: language=c++
# cython: infer_types=True
import numpy as np


cdef inline double interp1d_cython(double x, double[:] xp, double[:] fp, int n):
    """
    Linear interpolation. The function f with function values fp at coordinates
    xp is evaluated at x.
    """
    cdef int i
    if x <= xp[0]:
        return fp[0]
    elif x >= xp[n-1]:
        return fp[n-1]
    for i in range(n-1):
        if xp[i] <= x < xp[i+1]:
            return fp[i] + (fp[i+1] - fp[i]) * (x - xp[i]) / (xp[i+1] - xp[i])
    return fp[n-1]  # fallback


def integrate(dt, T, m0, V0, w_arr, D1_arr, D2_arr, seed):
    """
    Integrate a Langevin equation with drift and diffusion given through 
    datapoints between which the routine integrates.

    Assume Ito interpretation of the Langevin equation.

    Parameters
    ----------
    dt : float
        Time step.
    T : float
        Total integration time.
    m0 : float
        Initial position.
    V0 : float
        Initial variance.
    w_arr : array-like
        Weight points for interpolation.
    D1_arr : array-like
        Drift coefficient at w_arr.
    D2_arr : array-like
        Diffusion coefficient at w_arr.
    """
    rng = np.random.default_rng(seed)
    cdef int steps = int(np.round(T/dt))
    cdef double w = rng.normal(m0, np.sqrt(V0))
    cdef double [:] w_store = np.zeros(steps)

    # Convert input arrays to contiguous memoryviews for fast access
    cdef double[:] w_arr_mv = np.ascontiguousarray(w_arr, dtype=np.float64)
    cdef double[:] D1_arr_mv = np.ascontiguousarray(D1_arr, dtype=np.float64)
    cdef double[:] D2_arr_mv = np.ascontiguousarray(D2_arr, dtype=np.float64)

    cdef int n_points = w_arr_mv.shape[0]
    cdef double D1 = 0
    cdef double D2 = 0

    cdef double [:] z = rng.normal(0, 1, steps)
    cdef int step = 0
    for step in range(steps):
        D1 = interp1d_cython(w, w_arr_mv, D1_arr_mv, n_points)
        D2 = interp1d_cython(w, w_arr_mv, D2_arr_mv, n_points)
        w += dt*D1 + np.sqrt(2*D2*dt)*z[step]
        w_store[step] = w

    return np.asarray(w_store)