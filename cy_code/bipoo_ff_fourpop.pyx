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
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport cython
from libcpp.vector cimport vector
from scipy.special import erfc

"""
Test the supervised training of patterns into the 
cue-Poisson->Stdp-ff->brunel<-supervision-Poisson
network.

"""

def train(int seed, 
    double vr, double vt, double muE, double muI, double DE, double DI,
    int NE, int NI, int CE, int CI, double J, double g, double h,
    double m0, int M, double nu0, double nu1, double nus, double fc, double fs, double Js,
    double Dc, double rac, double tauc, double tauac, 
    double T, double dt, double Twarm, double V0, int only_endpoint):
    """
    Structure:
    ----------
    Just test population-mean and -variance dynamics    
    """
    rng = np.random.default_rng(seed)

    # Draw random connectivity -> generate mvpost and numpost
    cdef int N = NE+NI
    Mat = np.zeros((N, N), dtype=np.int32)
    cdef int i = 0
    for i in range(N):
        Mat[i, rng.choice(np.arange(NE), size=CE, replace=False)] = 1
        Mat[i, rng.choice(np.arange(NE, N), size=CI, replace=False)] = 1
    cdef int [:, :] mvMat = Mat 
    cdef int numpost = np.max(np.sum(Mat, axis=0))
    cdef int [:, :] mvpost = N * np.ones((N, numpost), dtype=np.int32)
    cdef int [:] counter = np.zeros(N, dtype=np.int32)
    cdef int n = 0
    cdef int m = 0
    for n in range(N):
        for m in range(N):
            if mvMat[n, m] == 1:
                mvpost[m, counter[m]] = n
                counter[m] += 1

    del counter, Mat, mvMat

    # Sample training patterns
    cdef int Mc = int(np.round(fc * M))
    cdef int Ms = int(np.round(fs * NE))
    indsc = np.arange(M)
    indsE = np.arange(NE)
    cdef int [:] q = np.array(
        rng.choice(indsc, size=Mc, replace=False), dtype=np.int32)
    cdef int [:] p = np.array(
        rng.choice(indsE, size=Ms, replace=False), dtype=np.int32)

    cdef int [:] nonq = np.array(np.setdiff1d(np.arange(M), q), dtype=np.int32)
    cdef int [:] nonp = np.array(np.setdiff1d(np.arange(NE), p), dtype=np.int32)

    # Sample initial feedforward matrix
    cdef double [:, :] w = rng.normal(m0, np.sqrt(V0), size=(NE, M))

    cdef int steps = int(np.round(T/dt))
    cdef int warmsteps = int(np.round(Twarm/dt))
    cdef double s2DEdt = np.sqrt(2*DE*dt)
    cdef double s2DIdt = np.sqrt(2*DI*dt)

    # Thermalize membrane voltages
    cdef double [:] v = np.zeros(NE+NI+1)

    cdef int num1 = rng.poisson(nu1*Mc*Twarm)
    cdef int [:] timesteps1 = np.array(np.concatenate((
        np.sort(rng.integers(0, warmsteps, num1)), [-1])), dtype=np.int32)
    cdef int [:] indices1 = np.array(rng.integers(0, Mc, num1), 
        dtype=np.int32)
    cdef int position1 = 0

    cdef int num0 = rng.poisson(nu0*(M-Mc)*Twarm)
    cdef int [:] timesteps0 = np.array(np.concatenate((
        np.sort(rng.integers(0, warmsteps, num0)), [-1])), dtype=np.int32)
    cdef int [:] indices0 = np.array(rng.integers(0, (M-Mc), num0), 
        dtype=np.int32)
    cdef int position0 = 0

    cdef int nums = rng.poisson(nus*Ms*Twarm)
    cdef int [:] timestepss = np.array(np.concatenate((
        np.sort(rng.integers(0, warmsteps, nums)), [-1])), dtype=np.int32)
    cdef int [:] indicess = np.array(rng.integers(0, Ms, nums), dtype=np.int32)
    cdef int positions = 0

    cdef double [:] z = np.zeros(NE+NI)
    cdef double [:] Apre = np.zeros(M)
    cdef double [:] Apost = np.zeros(NE)
    cdef int k = 0
    cdef int step = 0
    for step in range(warmsteps):
        if step%1000 == 0: print(f'{step}/{warmsteps}')

        if DE > 0 or DI > 0:
            z = rng.normal(0, 1, NE+NI)
        for n in range(NE):
            v[n] += -dt*v[n] + dt*muE + s2DEdt*z[n]
            Apost[n] += -dt*Apost[n]/tauac

        for n in range(NE, NE+NI):
            v[n] += -dt*v[n] + dt*muI + s2DIdt*z[n]

        for m in range(M):
            Apre[m] += -dt*Apre[m]/tauc

        while timesteps1[position1] == step:
            for n in range(NE): 
                v[n] += w[n, q[indices1[position1]]]
            Apre[q[indices1[position1]]] += 1  # pre-synaptic tracer
            position1 += 1

        while timesteps0[position0] == step:
            for n in range(NE): 
                v[n] += w[n, nonq[indices0[position0]]]
            Apre[nonq[indices0[position0]]] += 1  # pre-synaptic tracer
            position0 += 1

        while timestepss[positions] == step:
            v[p[indicess[positions]]] += Js
            positions += 1

        for n in range(NE+NI):
            if v[n] >= vt:
                v[n] = vr
                if n < NE:
                    Apost[n] += 1  # post-synaptic tracer
                    for k in range(numpost):
                        if mvpost[n, k] >= NE: v[mvpost[n, k]] += h*J
                        else: v[mvpost[n, k]] += J
                else:
                    for k in range(numpost):
                        v[mvpost[n, k]] -= g*J

        v[N] = 0

    # Train
    num1 = rng.poisson(nu1*Mc*T)
    timesteps1 = np.array(np.concatenate((
        np.sort(rng.integers(0, steps, num1)), [-1])), dtype=np.int32)
    indices1 = np.array(rng.integers(0, Mc, num1), 
        dtype=np.int32)
    position1 = 0

    num0 = rng.poisson(nu0*(M-Mc)*T)
    timesteps0 = np.array(np.concatenate((
        np.sort(rng.integers(0, steps, num0)), [-1])), dtype=np.int32)
    indices0 = np.array(rng.integers(0, M-Mc, num0), 
        dtype=np.int32)
    position0 = 0

    nums = rng.poisson(nus*Ms*T)
    timestepss = np.array(np.concatenate((
        np.sort(rng.integers(0, steps, nums)), [-1])), dtype=np.int32)
    indicess = np.array(rng.integers(0, Ms, nums), dtype=np.int32)
    positions = 0

    cdef int count_nontarget = 0
    cdef int count_target = 0
    cdef int count_inhib = 0

    cdef double [:] m00_t = np.zeros(steps//100)
    cdef double [:] m01_t = np.zeros(steps//100)
    cdef double [:] m10_t = np.zeros(steps//100)
    cdef double [:] m11_t = np.zeros(steps//100)
    cdef double [:] V00_t = np.zeros(steps//100)
    cdef double [:] V01_t = np.zeros(steps//100)
    cdef double [:] V10_t = np.zeros(steps//100)
    cdef double [:] V11_t = np.zeros(steps//100)

    for step in range(steps):
        if step%1000 == 0: print(f'{step}/{steps}')
        if DE > 0 or DI > 0:
            z = rng.normal(0, 1, NE+NI)
        for n in range(NE):
            v[n] += -dt*v[n] + dt*muE + s2DEdt*z[n]
            Apost[n] += -dt*Apost[n]/tauac

        for n in range(NE, NE+NI):
            v[n] += -dt*v[n] + dt*muI + s2DIdt*z[n]

        for m in range(M):
            Apre[m] += -dt*Apre[m]/tauc

        while timesteps1[position1] == step:
            for n in range(NE): 
                v[n] += w[n, q[indices1[position1]]]
                w[n, q[indices1[position1]]] = \
                    np.clip(w[n, q[indices1[position1]]]*(1+rac*Apost[n]), 0, 1)
            Apre[q[indices1[position1]]] += 1  # pre-synaptic tracer
            position1 += 1

        while timesteps0[position0] == step:
            for n in range(NE): 
                v[n] += w[n, nonq[indices0[position0]]]
                w[n, nonq[indices0[position0]]] = \
                    np.clip(w[n, nonq[indices0[position0]]]*(1+rac*Apost[n]), 0, 1)
            Apre[nonq[indices0[position0]]] += 1  # pre-synaptic tracer
            position0 += 1

        while timestepss[positions] == step:
            v[p[indicess[positions]]] += Js
            positions += 1

        for n in range(NE+NI):
            if v[n] >= vt:
                v[n] = vr
                if n < NE:
                    if n in p: count_target += 1
                    else: count_nontarget += 1
                    Apost[n] += 1  # post-synaptic tracer
                    for m in range(M):
                        w[n, m] += Dc * Apre[m]  # potentiation
                    for k in range(numpost):
                        if mvpost[n, k] >= NE: v[mvpost[n, k]] += h*J
                        else: v[mvpost[n, k]] += J
                else:
                    count_inhib += 1
                    for k in range(numpost):
                        v[mvpost[n, k]] -= g*J
        
        v[N] = 0
        if step % 100 == 0 and not only_endpoint:
            m00_t[step//100] = np.mean([[w[nonp[n], nonq[m]] for n in range(NE-Ms)] for m in range(M-Mc)])
            m01_t[step//100] = np.mean([[w[nonp[n], q[m]] for n in range(NE-Ms)] for m in range(Mc)])
            m10_t[step//100] = np.mean([[w[p[n], nonq[m]] for n in range(Ms)] for m in range(M-Mc)])
            m11_t[step//100] = np.mean([[w[p[n], q[m]] for n in range(Ms)] for m in range(Mc)])
            V00_t[step//100] = np.std([[w[nonp[n], nonq[m]] for n in range(NE-Ms)] for m in range(M-Mc)])**2
            V01_t[step//100] = np.std([[w[nonp[n], q[m]] for n in range(NE-Ms)] for m in range(Mc)])**2
            V10_t[step//100] = np.std([[w[p[n], nonq[m]] for n in range(Ms)] for m in range(M-Mc)])**2
            V11_t[step//100] = np.std([[w[p[n], q[m]] for n in range(Ms)] for m in range(Mc)])**2
        if only_endpoint and step == steps-1:
            m00_ = np.mean([[w[nonp[n], nonq[m]] for n in range(NE-Ms)] for m in range(M-Mc)])
            m01_ = np.mean([[w[nonp[n], q[m]] for n in range(NE-Ms)] for m in range(Mc)])
            m10_ = np.mean([[w[p[n], nonq[m]] for n in range(Ms)] for m in range(M-Mc)])
            m11_ = np.mean([[w[p[n], q[m]] for n in range(Ms)] for m in range(Mc)])
            V00_ = np.std([[w[nonp[n], nonq[m]] for n in range(NE-Ms)] for m in range(M-Mc)])**2
            V01_ = np.std([[w[nonp[n], q[m]] for n in range(NE-Ms)] for m in range(Mc)])**2
            V10_ = np.std([[w[p[n], nonq[m]] for n in range(Ms)] for m in range(M-Mc)])**2
            V11_ = np.std([[w[p[n], q[m]] for n in range(Ms)] for m in range(Mc)])**2
            return m00_, m01_, m10_, m11_, V00_, V01_, V10_, V11_

    return np.array(m00_t), np.array(m01_t), np.array(m10_t), np.array(m11_t), np.array(V00_t), np.array(V01_t), np.array(V10_t), np.array(V11_t)
