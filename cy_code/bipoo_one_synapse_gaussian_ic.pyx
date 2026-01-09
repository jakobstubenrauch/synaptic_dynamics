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
from utilities.tools import check, load, save, makename
import multiprocessing as mp
from functools import partial
    

def integrate(int seed, float nuc, float m0, float V0, float mu, float D, 
    float vt, float vr, float Dc, float rac, float tauc, float tauac, 
    float dt, float T, float Twarm, int boundary=0):
    """
    Integrate a Bi Poo 1998 synapse that delivers Poissonian spikes
    to a leaky integrate-and-fire neuron, with initial synaptic weight
    drawn from a Gaussian distribution with mean m0 and variance V0.
    """

    rng = np.random.default_rng(seed)
    cdef int steps = int(np.round(T/dt))
    cdef int warmsteps = int(np.round(Twarm/dt))
    cdef double s2Ddt = np.sqrt(2*D*dt)
    cdef double w = rng.normal(m0, np.sqrt(V0))

    # Thermalize membrane voltage
    cdef double v = 0
    cdef int [:] etawarm = np.array(
        rng.poisson(nuc*dt, size=warmsteps), dtype=np.int32)
    cdef double [:] zwarm = rng.normal(0, 1, size=warmsteps)
    cdef double Apre = 0
    cdef double Apost = 0
    cdef int step = 0
    for step in range(warmsteps):
        Apre += -dt*Apre/tauc + etawarm[step] # pre-synaptic tracer
        Apost += -dt*Apost/tauac
        v += -dt*v + dt*mu + s2Ddt*zwarm[step] + w*etawarm[step]
        if v >= vt: 
            v = vr
            Apost += 1  # post-synaptic tracer
    del etawarm, zwarm

    # Train
    cdef int [:] etac = np.array(
        rng.poisson(nuc*dt, size=steps), dtype=np.int32)
    cdef double [:] z = rng.normal(0, 1, size=steps)
    cdef double [:] w_store = np.zeros(steps)
    cdef int count = 0

    for step in range(steps):
        v += -dt*v + dt*mu + s2Ddt*z[step] + w*etac[step]
        Apre += -dt*Apre/tauc + etac[step]  # pre-synaptic tracer
        Apost += -dt*Apost/tauac
        w += w*rac*Apost*etac[step]  # depression
        if v >= vt:
            count += 1
            v = vr
            Apost += 1  # post-synaptic tracer
            w += Dc * Apre  # potentiation

        if boundary == 1:
            if w < 0:
                w = 0
        w_store[step] = w

    return np.asarray(w_store), count / T


def integrate_uniform_ic(int seed, float nuc, float wmin, float wmax, float mu, float D, 
    float vt, float vr, float Dc, float rac, float tauc, float tauac, 
    double dt, float T, float Twarm, int boundary=0):
    """
    Integrate a Bi Poo 1998 synapse that delivers Poissonian spikes
    to a leaky integrate-and-fire neuron, with initial synaptic weight
    drawn from a uniform distribution with mean m0 and variance V0.
    """

    rng = np.random.default_rng(seed)
    cdef int steps = int(np.round(T/dt))
    cdef int warmsteps = int(np.round(Twarm/dt))
    cdef double s2Ddt = np.sqrt(2*D*dt)

    # Uniform for broad coverage
    cdef double w = rng.uniform(wmin, wmax)

    # Thermalize membrane voltage and tracers
    cdef double v = 0
    cdef int [:] etawarm = np.array(
        rng.poisson(nuc*dt, size=warmsteps), dtype=np.int32)
    cdef double [:] zwarm = rng.normal(0, 1, size=warmsteps)
    cdef double Apre = 0
    cdef double Apost = 0
    cdef int step = 0
    for step in range(warmsteps):
        Apre += -dt*Apre/tauc + etawarm[step] # pre-synaptic tracer
        Apost += -dt*Apost/tauac
        v += -dt*v + dt*mu + s2Ddt*zwarm[step] + w*etawarm[step]
        if v >= vt: 
            v = vr
            Apost += 1  # post-synaptic tracer
    del etawarm, zwarm

    # Train
    cdef int [:] etac = np.array(
        rng.poisson(nuc*dt, size=steps), dtype=np.int32)
    cdef double [:] z = rng.normal(0, 1, size=steps)
    cdef double [:] w_store = np.zeros(steps)
    cdef int count = 0

    for step in range(steps):
        v += -dt*v + dt*mu + s2Ddt*z[step] + w*etac[step]
        Apre += -dt*Apre/tauc + etac[step]  # pre-synaptic tracer
        Apost += -dt*Apost/tauac
        w += w*rac*Apost*etac[step]  # depression
        if v >= vt:
            count += 1
            v = vr
            Apost += 1  # post-synaptic tracer
            w += Dc * Apre  # potentiation

        if boundary == 1:
            if w < 0:
                w = 0
        w_store[step] = w

    return np.asarray(w_store), count / T
