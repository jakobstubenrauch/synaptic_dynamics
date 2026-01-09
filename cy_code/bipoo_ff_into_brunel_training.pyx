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
cimport cython
from libcpp.vector cimport vector
from scipy.special import erfc

"""
Test the supervised training of patterns into the 
cue-Poisson->Stdp-ff->brunel<-supervision-Poisson
network.

Warmup and reset tracers before training to avoid unforeseable
difficulties.
"""

def train(int seed, 
    float vr, float vt, float muE, float muI, float DE, float DI,
    int NE, int NI, int CE, int CI, float J, float g, float h,
    float m0, int M, float nu, float nus, float fc, float fs, float Js,
    float Dc, float rac, float tauc, float tauac, 
    float T, float dt, float Twarm, int k, int P, float V0,
    return_weights=False, return_confusion=False, homeostasis=True, float Trecall=0):
    """
    Structure:
    ----------

    1. Generate network and sample patterns
    2. In a loop from i=0...P do
        a) train pattern pair q[i], p[i]
        b) compute score of pattern k
    
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
    cdef int [:, :] qs = np.array(
        [rng.choice(indsc, size=Mc, replace=False) \
        for i in range(P+1)], dtype=np.int32)
    cdef int [:, :] ps = np.array(
        [rng.choice(indsE, size=Ms, replace=False) \
        for i in range(P+1)], dtype=np.int32)

    # copy observed pattern pair
    cdef int [:] qk = qs[k]
    cdef int [:] nonqk = np.array(np.setdiff1d(np.arange(M), qk), dtype=np.int32)
    cdef int [:] pk = ps[k]
    cdef int [:] nonpk = np.array(np.setdiff1d(np.arange(NE), pk), dtype=np.int32)
    cdef int [:] qi = qs[k]
    cdef int [:] nonqi = np.array(np.setdiff1d(np.arange(M), qi), dtype=np.int32)
    cdef int [:] pi = ps[k]
    cdef int [:] nonpi = np.array(np.setdiff1d(np.arange(NE), pi), dtype=np.int32)

    # Sample initial feedforward matrix
    cdef double [:, :] w = rng.normal(m0, np.sqrt(V0), size=(NE, M))

    MTs = []  # mean weight cue k->target k post-homeostasts
    MNs = []  # mean weight cue k->non-target k post-homeostasts
    GTs = []  # variance of weights cue k->target k post-homeostasts
    GNs = []  # variance of weights cue k->non-target k post-homeostasts
    DeltaTs = []  # mean drift of weights cue i->target i pre-homeostasis
    DeltaNs = []  # mean drift of weights cue i->non-target i pre-homeostasis
    VTs = []  # variance of weights cue i -> target i after STDP pre-homeostasis
    VNs = []  # variance of weights cue i -> non-target i after STDP pre-homeostasis
    V0s = []  # total variance post-homeostasis
    rTs, rNs, rIs = [], [], []  # firing rates during training of three pops

    if return_weights:
        wspre = []
        wsT = []
        wsN = []
        wsNC = []
        wspost = []
        wspost.append(np.copy(w))

    if return_confusion:
        afp = []

    for i in range(P):
        print(f'step {i}/{P}')

        # Train one step
        seed_ = rng.integers(99999999)
        w, rT, rN, rI = train_one_step(seed_, numpost, mvpost, w, qs[i], ps[i],
            fc, fs, Js, M, NE, NI, CE, CI, J, g, h, nu, nus, muE, DE, muI, DI,
            vt, vr, Dc, rac, tauc, tauac, dt, T, Twarm)
        
        # Store pre-homeostasis variables
        qi = qs[i]
        nonqi = np.array(np.setdiff1d(np.arange(M), qi), dtype=np.int32)
        pi = ps[i]
        nonpi = np.array(np.setdiff1d(np.arange(NE), pi), dtype=np.int32)
        DeltaNs.append(np.mean([[w[n, m] for n in nonpi] for m in qi]) - m0)
        DeltaTs.append(np.mean([[w[n, m] for n in pi] for m in qi]) - m0)
        VNs.append(np.std([[w[n, m] for n in nonpi] for m in qi])**2)
        VTs.append(np.std([[w[n, m] for n in pi] for m in qi])**2)
        rTs.append(rT)
        rNs.append(rN)
        rIs.append(rI)

        if return_weights:
            wspre.append(np.copy(w))
            wsT.append(np.array([[w[n, m] for n in pk] for m in qk]))
            wsN.append(np.array([[w[n, m] for n in nonpk] for m in qk]))
            wsNC.append(np.array([[w[n, m] for n in range(NE)] for m in nonqk]))
        # Homeostasis
        sums = np.sum(w, axis=-1)
        for n in range(NE):
            for m in range(M):
                w[n, m] *= m0 / (sums[n] / M)

        # Store post-homeostasis variables
        MNs.append(np.mean([[w[n, m] for n in nonpk] for m in qk]))
        MTs.append(np.mean([[w[n, m] for n in pk] for m in qk]))
        GNs.append(np.std([[w[n, m] for n in nonpk] for m in qk])**2)
        GTs.append(np.std([[w[n, m] for n in pk] for m in qk])**2)
        V0s.append(np.std(w)**2)

        if return_weights:
            wspost.append(np.copy(w))

        if return_confusion:
            if Trecall == np.inf:
                s = np.sum([[w[n, m] for m in qk] for n in range(NE)], axis=-1)
                assert len(s) == NE
                inds = np.argsort(s)[-int(np.round(fs*NE)):]
                afp.append( 1 - len(np.intersect1d(inds, pk))/len(inds) )
            else:
                seed_ = rng.integers(99999999)
                afp.append(recall(seed_, numpost, mvpost, w, qk, pk,
                    fc, fs, Js, M, NE, NI, CE, CI, J, g, h, nu, nus, muE, DE, muI, DI,
                    vt, vr, dt, Trecall, Twarm))

    if return_weights:
        return wspre, wspost, wsT, wsN, wsNC
    if return_confusion:
        return MTs, MNs, GTs, GNs, DeltaTs, DeltaNs, VTs, VNs, V0s, rTs, rNs, rIs, afp

    return MTs, MNs, GTs, GNs, DeltaTs, DeltaNs, VTs, VNs, V0s, rTs, rNs, rIs


def train_no_homeostasis(int seed, 
    float vr, float vt, float muE, float muI, float DE, float DI,
    int NE, int NI, int CE, int CI, float J, float g, float h,
    float m0, int M, float nu, float nus, float fc, float fs, float Js,
    float Dc, float rac, float tauc, float tauac, 
    float T, float dt, float Twarm, int k, int P, float V0,
    return_weights=False, return_confusion=False, float Trecall=0):
    """
    Structure:
    ----------

    1. Generate network and sample patterns
    2. In a loop from i=0...P do
        a) train pattern pair q[i], p[i]
        b) compute score of pattern k
    
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
    cdef int [:, :] qs = np.array(
        [rng.choice(indsc, size=Mc, replace=False) \
        for i in range(P+1)], dtype=np.int32)
    cdef int [:, :] ps = np.array(
        [rng.choice(indsE, size=Ms, replace=False) \
        for i in range(P+1)], dtype=np.int32)

    # copy observed pattern pair
    cdef int [:] qk = qs[k]
    cdef int [:] nonqk = np.array(np.setdiff1d(np.arange(M), qk), dtype=np.int32)
    cdef int [:] pk = ps[k]
    cdef int [:] nonpk = np.array(np.setdiff1d(np.arange(NE), pk), dtype=np.int32)
    cdef int [:] qi = qs[k]
    cdef int [:] nonqi = np.array(np.setdiff1d(np.arange(M), qi), dtype=np.int32)
    cdef int [:] pi = ps[k]
    cdef int [:] nonpi = np.array(np.setdiff1d(np.arange(NE), pi), dtype=np.int32)

    # Sample initial feedforward matrix
    cdef double [:, :] w = rng.normal(m0, np.sqrt(V0), size=(NE, M))

    MTs = []  # mean weight cue k->target k post-homeostasts
    MNs = []  # mean weight cue k->non-target k post-homeostasts
    GTs = []  # variance of weights cue k->target k post-homeostasts
    GNs = []  # variance of weights cue k->non-target k post-homeostasts
    DeltaTs = []  # mean drift of weights cue i->target i pre-homeostasis
    DeltaNs = []  # mean drift of weights cue i->non-target i pre-homeostasis
    VTs = []  # variance of weights cue i -> target i after STDP pre-homeostasis
    VNs = []  # variance of weights cue i -> non-target i after STDP pre-homeostasis
    V0s = []  # total variance post-homeostasis
    rTs, rNs, rIs = [], [], []  # firing rates during training of three pops
    Mall = []  # total mean weight
    Vall = []  # total weight variance


    if return_weights:
        wspre = []
        wsT = []
        wsN = []
        wsNC = []
        wspost = []
        wspost.append(np.copy(w))

    if return_confusion:
        afp = []

    for i in range(P):
        print(f'step {i}/{P}')

        # Train one step
        seed_ = rng.integers(99999999)
        w, rT, rN, rI = train_one_step(seed_, numpost, mvpost, w, qs[i], ps[i],
            fc, fs, Js, M, NE, NI, CE, CI, J, g, h, nu, nus, muE, DE, muI, DI,
            vt, vr, Dc, rac, tauc, tauac, dt, T, Twarm)
        
        # Store pre-homeostasis variables
        qi = qs[i]
        nonqi = np.array(np.setdiff1d(np.arange(M), qi), dtype=np.int32)
        pi = ps[i]
        nonpi = np.array(np.setdiff1d(np.arange(NE), pi), dtype=np.int32)
        # CAREFUL! THIS DOES NOT CAPTURE THE SHIFT, SINCE m0 is not the init weight anymore
        # HOWEVER: solve this issue outside this script. If I ever fix it here, I have
        # to check where I corrected it in post.
        DeltaNs.append(np.mean([[w[n, m] for n in nonpi] for m in qi]) - m0)
        DeltaTs.append(np.mean([[w[n, m] for n in pi] for m in qi]) - m0)
        VNs.append(np.std([[w[n, m] for n in nonpi] for m in qi])**2)
        VTs.append(np.std([[w[n, m] for n in pi] for m in qi])**2)
        rTs.append(rT)
        rNs.append(rN)
        rIs.append(rI)

        if return_weights:
            wspre.append(np.copy(w))
            wsT.append(np.array([[w[n, m] for n in pk] for m in qk]))
            wsN.append(np.array([[w[n, m] for n in nonpk] for m in qk]))
            wsNC.append(np.array([[w[n, m] for n in range(NE)] for m in nonqk]))

        # IN THIS VARIANT NO HOMEOSTASIS IS APPLIED!!
        # # Homeostasis
        # sums = np.sum(w, axis=-1)
        # for n in range(NE):
        #     for m in range(M):
        #         w[n, m] *= m0 / (sums[n] / M)

        # Store post-homeostasis variables
        MNs.append(np.mean([[w[n, m] for n in nonpk] for m in qk]))
        MTs.append(np.mean([[w[n, m] for n in pk] for m in qk]))
        GNs.append(np.std([[w[n, m] for n in nonpk] for m in qk])**2)
        GTs.append(np.std([[w[n, m] for n in pk] for m in qk])**2)
        V0s.append(np.std(w)**2)
        Mall.append(np.mean(w))
        Vall.append(np.var(w))

        if return_weights:
            wspost.append(np.copy(w))

        if return_confusion:
            if Trecall == np.inf:
                s = np.sum([[w[n, m] for m in qk] for n in range(NE)], axis=-1)
                assert len(s) == NE
                inds = np.argsort(s)[-int(np.round(fs*NE)):]
                afp.append( 1 - len(np.intersect1d(inds, pk))/len(inds) )
            else:
                seed_ = rng.integers(99999999)
                afp.append(recall(seed_, numpost, mvpost, w, qk, pk,
                    fc, fs, Js, M, NE, NI, CE, CI, J, g, h, nu, nus, muE, DE, muI, DI,
                    vt, vr, dt, Trecall, Twarm))

    if return_weights:
        return wspre, wspost, wsT, wsN, wsNC
    if return_confusion:
        return Mall, Vall, MTs, MNs, GTs, GNs, DeltaTs, DeltaNs, VTs, VNs, V0s, rTs, rNs, rIs, afp

    return Mall, Vall, MTs, MNs, GTs, GNs, DeltaTs, DeltaNs, VTs, VNs, V0s, rTs, rNs, rIs


def recall(int seed,
    int numpost, int [:, :] mvpost, double [:, :] w, int [:] q, int [:] p, float fc, float fs, float Js,
    int M, int NE, int NI, int CE, int CI, float J, float g, float h, 
    float nu, float nus, float muE, float DE, float muI, float DI, 
    float vt, float vr, float dt, float T, float Twarm):

    cdef int n = 0
    cdef int m = 0

    rng = np.random.default_rng(seed)

    cdef int Mc = int(np.round(M*fc))
    cdef int Ms = int(np.round(NE*fs))
    cdef int N = NE+NI

    cdef int steps = int(np.round(T/dt))
    cdef int warmsteps = int(np.round(Twarm/dt))
    cdef double s2DEdt = np.sqrt(2*DE*dt)
    cdef double s2DIdt = np.sqrt(2*DI*dt)

    # Thermalize membrane voltages
    cdef double [:] v = np.zeros(NE+NI+1)

    cdef int numc = rng.poisson(nu*Mc*Twarm)
    cdef int [:] timestepsc = np.array(np.concatenate((
        np.sort(rng.integers(0, warmsteps, numc)), [-1])), dtype=np.int32)
    cdef int [:] indicesc = np.array(rng.integers(0, Mc, numc), 
        dtype=np.int32)
    cdef int positionc = 0

    cdef double [:] z = np.zeros(NE+NI)
    cdef int k = 0
    cdef int step = 0
    for step in range(warmsteps):

        if DE > 0 or DI > 0:
            z = rng.normal(0, 1, NE+NI)
        for n in range(NE):
            v[n] += -dt*v[n] + dt*muE + s2DEdt*z[n]

        for n in range(NE, NE+NI):
            v[n] += -dt*v[n] + dt*muI + s2DIdt*z[n]

        while timestepsc[positionc] == step:
            for n in range(NE): 
                v[n] += w[n, q[indicesc[positionc]]]
            positionc += 1

        for n in range(NE+NI):
            if v[n] >= vt:
                v[n] = vr
                if n < NE:
                    for k in range(numpost):
                        if mvpost[n, k] >= NE: v[mvpost[n, k]] += h*J
                        else: v[mvpost[n, k]] += J
                else:
                    for k in range(numpost):
                        v[mvpost[n, k]] -= g*J

        v[N] = 0

    # Train
    numc = rng.poisson(nu*Mc*T)
    timestepsc = np.array(np.concatenate((
        np.sort(rng.integers(0, steps, numc)), [-1])), dtype=np.int32)
    indicesc = np.array(rng.integers(0, Mc, numc), 
        dtype=np.int32)
    positionc = 0

    cdef int [:] counts = np.zeros(NE, dtype=np.int32)

    for step in range(steps):
        if DE > 0 or DI > 0:
            z = rng.normal(0, 1, NE+NI)
        for n in range(NE):
            v[n] += -dt*v[n] + dt*muE + s2DEdt*z[n]

        for n in range(NE, NE+NI):
            v[n] += -dt*v[n] + dt*muI + s2DIdt*z[n]

        while timestepsc[positionc] == step:
            for n in range(NE): 
                v[n] += w[n, q[indicesc[positionc]]]
            positionc += 1

        for n in range(NE+NI):
            if v[n] >= vt:
                v[n] = vr
                if n < NE:
                    counts[n] += 1
                    for k in range(numpost):
                        if mvpost[n, k] >= NE: v[mvpost[n, k]] += h*J
                        else: v[mvpost[n, k]] += J
                else:
                    for k in range(numpost):
                        v[mvpost[n, k]] -= g*J
        
        v[N] = 0

    # obtain indices of fs NE most active neurons
    inds = np.argsort(counts)[-int(np.round(fs*NE)):]
    afp = 1 - len(np.intersect1d(inds, p))/len(inds)

    return afp


def train_one_step(int seed,
    int numpost, int [:, :] mvpost, double [:, :] w, int [:] q, int [:] p, float fc, float fs, float Js,
    int M, int NE, int NI, int CE, int CI, float J, float g, float h, 
    float nu, float nus, float muE, float DE, float muI, float DI, 
    float vt, float vr, float Dc, float rac, float tauc, float tauac, 
    float dt, float T, float Twarm):

    cdef int n = 0
    cdef int m = 0

    rng = np.random.default_rng(seed)

    cdef int Mc = int(np.round(M*fc))
    cdef int Ms = int(np.round(NE*fs))
    cdef int N = NE+NI

    cdef int steps = int(np.round(T/dt))
    cdef int warmsteps = int(np.round(Twarm/dt))
    cdef double s2DEdt = np.sqrt(2*DE*dt)
    cdef double s2DIdt = np.sqrt(2*DI*dt)

    # Thermalize membrane voltages
    cdef double [:] v = np.zeros(NE+NI+1)

    cdef int numc = rng.poisson(nu*Mc*Twarm)
    cdef int [:] timestepsc = np.array(np.concatenate((
        np.sort(rng.integers(0, warmsteps, numc)), [-1])), dtype=np.int32)
    cdef int [:] indicesc = np.array(rng.integers(0, Mc, numc), 
        dtype=np.int32)
    cdef int positionc = 0
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

        if DE > 0 or DI > 0:
            z = rng.normal(0, 1, NE+NI)
        for n in range(NE):
            v[n] += -dt*v[n] + dt*muE + s2DEdt*z[n]
            Apost[n] += -dt*Apost[n]/tauac

        for n in range(NE, NE+NI):
            v[n] += -dt*v[n] + dt*muI + s2DIdt*z[n]

        for m in range(M):
            Apre[m] += -dt*Apre[m]/tauc

        while timestepsc[positionc] == step:
            for n in range(NE): 
                v[n] += w[n, q[indicesc[positionc]]]
            Apre[q[indicesc[positionc]]] += 1  # pre-synaptic tracer
            positionc += 1

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
    numc = rng.poisson(nu*Mc*T)
    timestepsc = np.array(np.concatenate((
        np.sort(rng.integers(0, steps, numc)), [-1])), dtype=np.int32)
    indicesc = np.array(rng.integers(0, Mc, numc), 
        dtype=np.int32)
    positionc = 0

    nums = rng.poisson(nus*Ms*T)
    timestepss = np.array(np.concatenate((
        np.sort(rng.integers(0, steps, nums)), [-1])), dtype=np.int32)
    indicess = np.array(rng.integers(0, Ms, nums), dtype=np.int32)
    positions = 0

    cdef int count_nontarget = 0
    cdef int count_target = 0
    cdef int count_inhib = 0

    for step in range(steps):
        if DE > 0 or DI > 0:
            z = rng.normal(0, 1, NE+NI)
        for n in range(NE):
            v[n] += -dt*v[n] + dt*muE + s2DEdt*z[n]
            Apost[n] += -dt*Apost[n]/tauac

        for n in range(NE, NE+NI):
            v[n] += -dt*v[n] + dt*muI + s2DIdt*z[n]

        for m in range(M):
            Apre[m] += -dt*Apre[m]/tauc

        while timestepsc[positionc] == step:
            for n in range(NE): 
                v[n] += w[n, q[indicesc[positionc]]]
                w[n, q[indicesc[positionc]]] = \
                    np.clip(w[n, q[indicesc[positionc]]]*(1+rac*Apost[n]), 0, 1)
            Apre[q[indicesc[positionc]]] += 1  # pre-synaptic tracer
            positionc += 1

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
        
        v[N] = 0  # empty the incinerator

    return w, count_target/T/(fs*NE), count_nontarget/T/((1-fs)*NE), count_inhib/T/NI
