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

"""
This script contains:
- Implementations of the theory
- Wrappers for the cython-based simulations
The system studied here is a plastic feed-forward layer with presynaptic
Poisson process and a postsynaptic recurrent network of LIF neurons.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from cy_code.bipoo_ff_into_brunel_training import train, train_no_homeostasis
from cy_code.bipoo_ff_fourpop import train as train_fourpop
from utilities.tools import check, load, save, makename, makehash
from response_functions import alpha, beta, siegert_rateD


def brunel_TNI_theory(
    vr, vt, CE, CI, J, g, h, muT, muN, muI, DT, DN, DI, fs,
    epsilon=.3, maxiter=100000, atol=1e-6):
    """
    Determine the firing rates of a three-population variant of
    Brunel et al 2000, Model A.
    """
    # Initial guesses for the rates (not very sensitive)
    rT, rN, rI = .1, .1, .1

    for i in range(maxiter):
        # input statistics
        mu_netE = J*CE*(fs*rT+(1-fs)*rN) - g*J*CI*rI
        D_netE = 0.5*(J**2*CE*(fs*rT+(1-fs)*rN) + (g*J)**2*CI*rI) 
        mu_netI = h*J*CE*(fs*rT+(1-fs)*rN) - g*J*CI*rI
        D_netI = 0.5*((h*J)**2*CE*(fs*rT+(1-fs)*rN) + (g*J)**2*CI*rI)

        # firing rates of the three populations
        rTnew = siegert_rateD(muT+mu_netE, DT+D_netE, vr, vt)
        rNnew = siegert_rateD(muN+mu_netE, DN+D_netE, vr, vt)
        rInew = siegert_rateD(muI+mu_netI, DI+D_netI, vr, vt)

        # check convergence
        errors = np.abs([rT-rTnew, rN-rNnew, rI-rInew])
        if np.max(errors) < atol:
            return rT, rN, rI, mu_netE, D_netE, mu_netI, D_netI

        # update rates
        rT = (1-epsilon) * rT + epsilon * rTnew
        rN = (1-epsilon) * rN + epsilon * rNnew
        rI = (1-epsilon) * rI + epsilon * rInew


def wrap_training(sargs, trials, redo=False):
    """
    This is a wrapper for the cython-based method "train" that simulates the 
    training process of a feed-forward layer into a Brunel network.

    The wrapper enables parallelization and caching of results.
    """

    prename = 'bipoo_train'
    name = prename + makename(sargs) + str(trials)
    print(name, check(name))
    if len(name) > 250:  # most OSs require len(filename) < 256 chars
        name = prename + makehash(sargs) + str(trials)  # less readable but compact and unique

    if not redo and check(name):
        return load(name, 10)

    if trials == 1:
        res = train(**(sargs | {'seed': 0}))
    else:
        import multiprocess as mp
        with mp.Pool(mp.cpu_count()-2) as pool:
            workers = [pool.apply_async(train, (),
                sargs | {'seed': seed}) for seed in range(trials)]
            res = np.mean([wrk.get() for wrk in workers], axis=0)
    save(name, res)
    return res


def wrap_training_return_weights(sargs, redo=False):
    """
    Same as wrap_training but returns the weights during training.
    The function is left separate to avoid unnecessary data storage in
    the method above.
    """

    prename = 'nnnbipoo_train_ret_weights'
    name = prename + makename(sargs)
    print(name)
    if len(name) > 250:  # most OSs require len(filename) < 256 chars
        name = prename + makehash(sargs)  # less readable but compact and unique

    if not redo and check(name):
        return load(name, 5)

    wspre, wspost, wT, wN, wNC = train(**(sargs | {'seed': 0, 'return_weights': True}))
    save(name, [wspre, wspost, wT, wN, wNC])
    return wspre, wspost, wT, wN, wNC


def wrap_training_return_confusion(sargs, trials, redo=False):
    """
    This is a wrapper for the cython-based method "train" that simulates the
    training process of a feed-forward layer into a Brunel network.

    The wrapper enables parallelization and caching of results.

    In addition to wrap_training, this function measures the fraction of
    correct hits.
    """
    from time import time
    assert 'Trecall' in sargs.keys()
    prename = 'nnbipoo_train_ret_conf'  # changed to include simulation metadata
    name = prename + makename(sargs) + str(trials)
    if len(name) > 250:  # most OSs require len(filename) < 256 chars
        name = prename + makehash(sargs) + str(trials)  # less readable but compact and unique

    print(name, check(name))
    if not redo and check(name):
        return load(name, 14)
    elif not redo and check(name[1:]):  # before I added metadata
        return load(name[1:], 13)

    start = time()
    num_cpus = 1
    if trials == 1:
        res = train(**(sargs | {'seed': 0, 'return_confusion': True}))
    else:
        import multiprocess as mp
        num_cpus = mp.cpu_count()-2
        with mp.Pool(num_cpus) as pool:
            workers = [pool.apply_async(train, (),
                sargs | {'seed': seed, 'return_confusion': True}) \
                     for seed in range(trials)]
            res = np.mean([wrk.get() for wrk in workers], axis=0)
    end = time()
    metadata = [num_cpus, end-start]
    save(name, [*res, metadata])
    return *res, metadata


def wrap_training_return_confusion_no_homeostasis(sargs, trials, redo=False):
    """
    This is a wrapper for the cython-based method "train_no_homeostasis" that simulates the
    training process of a feed-forward layer into a Brunel network without homeostasis.
    """
    from time import time
    assert 'Trecall' in sargs.keys()
    prename = 'nnbipoo_train_ret_conf_no_hs'  # changed to include simulation metadata
    name = prename + makename(sargs) + str(trials)
    if len(name) > 250:  # most OSs require len(filename) < 256 chars
        name = prename + makehash(sargs) + str(trials)  # less readable but compact and unique

    print(name, check(name))
    if not redo and check(name):
        return load(name, 16)

    start = time()
    num_cpus = 1
    if trials == 1:
        res = train_no_homeostasis(**(sargs | {'seed': 0, 'return_confusion': True}))
    else:
        import multiprocess as mp
        num_cpus = mp.cpu_count()-2
        with mp.Pool(num_cpus) as pool:
            workers = [pool.apply_async(train_no_homeostasis, (),
                sargs | {'seed': seed, 'return_confusion': True}) \
                     for seed in range(trials)]
            res = np.mean([wrk.get() for wrk in workers], axis=0)
    end = time()
    metadata = [num_cpus, end-start]
    save(name, [*res, metadata])
    # save(name, res)
    return *res, metadata


def sc_variance_fp(
    vr, vt, muE, DE, muI, DI,  # neuron
    CE, CI, J, g, h,  # network
    m0, M, nu0, nu1, fc, fs, Js, nus,  # input
    Dc, rac, tauc, tauac, T,  # training
    contributions=['all']):
    """
    This function computes the stationary variance of synaptic weights, Eq. (39).
    
    This function contains:
    - The ODE system's rhs "rhs" for the means and variances of the synaptic 
      weights Eq. (32)
    - The stationary condition "zero" for the variance Eq. (39) which demands that
      the variance after one training period T equals the initial variance.
    - The root finding call on "zero" to determine the stationary variance.
    """
    if 'all' in contributions: contributions = ['rates', 'mean', 'noise']

    def rhs(t, x):
        """
        R.h.s. of Eq. (32) for the means and variances of the synaptic weights.
        Here, t is a dummy argument required by solve_ivp, in reality the system
        is autonomous.
        """
        m00, m01, m10, m11, V00, V01, V10, V11 = x

        # Determine the input statistics to the recurrent network
        mu0_c = M*(m01*fc*nu1+m00*(1-fc)*nu0)
        mu1_c = M*(m11*fc*nu1+m10*(1-fc)*nu0)
        D0_c = (1/2)*M*((m01**2+V01)*fc*nu1 + (m00**2+V00)*(1-fc)*nu0)
        D1_c = (1/2)*M*((m11**2+V11)*fc*nu1 + (m10**2+V10)*(1-fc)*nu0)
        mus, Ds = nus*Js, 0.5*Js**2*nus 

        # Evaluate the brunel theory for the three-population network
        # to get the firing rates of target and non-target neurons
        r1, r0, rI, mu_netE, D_netE, mu_netI, D_netI = brunel_TNI_theory(
            vr, vt, CE, CI, J, g, h, muE+mu1_c+mus, muE+mu0_c, muI, DE+D1_c+Ds, 
            DE+D0_c, DI, fs)

        # Conclude from above the diffusion approximation of inputs to the 
        # representative LIF neurons
        mu1 = muE + mu1_c + mus + mu_netE
        mu0 = muE + mu0_c + mu_netE
        D1 = DE + D1_c + Ds + D_netE
        D0 = DE + D0_c + D_netE

        # Rate based contributions to the drift coefficients
        drift00 = (Dc*tauc+rac*m00*tauac)*nu0*r0
        drift01 = (Dc*tauc+rac*m01*tauac)*nu1*r0
        drift10 = (Dc*tauc+rac*m10*tauac)*nu0*r1
        drift11 = (Dc*tauc+rac*m11*tauac)*nu1*r1
        # Rate based contributions to the derivative of the drift coefficients
        drift00_p = rac*tauac*nu0*r0
        drift01_p = rac*tauac*nu1*r0
        drift10_p = rac*tauac*nu0*r1
        drift11_p = rac*tauac*nu1*r1

        if 'mean' in contributions:
            # Add the mean response contributions to the drift coefficients
            # and their derivatives
            alpha0 = alpha(mu0, D0, vr, vt, -1/tauc, r0)
            alpha1 = alpha(mu1, D1, vr, vt, -1/tauc, r1)
            drift00 += Dc*nu0*m00*alpha0
            drift01 += Dc*nu1*m01*alpha0
            drift10 += Dc*nu0*m10*alpha1
            drift11 += Dc*nu1*m11*alpha1
            drift00_p += Dc*nu0*alpha0
            drift01_p += Dc*nu1*alpha0
            drift10_p += Dc*nu0*alpha1
            drift11_p += Dc*nu1*alpha1
        if 'noise' in contributions:
            # Add the noise response contributions to the drift coefficients
            # and their derivatives
            beta0 = beta(mu0, D0, vr, vt, -1/tauc, r0)
            beta1 = beta(mu1, D1, vr, vt, -1/tauc, r1)
            drift00 += Dc*nu0*0.5*m00**2*beta0
            drift01 += Dc*nu1*0.5*m01**2*beta0
            drift10 += Dc*nu0*0.5*m10**2*beta1
            drift11 += Dc*nu1*0.5*m11**2*beta1
            drift00_p += Dc*nu0*m00*beta0
            drift01_p += Dc*nu1*m01*beta0
            drift10_p += Dc*nu0*m10*beta1
            drift11_p += Dc*nu1*m11*beta1

        # Lump together the right-hand sides of the variance equations
        rhsV00 = 2*drift00_p*V00 + r0*nu0/2 * (Dc**2*tauc + rac**2*tauac*(V00+m00**2))
        rhsV01 = 2*drift01_p*V01 + r0*nu1/2 * (Dc**2*tauc + rac**2*tauac*(V01+m01**2))
        rhsV10 = 2*drift10_p*V10 + r1*nu0/2 * (Dc**2*tauc + rac**2*tauac*(V10+m10**2))
        rhsV11 = 2*drift11_p*V11 + r1*nu1/2 * (Dc**2*tauc + rac**2*tauac*(V11+m11**2))

        return np.array([drift00, drift01, drift10, drift11,
                        rhsV00, rhsV01, rhsV10, rhsV11])
    

    def zero(V, return_all=False):
        """
        Difference of initial variance and marginal variance after one step.
        Must vanish at the stationary variance.
        """

        init = np.array([m0, m0, m0, m0, V, V, V, V])
        sol = solve_ivp(rhs, [0, T], init)
        m00, m01, m10, m11, V00, V01, V10, V11 = sol.y[:, -1]

        # Homeostatic rescaling factors
        gamma0 = m0/(fc*m01 + (1-fc)*m00)
        gamma1 = m0/(fc*m11 + (1-fc)*m10)

        if return_all:
            mu0_c = M*(m01*fc*nu1+m00*(1-fc)*nu0)
            mu1_c = M*(m11*fc*nu1+m10*(1-fc)*nu0)
            D0_c = (1/2)*M*((m01**2+V01)*fc*nu1 + (m00**2+V00)*(1-fc)*nu0)
            D1_c = (1/2)*M*((m11**2+V11)*fc*nu1 + (m10**2+V10)*(1-fc)*nu0)
            mus, Ds = nus*Js, 0.5*Js**2*nus 

            r1, r0, rI, mu_netE, D_netE, mu_netI, D_netI = brunel_TNI_theory(
                vr, vt, CE, CI, J, g, h, muE+mu1_c+mus, muE+mu0_c, muI, DE+D1_c+Ds, DE+D0_c, DI, fs)
            
            print('D1 constituents: intr, input, net, super', DE, D1_c, D_netE, Ds)
            print('D0 constituents: intr, input, net', DE, D0_c, D_netE)
            print('DI_tot constituents: intr, input, net', DI, D_netI)
            return m00, m01, m10, m11, V00, V01, V10, V11, gamma0, gamma1, r0, r1, rI

        # The variance after one training period T with homeostatic rescaling 
        # is a weighted sum of population-wise variances and displacements, Eq. (37)
        Vnew = (1-fc)*(1-fs)*(gamma0**2*V00+(gamma0*m00-m0)**2)+\
            (1-fc)*fs*(gamma1**2*V10+(gamma1*m10-m0)**2)+\
            fc*(1-fs)*(gamma0**2*V01+(gamma0*m01-m0)**2)+\
            fc*fs*(gamma1**2*V11+(gamma1*m11-m0)**2)

        return float(V-Vnew)
    
    a, b = 1e-10, 1e-1  # 1e-7, 1e-3
    try:
        scV = brentq(zero, a, b, full_output=False)
    except ValueError:
        print('sc variance failed')
        return 0

    m00, m01, m10, m11, V00, V01, V10, V11, gamma0, gamma1, r0, r1, rI = zero(scV, return_all=True)
    return scV, m00, m01, m10, m11, V00, V01, V10, V11, gamma0, gamma1, r0, r1, rI


def fp_traces_and_afp(
    vr, vt, muE, DE, muI, DI,  # neuron
    CE, CI, J, g, h,  # network
    m0, M, nu0, nu1, fc, fs, Js, nus,  # input
    Dc, rac, tauc, tauac, T,  # training
    P, contributions=['all']):  # number of associations to consider
    from scipy.special import erfc
    """
    Track the means and variances of subpopulations w.r.t. a fixed reference
    association during the training of subsequent associations.
    """

    # Determine the properties of the stationary state
    scV, m00, m01, m10, m11, V00, V01, V10, V11, gamma0, gamma1, r0, r1, rI = sc_variance_fp(
        vr, vt, muE, DE, muI, DI,  # neuron
        CE, CI, J, g, h,  # network
        m0, M, nu0, nu1, fc, fs, Js, nus,  # input
        Dc, rac, tauc, tauac, T,  # training
        contributions=contributions)
    
    Mc = fc*M

    # Parameters from Eq. (43) determining trace dynamics
    phi = 1-fs*gamma1-(1-fs)*gamma0
    c = (1-fs)*(1-fc)*gamma0*(m00-m0) \
        + (1-fs)*fc*gamma0*(m01-m0) \
        + fs*(1-fc)*gamma1*(m10-m0) \
        + fs*fc*gamma1*(m11-m0)

    # Fraction of false positives (reversed to the fraction of correctly 
    # activated neurons used in the manuscript)
    afp = []  

    # Initialize the population-wise mean and variance traces
    Kk = np.zeros((P, 2, 2))
    Gk = np.zeros((P, 2, 2))
    # Eqs. (40) and (41)
    Kk[0] = np.array([[gamma0*m00, gamma0*m01], [gamma1*m10, gamma1*m11]])
    Gk[0] = np.array([[gamma0**2*V00, gamma0**2*V01], [gamma1**2*V10, gamma1**2*V11]])

    for k in range(P):
        if k >= 1:

            # Eq. (44)
            Kk[k] = (1-phi)*Kk[k-1] + c

            # Eq. (46)
            Gk[k] = (1-fs)*(1-fc)*gamma0**2*(Gk[k-1]+V00-scV+(Kk[k-1]+m00-m0)**2) \
                + (1-fs)*fc*gamma0**2*(Gk[k-1]+V01-scV+(Kk[k-1]+m01-m0)**2) \
                + fs*(1-fc)*gamma1**2*(Gk[k-1]+V10-scV+(Kk[k-1]+m10-m0)**2) \
                + fs*fc*gamma1**2*(Gk[k-1]+V11-scV+(Kk[k-1]+m11-m0)**2)-Kk[k]**2

        # Needed in Eq. (52)        
        sqrtT = np.sqrt(2*fc*M*Gk[k, 1, 1])
        sqrtN = np.sqrt(2*fc*M*Gk[k, 0, 1])

        def zero(s):
            # rhs-lhs of Eq. (52)
            return fs - fs*erfc((s-Mc*Kk[k, 1, 1])/sqrtT)/2 - \
                (1-fs)*erfc((s-Mc*Kk[k, 0, 1])/sqrtN)/2

        # Determine the (1-fs)th percentile of summed weight, i.e., the root of "zero"
        s_ast = brentq(zero, 0, 4*Mc*Kk[k, 1, 1])
        # 1-Eq.(53): CAREFUL: this is the fraction of false positives 
        # afp = 1 - a, where a is used in the manuscript
        afp.append(1-erfc((s_ast-fc*M*Kk[k, 1, 1])/sqrtT)/2)

    # return afp
    return scV, Kk, Gk, np.array(afp)


def four_pop_coupled_theory(
    vr, vt, muE, DE, muI, DI,  # neuron
    CE, CI, J, g, h,  # network
    m0, V0, M, nu0, nu1, fc, fs, Js, nus,  # input
    Dc, rac, tauc, tauac, dt, T):
    """
    Integrate Eq. (32) for the means and variances of synaptic weights
    without homeostasis over time T with time step dt.
    """

    def rhs(t, x):
        """Rhs of Eq. (32) for the means and variances of the synaptic weights.
        Here, t is a dummy argument required by solve_ivp, in reality the system
        is autonomous."""
        m00, m01, m10, m11, V00, V01, V10, V11 = x

        # Determine the input statistics to the recurrent network
        mu0_c = M*(m01*fc*nu1+m00*(1-fc)*nu0)
        mu1_c = M*(m11*fc*nu1+m10*(1-fc)*nu0)
        D0_c = (1/2)*M*((m01**2+V01)*fc*nu1 + (m00**2+V00)*(1-fc)*nu0)
        D1_c = (1/2)*M*((m11**2+V11)*fc*nu1 + (m10**2+V10)*(1-fc)*nu0)
        mus, Ds = nus*Js, 0.5*Js**2*nus 

        # Evaluate the brunel theory for the three-population network
        # to get the firing rates of target and non-target neurons
        r1, r0, rI, mu_netE, D_netE, mu_netI, D_netI = brunel_TNI_theory(
            vr, vt, CE, CI, J, g, h, muE+mu1_c+mus, muE+mu0_c, muI, DE+D1_c+Ds, 
            DE+D0_c, DI, fs)

        # Conclude from above the diffusion approximation of inputs to the 
        # representative LIF neurons
        mu1 = muE + mu1_c + mus + mu_netE
        mu0 = muE + mu0_c + mu_netE
        D1 = DE + D1_c + Ds + D_netE
        D0 = DE + D0_c + D_netE

        alpha0 = alpha(mu0, D0, vr, vt, -1/tauc, r0)
        beta0 = beta(mu0, D0, vr, vt, -1/tauc, r0)
        alpha1 = alpha(mu1, D1, vr, vt, -1/tauc, r1)
        beta1 = beta(mu1, D1, vr, vt, -1/tauc, r1)

        # Drift coefficients Eq. (31)
        drift00 = (Dc*tauc+rac*m00*tauac)*nu0*r0 \
            + Dc*nu0*(m00*alpha0 + 0.5*m00**2*beta0)  # maybe m00**2 + V00?
        drift01 = (Dc*tauc+rac*m01*tauac)*nu1*r0 \
            + Dc*nu1*(m01*alpha0 + 0.5*m01**2*beta0)
        drift10 = (Dc*tauc+rac*m10*tauac)*nu0*r1 \
            + Dc*nu0*(m10*alpha1 + 0.5* m10**2*beta1)
        drift11 = (Dc*tauc+rac*m11*tauac)*nu1*r1 \
            + Dc*nu1*(m11*alpha1 + 0.5*m11**2*beta1)
        
        # Derivatives of the drift coefficients
        drift00_p = +rac*tauac*nu0*r0 + Dc*nu0*(alpha0+m00*beta0)
        drift01_p = +rac*tauac*nu1*r0 + Dc*nu1*(alpha0+m01*beta0)
        drift10_p = +rac*tauac*nu0*r1 + Dc*nu0*(alpha1+m10*beta1)
        drift11_p = +rac*tauac*nu1*r1 + Dc*nu1*(alpha1+m11*beta1)

        # Lump together the right-hand sides of the variance equations
        rhsV00 = 2*drift00_p*V00 + r0*nu0/2 * (Dc**2*tauc + rac**2*tauac*(V00+m00**2))
        rhsV01 = 2*drift01_p*V01 + r0*nu1/2 * (Dc**2*tauc + rac**2*tauac*(V01+m01**2))
        rhsV10 = 2*drift10_p*V10 + r1*nu0/2 * (Dc**2*tauc + rac**2*tauac*(V10+m10**2))
        rhsV11 = 2*drift11_p*V11 + r1*nu1/2 * (Dc**2*tauc + rac**2*tauac*(V11+m11**2))

        return np.array([drift00, drift01, drift10, drift11,
                         rhsV00, rhsV01, rhsV10, rhsV11])
    
    init = np.array([m0, m0, m0, m0, V0, V0, V0, V0])
    t_arr = dt * np.arange(int(np.round(T/dt)))
    sol = solve_ivp(rhs, [0, T], init, t_eval=t_arr, method='RK45')
    m00, m01, m10, m11, V00, V01, V10, V11 = sol.y
    return t_arr, m00, m01, m10, m11, V00, V01, V10, V11


def wrap_fourpop( 
        vr,  vt,  muE,  muI,  DE,  DI,
        NE, NI, CE, CI,  J,  g,  h,
        m0, M,  nu0,  nu1,  nus,  fc,  fs,  Js,
        Dc,  rac,  tauc,  tauac, 
        T,  dt,  Twarm,  V0, trials, seed=234, redo=False):
    
    """
    Wrap the cython-based simulation of a feed-forward layer into a
    four-population recurrent network with plastic synapses.
    """
    
    # Check if already simulated
    name = 'wrap_fourpop' + makename([vr,  vt,  muE,  muI,  DE,  DI,
        NE, NI, CE, CI,  J,  g,  h,
        m0, M,  nu0,  nu1,  nus,  fc,  fs,  Js,
        Dc,  rac,  tauc,  tauac, 
        T,  dt,  Twarm,  V0, trials])
    if len(name) > 250:
        name = 'wrap_fourpop' + makehash([vr,  vt,  muE,  muI,  DE,  DI,
        NE, NI, CE, CI,  J,  g,  h,
        m0, M,  nu0,  nu1,  nus,  fc,  fs,  Js,
        Dc,  rac,  tauc,  tauac, 
        T,  dt,  Twarm,  V0, trials])

    print(name, check(name))

    if not redo and check(name):
        return load(name, 8)
    
    # deploy simulation
    assert trials == 1, 'Only single trial supported so far!'
    m00, m01, m10, m11, V00, V01, V10, V11 = train_fourpop(
        seed, vr,  vt,  muE,  muI,  DE,  DI,
        NE, NI, CE, CI,  J,  g,  h,
        m0, M,  nu0,  nu1,  nus,  fc,  fs,  Js,
        Dc,  rac,  tauc,  tauac, 
        T,  dt,  Twarm,  V0, only_endpoint=0)
    save(name, [m00, m01, m10, m11, V00, V01, V10, V11])
    return m00, m01, m10, m11, V00, V01, V10, V11
