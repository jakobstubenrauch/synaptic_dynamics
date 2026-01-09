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
Implementation of the Fokker-Planck theory for the synaptic weight dynamics
of a single synapse feeding Poisson spikes into a LIF neuron under STDP.
"""
import numpy as np
from scipy.integrate import solve_ivp
from response_functions import alpha, beta, siegert_rateD


def drift_and_diffusion_coefficient(
    vr, vt, mu, D,  # neuron - input WITHOUT w0*nu
    nu,  # input
    Dc, rac, tauc, tauac,  # training
    w_arr, contributions='all'):
    """
    Compute the drift- and infinitesimal diffusion coefficient of the synaptic 
    process (if contributions='all') or only single contributions
    in ('rates', 'mean_response', 'noise_intensity_response')
    """

    mu_tot = mu + w_arr*nu
    D_tot = D + .5*w_arr**2*nu
    r0s = siegert_rateD(mu_tot, D_tot, vr, vt)

    D2 = (1/4)*r0s*nu*(Dc**2*tauc + rac**2*w_arr**2*tauac)
    if contributions == 'rates':
        D1 = (Dc*tauc+rac*w_arr*tauac)*r0s*nu
    elif contributions == 'mean_response':
        alphas = alpha(mu_tot, D_tot, vr, vt, -1/tauc)
        D1 = Dc*nu*w_arr*np.array(alphas, dtype=float)
    elif contributions == 'noise_intensity_response':
        betas = beta(mu_tot, D_tot, vr, vt, -1/tauc)
        D1 = 0.5*Dc*nu*w_arr**2*np.array(betas, dtype=float)
    elif contributions == 'all':
        # rates
        D1 = (Dc*tauc+rac*w_arr*tauac)*r0s*nu
        # mean response 
        alphas = alpha(mu_tot, D_tot, vr, vt, -1/tauc)
        D1 += Dc*nu*w_arr*np.array(alphas, dtype=float)
        # noise intensity response
        betas = beta(mu_tot, D_tot, vr, vt, -1/tauc)
        D1 += 0.5*Dc*nu*w_arr**2*np.array(betas, dtype=float)
    
    return D1, D2, r0s


def cumulated_diffusion(
    vr, vt, mu, D,  # neuron - input WITHOUT w0*nu
    nu,  # input
    Dc, rac, tauc, tauac,  # training
    m0, Dt, eps=1e-8):
    """
    Compute the cumulated diffusion over time Dt by integrating Eqs. (24,25)
    """

    def rhs(t, x):
        """
        Right-hand side of the ODEs for mean and variance evolution
        as given in Eqs. (24,25). Here, t is a dummy time argument required
        by solve_ivp, and x = [m, V] contains the mean and variance
        of the synaptic variable w.
        """
        m, V = x
        # PERTURBED drift (to obtain derivative w.r.t. m)
        mu_c = (m+eps)*nu
        D_c = 0.5*(m+eps)**2*nu
        mutot = mu + mu_c
        Dtot = D + D_c
        r = siegert_rateD(mutot, Dtot)
        alpha_ = alpha(mutot, Dtot, vr, vt, -1/tauc, r)
        beta_ = beta(mutot, Dtot, vr, vt, -1/tauc, r)
        D1eps = (Dc*tauc+rac*m*tauac)*nu*r \
            + Dc*nu*(m*alpha_ + 0.5*m**2*beta_) 
        
        # TRUE
        mu_c = m*nu
        D_c = 0.5*m**2*nu
        mutot = mu + mu_c
        Dtot = D + D_c
        r = siegert_rateD(mutot, Dtot)

        alpha_ = alpha(mutot, Dtot, vr, vt, -1/tauc, r)
        beta_ = beta(mutot, Dtot, vr, vt, -1/tauc, r)

        D1 = (Dc*tauc+rac*m*tauac)*nu*r \
            + Dc*nu*(m*alpha_ + 0.5*m**2*beta_) 
        D2 = (1/4)*r*nu*(Dc**2*tauc + rac**2*m**2*tauac)
        D1p = (D1eps-D1)/eps

        return np.array([D1, 2*D1p*V+r*nu/2 * (Dc**2*tauc + rac**2*tauac*(V+m**2))])
    
    # start integration
    res = solve_ivp(rhs, [0, Dt], np.array([m0, 0]))
    Vf = res.y[1, -1]
    mf = res.y[0, -1]
    return (Vf + (mf-m0)**2) / (2*Dt)
    
