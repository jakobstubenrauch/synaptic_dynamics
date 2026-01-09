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
Produce Figure 4 of the manuscript.
Evolution of four coupled synaptic populations.
"""
import numpy as np
from network_training_scenario_theory_and_wrappers import sc_variance_fp, four_pop_coupled_theory, wrap_fourpop


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle')

    vr, vt, muE, muI, DE, DI = 0, 1, 0, .5, .1, .05  # neuron
    NE, NI, CE, CI, J, g, h = 4000, 1000, 200, 50, 0.01, 5., 2.  # network
    # NE, NI, CE, CI, J, g, h = 1000, 250, 100, 25, 0.01, 5., 2.  # network
    M, fc, fs, nu, m0, nus, Js = 200, .05, .1, .5, .05, 64, .0125  # input
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20  # STDP
    dt, T, Twarm = 1e-4, 50, 10  # simulation
    nu0, nu1 = 0.1, 1  #0.8

    # Compute the self-consistent variance
    scV, m00, m01, m10, m11, V00, V01, V10, V11, gamma0, gamma1, r0, r1, rI = sc_variance_fp(
        vr, vt, muE, DE, muI, DI, CE, CI, J, g, h, m0, M, nu0, nu1, fc, fs, Js, nus,
        Dc, rac, tauc, tauac, T)
    V0 = scV

    
    t_ = 1e-2 * np.arange(int(np.round(T/1e-2)))
    fig, axsr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(3.375, 1*2.085864712))
    fig.suptitle('Mean and variance per population under STDP')

    dt = 1e-2
    t_arr, m00, m01, m10, m11, V00, V01, V10, V11 = four_pop_coupled_theory(
        vr, vt, muE, DE, muI, DI,  # neuron
        CE, CI, J, g, h,  # network
        m0, V0, M, nu0, nu1, fc, fs, Js, nus,  # input
        Dc, rac, tauc, tauac, dt, T)
    
    ylim = (m11[0]-np.sqrt(V11[0])*1.1, m11[-1]+np.sqrt(V11[-1])*1.1)
    gamma0 = m0/(fc*m01[-1]+(1-fc)*m00[-1])
    gamma1 = m0/(fc*m11[-1]+(1-fc)*m10[-1])
    
    axsr[0, 0].plot(t_arr, m00, 'k')
    axsr[0, 0].fill_between(t_arr, m00-np.sqrt(V00), m00+np.sqrt(V00), color='k', alpha=.4)
    axsr[0, 1].plot(t_arr, m01, 'k')
    axsr[0, 1].fill_between(t_arr, m01-np.sqrt(V01), m01+np.sqrt(V01), color='k', alpha=.4)
    axsr[1, 0].plot(t_arr, m10, 'k')
    axsr[1, 0].fill_between(t_arr, m10-np.sqrt(V10), m10+np.sqrt(V10), color='k', alpha=.4)
    axsr[1, 1].plot(t_arr, m11, 'k')
    axsr[1, 1].fill_between(t_arr, m11-np.sqrt(V11), m11+np.sqrt(V11), color='k', alpha=.4)
    axsr[0, 0].errorbar([1.4*T], [gamma0*m00[-1]],
        yerr=[np.sqrt(gamma0**2*V00[-1])], color='k', marker='o', ls='', capsize=3)
    axsr[0, 1].errorbar([1.4*T], [gamma0*m01[-1]],
        yerr=[np.sqrt(gamma0**2*V01[-1])], color='k', marker='o', ls='', capsize=3)
    axsr[1, 0].errorbar([1.4*T], [gamma1*m10[-1]],
        yerr=[np.sqrt(gamma1**2*V10[-1])], color='k', marker='o', ls='', capsize=3)
    axsr[1, 1].errorbar([1.4*T], [gamma1*m11[-1]],
        yerr=[np.sqrt(gamma1**2*V11[-1])], color='k', marker='o', ls='', capsize=3)
    

    m00, m01, m10, m11, V00, V01, V10, V11 = wrap_fourpop( 
        vr,  vt,  muE,  muI,  DE,  DI,
        NE, NI, CE, CI,  J,  g,  h,
        m0, M,  nu0,  nu1,  nus,  fc,  fs,  Js,
        Dc,  rac,  tauc,  tauac, 
        T,  dt,  Twarm,  V0, trials=1)
    numdict = {'ls': '', 'color': 'r', 'marker': 'o'}
    axsr[0, 0].plot([-1], [-1], ls='', label='non-cue to non-target')
    axsr[0, 1].plot([-1], [-1], ls='', label='cue to non-target')
    axsr[1, 0].plot([-1], [-1], ls='', label='non-cue to target')
    axsr[1, 1].plot([-1], [-1], ls='', label='cue to target')

    # t_ = dt*100*np.arange(int(np.round(T/(dt*100))))
    t_ = dt*100*np.arange(int(np.round(T/(dt*100))))
    jump = 400
    axsr[0, 0].errorbar(t_[::jump], m00[::jump], yerr=np.sqrt(V00)[::jump], **numdict)
    axsr[0, 1].errorbar(t_[::jump], m01[::jump], yerr=np.sqrt(V01)[::jump], **numdict)
    axsr[1, 0].errorbar(t_[::jump], m10[::jump], yerr=np.sqrt(V10)[::jump], **numdict)
    axsr[1, 1].errorbar(t_[::jump], m11[::jump], yerr=np.sqrt(V11)[::jump], **numdict)
    axsr[0, 0].set_title('non-cue to non-target')
    axsr[0, 1].set_title('cue to non-target')
    axsr[1, 0].set_title('non-cue to target')
    axsr[1, 1].set_title('cue to target')
    for ax in axsr[:, 0]: ax.set_yticks([m0, 2*m0], [r'$m_0$', r'$2m_0$'])
    for ax in axsr[1, :]: 
        ax.set_xticks([0, T, 1.4*t_arr[-1]], [r'$0$', r'$T$', 'post\n homeostasis'])
    axsr[0, 0].set_xlim(-2, 1.4*t_arr[-1]+2)
    axsr[0, 0].set_ylim(ylim[0], ylim[1])

    for ax in axsr[1]:
        ax.set_xlabel('time')
    for ax in axsr[:, 0]:
        ax.set_ylabel('weight')

    fig.savefig('figs/v2_population_dynamics.pdf')

    plt.show()


