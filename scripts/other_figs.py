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
Produce Figures 3, 6, 7, and 8 of the manuscript.
Uncomment the respective function calls at the bottom of this file to run them.
"""
import numpy as np
import multiprocess as mp
from functools import partial
from single_synapse_theory import drift_and_diffusion_coefficient
from utilities.tools import check, load, save, makename, makehash, autoscale_y
from network_training_scenario_theory_and_wrappers import brunel_TNI_theory, fp_traces_and_afp, wrap_training_return_confusion_no_homeostasis
from response_functions import siegert_rateD, siegert_cv_unitless
from scripts.fig2 import km_coeffs
from scipy.optimize import brentq


def capa_and_initial_varaparam(targs, key, vals, redo=False):
    """
    Compute the capacity and accuracy upon varying the parameter given by 'key'
    across the values in 'vals'.
    """

    name = 'cap_and_init_acc' + makename(targs) + key + makehash(vals)
    if not redo and check(name):
        return load(name, 4)

    with mp.Pool(mp.cpu_count()-2) as pool:
        wrkrs = [pool.apply_async(fp_traces_and_afp, (), 
                targs | {key: val}) for val in vals]
        afps = [wrk.get()[-1] for wrk in wrkrs]
        afp0s = np.array([afp[0] for afp in afps])
        capas = np.array([np.argmin(np.abs(np.asarray(afp)-0.5)) for afp in afps])

    targs['contributions'] = ['rates']
    with mp.Pool(mp.cpu_count()-2) as pool:
        wrkrs = [pool.apply_async(fp_traces_and_afp, (), 
                targs | {key: val}) for val in vals]
        afps = [wrk.get()[-1] for wrk in wrkrs]
        afp0s_rates = np.array([afp[0] for afp in afps])
        capas_rates = np.array([np.argmin(np.abs(np.asarray(afp)-0.5)) for afp in afps])

    targs.pop('contributions')
    save(name, [capas, afp0s, capas_rates, afp0s_rates])
    return capas, afp0s, capas_rates, afp0s_rates


def speed_accuracy_tradeoff(targs, fact_arr, redo=False):
    """
    Simultaniosly vary Dc, rac, and T by same factor to investigate the effect
    of speed on accuracy and capacity.
    """

    name = 'nnspeed_accuracy' + makename(targs) + makehash(fact_arr)
    if not redo and check(name):
        return load(name, 3)
    
    Dcs = targs['Dc']*fact_arr
    racs = targs['rac']*fact_arr
    Ts = targs['T']/fact_arr

    with mp.Pool(mp.cpu_count()-2) as pool:
        wrkrs = [pool.apply_async(fp_traces_and_afp, (), 
                targs | {'Dc': Dcs[i], 'rac': racs[i], 'T': Ts[i]}) for i in range(len(fact_arr))]
        res = [wrk.get() for wrk in wrkrs]
        K11s = np.array([r[1][0, 1, 1] for r in res])
        afps = [r[-1] for r in res]
        afp0s = np.array([afp[0] for afp in afps])
        capas = np.array([np.argmin(np.abs(np.asarray(afp)-0.5)) for afp in afps])

    save(name, [afp0s, capas, K11s])
    return afp0s, capas, K11s


def fig6_tradeoffs():
    """
    Investigate speed-accuracy and accuracy-capacity tradeoffs by varying
    parameters of the STDP learning rule and training time.
    """
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle') 

    vr, vt, muE, muI, DE, DI = 0, 1, 0, .5, .1, .05  # neuron
    NE, NI, CE, CI, J, g, h = 4000, 1000, 200, 50, 0.01, 5., 2.  # network
    M, fc, fs, m0, nus, Js = 200, .05, .1, .05, 64, .0125  # input
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20  # STDP
    dt, T, Twarm = 1e-4, 50, 10  # simulation
    nu0, nu1 = 0, 1  #0.8
    P = 300

    targs = dict(zip(  # parameters relevant for theory
        ['vr', 'vt', 'muE', 'muI', 'DE', 'DI', 'CE', 'CI', 'J', 'g', 'h', 
        'm0', 'M', 'nu0', 'nu1', 'fc', 'fs', 'Js', 'nus', 'Dc', 'rac', 'tauc', 
        'tauac', 'T', 'P'],
        [vr, vt, muE, muI, DE, DI, CE, CI, J, g, h, m0, M, nu0, nu1, fc, fs,
        Js, nus, Dc, rac, tauc, tauac, T, P]))
    
    redo = True
    import time
    start = time.time()    
    
    # Vary different single parameters and observe effect on accuracy and capacity
    T_arr = np.array(np.round(np.linspace(10, 80, 40)), dtype=int)
    capasT, afp0sT, capas_ratesT, afp0s_ratesT = capa_and_initial_varaparam(
        targs, 'T', T_arr, redo=redo)
    Dc_arr = np.linspace(0.45*Dc, Dc*2, 40)
    capasDc, afp0sDc, capas_ratesDc, afp0s_ratesDc = capa_and_initial_varaparam(
        targs | {'T': T_arr[np.argmax(capasT)]}, 'Dc', Dc_arr, redo=redo)
    tauc_arr = np.linspace(tauc/2, tauc*2, 40)
    capastauc, afp0stauc, capas_ratestauc, afp0s_ratestauc = capa_and_initial_varaparam(
        targs | {'T': T_arr[np.argmax(capasT)]}, 'tauc', tauc_arr, redo=redo)
    
    # Vary speed (joint scaling of Dc, rac, and T) and observe accuracy and capacity
    speed_arr = 10**np.linspace(-2, np.log10(10), 15)
    afp0s_speed, capas_speed, K11s = speed_accuracy_tradeoff(targs | {'T': T_arr[np.argmax(capasT)]}, speed_arr, redo=redo)

    # What is the optimal training time dependent on sparseness?
    # Input sparseness
    fc_arr = (5+np.arange(15))/M 
    capas = []
    T_sparse = np.linspace(10, 50, 40)
    T_opts = []
    for fc_ in fc_arr:
        capas.append(capa_and_initial_varaparam(targs|{'fc': fc_}, 'T', T_sparse, redo=redo)[0])
        T_opts.append(T_sparse[np.argmax(capas[-1])])
    capas = np.array(capas).T  # columns are fixed fc
    maxcapas = np.max(capas, axis=0)
    # Output sparseness
    fs_arr = 0.025 + 0.005*np.arange(15)
    capas_s = []
    T_sparse = np.linspace(10, 50, 40)
    T_opts_s = []
    for fs_ in fs_arr:
        capas_s.append(capa_and_initial_varaparam(targs|{'fs': fs_}, 'T', T_sparse, redo=redo)[0])
        T_opts_s.append(T_sparse[np.argmax(capas_s[-1])])

    # stop = time.time()
    # print('Time: ', stop-start)
    # exit()

    capas_s = np.array(capas_s).T  # columns are fixed fc
    maxcapas_s = np.max(capas_s, axis=0)

    min_inds = [np.min(np.arange(len(capas[:, i]))[capas[:, i]==maxcapas[i]])
                for i in range(len(fc_arr))]
    max_inds = [np.max(np.arange(len(capas[:, i]))[capas[:, i]==maxcapas[i]])
                for i in range(len(fc_arr))]
    min_inds_s = [np.min(np.arange(len(capas_s[:, i]))[capas_s[:, i]==maxcapas_s[i]])
                for i in range(len(fs_arr))]
    max_inds_s = [np.max(np.arange(len(capas_s[:, i]))[capas_s[:, i]==maxcapas_s[i]])
                for i in range(len(fs_arr))]

    # Set up figure
    fig = plt.figure(figsize=(3.375, 2.2))
    subfigs = fig.subfigures(2, 1)  #, width_ratios=[0.45, 0.55])
    ssfigs = subfigs[0].subfigures(1, 2)
    axs = dict(zip(['A'], [ssfigs[0].subplots(1, 1)]))
    axs = axs | dict(zip(['B1', 'B2'], ssfigs[1].subplots(2, 1, sharex=True)))
    axs = axs | dict(zip(['C', 'D'], subfigs[1].subplots(1, 2, sharey=True)))

    # Plot accuracy vs. capacity for different parameter variations
    axs['A'].plot(1-afp0sT, capasT, 'k', label=r'$T$')
    axs['A'].plot(1-afp0sDc, capasDc, 'b:', label=r'$\Delta_c$')
    axs['A'].plot(1-afp0stauc, capastauc, 'g--', label=r'$\tau_c$')
    axs['A'].plot(1-afp0s_speed, capas_speed, 'r-.', label=r'speed $\rho$')
    axs['A'].legend()
    axs['A'].set_xlabel(r'early accuracy $a_0$')
    axs['A'].set_ylabel('capacity')
    axs['A'].set_title('Accuracy vs. capacity')
    axs['A'].axvline(1-afp0sT[np.argmax(capasT)], ls='--', color='gray')

    # Show speed vs. capacity and initial accuracy
    axs['B1'].plot(speed_arr, capas_speed, 'k')
    axs['B1'].set_ylabel('capa')
    axs['B2'].plot(speed_arr, 1-afp0s_speed, 'k')
    axs['B2'].set_xlabel(r'speed $\rho$')
    axs['B2'].set_ylabel(r'$a_0$')
    axs['B2'].set_xscale('log')
    axs['B1'].set_title('Speed is detrimental')

    # Show T (gray-coded) vs. sparseness
    FC, TMG = np.meshgrid(fc_arr, T_sparse)
    vmax = np.max((np.max(capas), np.max(capas_s)))
    vmin = np.min((np.min(capas), np.min(capas_s)))
    pcm = axs['C'].pcolormesh(FC, TMG, capas, cmap='Greys', vmin=vmin, vmax=vmax)
    axs['C'].errorbar(fc_arr, 0.5*(T_sparse[min_inds]+T_sparse[max_inds]), 
                      yerr=0.5*(T_sparse[max_inds]-T_sparse[min_inds]), ls='', marker='.', color='red')
    axs['C'].set_xlabel(r'$f_c$')
    axs['C'].set_ylabel(r'$T$')
    axs['C'].set_title(r'Optimal $T$, vary $f_c$')

    FS, TMGs = np.meshgrid(fs_arr, T_sparse)
    pcm = axs['D'].pcolormesh(FS, TMGs, capas_s, cmap='Greys', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=axs['D'], label='capacity', aspect=10)
    axs['D'].errorbar(fs_arr, 0.5*(T_sparse[min_inds_s]+T_sparse[max_inds_s]), 
                      yerr=0.5*(T_sparse[max_inds_s]-T_sparse[min_inds_s]), ls='', marker='.', color='red')
    axs['D'].set_xlabel(r'$f_s$')
    axs['D'].set_title(r'Optimal $T$, vary $f_s$')

    keys = ['A', 'B1', 'C', 'D']
    labels = ['(a)', '(b)', '(c)', '(d)']
    for key, label in zip(keys, labels):
        if key in ['C', 'D']:
            axs[key].set_title(label, loc='left', x=-.12)
        else:
            axs[key].set_title(label, loc='left', x=-.2)

    fig.savefig('figs/revision_tradeoffs.pdf')

    plt.show()


def fig3_D1slope(redo=False):
    """
    Investigate the slope of D1 as a function of rate and CV.
    """
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle') 

    # Parameters
    nuc, vt, vr = .1, 1, 0
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20

    w_rough = 0.01*(np.arange(21)+1)

    w = 0.1  # weight at which rate and CV are computed
    epsilon = 1e-3  # for numerical derivative

    # Grid for mu and D which corrspond to a set of rate and CV
    size = 50
    mu_arr = np.linspace(-.5, 1.2, size)
    D_arr = np.linspace(.01, 0.3, size)

    name = 'nD1slope' + makename([nuc, vt, vr, Dc, rac, tauc, tauac, w_rough, size, mu_arr, D_arr])
    assert len(name) < 250
    if not redo and check(name):
        Slopes, Rates, CVs = load(name, 3)
    else:
        Slopes = np.zeros((size, size))
        Rates = np.zeros_like(Slopes)
        CVs = np.zeros_like(Slopes)
        for i, mu in enumerate(mu_arr):
            for j, D in enumerate(D_arr):
                D1, _, _ = drift_and_diffusion_coefficient(vr, vt, mu, 
                                D, nuc, Dc, rac, tauc, tauac, np.array([w, w+epsilon]))
                Slopes[i, j] = np.diff(D1)[0]/epsilon 

                # rate and CV for w=0.1
                mu_tot = mu + w*nuc
                D_tot = D + .5*w**2*nuc
                Rates[i, j] = siegert_rateD(mu_tot, D_tot, vr=vr, vt=vt)
                CVs[i, j] = siegert_cv_unitless(mu_tot, np.sqrt(2*D_tot), vr=vr, vt=vt)

    save(name, [Slopes, Rates, CVs])

    # Prepare figure
    fig, axs = plt.subplot_mosaic([['A', 'C'],
                                   ['B', 'C']], figsize=(3.375, 1.5))
    ax = axs['C']

    # C: Plot slope in gray-code in the rate-CV plane
    vmax = np.max(np.abs(1e3*Slopes))
    vmin = -vmax
    pcm = ax.pcolormesh(Rates, CVs, 1e3*Slopes, shading='nearest', cmap='PuOr_r', 
                        vmin=vmin, vmax=vmax)
    RatesOG, CVsOG, SlopesOG = np.copy(Rates), np.copy(CVs), np.copy(Slopes)
    ax.contour(Rates, CVs, Slopes, levels=[0], colors=['gray'], 
               alpha=.8, linestyles=[':'])
    ax.set_xlabel('rate')
    fig.colorbar(pcm, ax=ax, label=r'slope/$10^{-3}$')
    mu_ind, D_ind = 36, 3
    ax.plot([Rates[mu_ind, D_ind]], CVs[mu_ind, D_ind], 'rx')
    mu_ind2, D_ind2 = 35, 8
    ax.plot([Rates[mu_ind2, D_ind2]], CVs[mu_ind2, D_ind2], 'gx')
    ax.set_title('Slope transition')
    ax.set_xlim(left=-.05)
    ax.set_ylim(0.18, 0.92)

    # Robustness: is the slope transition also there is we vary input parameters?
    vary_w_and_nu = False  # tested, but too messy in figure
    if vary_w_and_nu:
        # Plot contour lines of varied w and nu cases
        nuc_vary = [0.05, 0.15]
        w_vary = [0.05, 0.15]

        for w_ in w_vary:
            name = 'nD1slope' + makename([nuc, vt, vr, Dc, rac, tauc, tauac, w_rough, size, mu_arr, D_arr]) + f'w{w_}'
            assert len(name) < 250
            if not redo and check(name):
                Slopes, Rates, CVs = load(name, 3)
            else:
                Slopes = np.zeros((size, size))
                Rates = np.zeros_like(Slopes)
                CVs = np.zeros_like(Slopes)
                for i, mu in enumerate(mu_arr):
                    print(i)
                    for j, D in enumerate(D_arr):
                        D1, _, _ = drift_and_diffusion_coefficient(vr, vt, mu, 
                                        D, nuc, Dc, rac, tauc, tauac, np.array([w_, w_+epsilon]))
                        Slopes[i, j] = np.diff(D1)[0]/epsilon 
                        mu_tot = mu + w_*nuc
                        D_tot = D + .5*w_**2*nuc
                        Rates[i, j] = siegert_rateD(mu_tot, D_tot, vr=vr, vt=vt)
                        CVs[i, j] = siegert_cv_unitless(mu_tot, np.sqrt(2*D_tot), vr=vr, vt=vt)
                print('done')
                save(name, [Slopes, Rates, CVs])
            vmax = np.max(np.abs(1e3*Slopes))
            vmin = -vmax
            ax.contour(Rates, CVs, Slopes, levels=[0], colors=['gray'], alpha=.8)
        for nuc_ in nuc_vary:
            name = 'nD1slope' + makename([nuc_, vt, vr, Dc, rac, tauc, tauac, w_rough, size, mu_arr, D_arr])
            assert len(name) < 250
            if not redo and check(name):
                Slopes, Rates, CVs = load(name, 3)
            else:
                Slopes = np.zeros((size, size))
                Rates = np.zeros_like(Slopes)
                CVs = np.zeros_like(Slopes)
                for i, mu in enumerate(mu_arr):
                    print(i)
                    for j, D in enumerate(D_arr):
                        D1, _, _ = drift_and_diffusion_coefficient(vr, vt, mu, 
                                        D, nuc_, Dc, rac, tauc, tauac, np.array([w, w+epsilon]))
                        Slopes[i, j] = np.diff(D1)[0]/epsilon
                        # rate and CV for w=0.1
                        mu_tot = mu + w*nuc_
                        D_tot = D + .5*w**2*nuc_
                        Rates[i, j] = siegert_rateD(mu_tot, D_tot, vr=vr, vt=vt)
                        CVs[i, j] = siegert_cv_unitless(mu_tot, np.sqrt(2*D_tot), vr=vr, vt=vt)
                print('done')
                save(name, [Slopes, Rates, CVs])
            vmax = np.max(np.abs(1e3*Slopes))
            vmin = -vmax
            ax.contour(Rates, CVs, Slopes, levels=[0], colors=['gray'], alpha=.8)

        ax.contour(RatesOG, CVsOG, SlopesOG, levels=[0], colors=['k'])

    # A: Show the drift coefficient for an example situation in the accelerating phase
    D1, _, _ = drift_and_diffusion_coefficient(vr, vt, mu_arr[mu_ind], 
                            D_arr[D_ind], nuc, Dc, rac, tauc, tauac, w_rough)
    # SIMULATION
    w_edges = 0.005 * np.arange(100)
    print('Red marker mu, D, rate, CV', mu_arr[mu_ind], D_arr[D_ind], Rates[mu_ind, D_ind], CVs[mu_ind, D_ind])
    args = {'nuc': nuc, 'wmin': 0, 'wmax': .31, 'mu': mu_arr[mu_ind], 'D': D_arr[D_ind], 'vt': vt, 'vr': vr,
        'Dc': Dc, 'rac': rac, 'tauc': tauc, 'tauac': tauac, 
        'dt': 1e-4, 'T': 100, 'Twarm': 50, 'boundary': 1}
    D1_num = km_coeffs(10000, args, w_edges, Dt=1e-2, redo=False)[0]
    num_dict = {'marker': 'o', 'c': 'darkorange', 'markerfacecolor': 'none', 'ls': '', 'marker': 'o'}
    axs['A'].plot(w_edges, D1_num*1e5, **num_dict)
    axs['A'].plot(w_rough, D1*1e5, 'r')
    axs['A'].axvline(w, color='gray', ls=':')
    axs['A'].set_ylabel(r'$D^{(1)}/10^{-5}$')
    axs['A'].set_title('Acceleration')
    axs['A'].set_xlim(0, .2)
    autoscale_y(axs['A'])
    axs['A'].set_xticks([])
    
    # B: Show the drift coefficient for an example situation in the decelerating phase
    D1, _, _ = drift_and_diffusion_coefficient(vr, vt, mu_arr[mu_ind2], 
                            D_arr[D_ind2], nuc, Dc, rac, tauc, tauac, w_rough)
    
    args = {'nuc': nuc, 'wmin': 0, 'wmax': .31, 'mu': mu_arr[mu_ind2], 'D': D_arr[D_ind2], 'vt': vt, 'vr': vr,
        'Dc': Dc, 'rac': rac, 'tauc': tauc, 'tauac': tauac, 
        'dt': 1e-4, 'T': 100, 'Twarm': 50, 'boundary': 1}
    print('Green marker mu, D, rate, CV', mu_arr[mu_ind2], D_arr[D_ind2], Rates[mu_ind2, D_ind2], CVs[mu_ind2, D_ind2])
    D1_num = km_coeffs(10000, args, w_edges, Dt=1e-2, redo=False)[0]
    num_dict = {'marker': 'o', 'c': 'lightgreen', 'markerfacecolor': 'none', 'ls': '', 'marker': 'o'}

    axs['B'].plot(w_edges, D1_num*1e5, **num_dict)
    axs['B'].plot(w_rough, D1*1e5, 'g')
    axs['B'].axvline(w, color='gray', ls=':')
    axs['B'].set_xlabel(r'$w$')
    axs['B'].set_ylabel(r'$D^{(1)}/10^{-5}$')
    axs['B'].set_title('Deceleration')
    axs['B'].set_xlim(0, .2)
    autoscale_y(axs['B'])

    axs['A'].set_title('(a)', loc='left', x=-.2)
    axs['B'].set_title('(b)', loc='left', x=-.2)
    axs['C'].set_title('(c)', loc='left', x=-.2)
    ax.set_ylabel(r'$C_v$')

    fig.savefig('figs/revision_accel_deaccel.pdf')
    plt.show()


def fig7_TrainingWithoutHomeostasis():
    """
    The referee asked what happens if we train without homeostasis.
    1) The dynamics does not explode because Bi Poo 1998 is stable.
    2) But learning as we like to think of it (coincidental activity 
       potentiates) does not happen.

    These points are demonstrated here by calling the training wrapper without
    homeostasis.
    """
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle') 

    vr, vt, muE, muI, DE, DI = 0, 1, 0, .5, .1, .05  # neuron
    NE, NI, CE, CI, J, g, h = 4000, 1000, 200, 50, 0.01, 5., 2.  # network
    M, fc, fs, nu, m0, nus, Js = 200, .05, .1, .5, .05, 64, .0125  # input
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20  # STDP
    dt, T, Twarm = 1e-4, 50, 10  # simulation
    nu0, nu1 = 0, 1  #0.8

    P = 400
    assert nu0 == 0
    targs = dict(zip(  # parameters relevant for theory
        ['vr', 'vt', 'muE', 'muI', 'DE', 'DI', 'CE', 'CI', 'J', 'g', 'h', 
        'm0', 'M', 'nu', 'nus', 'fc', 'fs', 'Js', 'Dc', 'rac', 'tauc', 
        'tauac', 'T'],
        [vr, vt, muE, muI, DE, DI, CE, CI, J, g, h, m0, M, nu1, nus, fc, fs,
        Js, Dc, rac, tauc, tauac, T]))
    k = 200
    sargs = targs | dict(zip(  # add simulation parameters
        ['NE', 'NI', 'dt', 'Twarm', 'k', 'P', 'V0', 'Trecall'],
        [NE, NI, dt, Twarm, k, P, 0, np.inf]))

    
    Mall_1, Vall_1, MTs_1, MNs_1, GTs_1, GNs_1, DeltaTs_1, DeltaNs_1, VTs_1, VNs_1, V0s_1, rTs_1, rNs_1, rIs_1, afp_1, meta_1 = wrap_training_return_confusion_no_homeostasis(
        sargs, 1, redo=False)
    
    DeltaNs_1 += m0 - np.concatenate(([m0], Mall_1[:-1]))
    DeltaTs_1 += m0 - np.concatenate(([m0], Mall_1[:-1]))
    
    # COMPUTE DRIFT COEFFICIENT
    mu_c = M*Mall_1[-1]*fc*nu1
    D_c = (1/2)*M*(Mall_1[-1]**2+Vall_1[-1])*fc*nu1
    mus, Ds = nus*Js, 0.5*Js**2*nus 

    r1, r0, rI, mu_netE, D_netE, mu_netI, D_netI = brunel_TNI_theory(
        vr, vt, CE, CI, J, g, h, muE+mu_c+mus, muE+mu_c, muI, DE+D_c+Ds, DE+D_c, DI, fs)

    mu1 = muE + mu_c + mus + mu_netE
    mu0 = muE + mu_c + mu_netE
    D1 = DE + D_c + Ds + D_netE
    D0 = DE + D_c + D_netE
    w_arr = 0.01*(np.arange(21)+1)
    D1T, D2T, rateT = drift_and_diffusion_coefficient(vr, vt, mu1, D1, nu1, Dc, rac, tauc, tauac, w_arr, 'all')
    D1N, D2N, rateN = drift_and_diffusion_coefficient(vr, vt, mu0, D0, nu1, Dc, rac, tauc, tauac, w_arr, 'all')  # still nu1 (this is 0 <- 1)
    w_dense = np.linspace(w_arr[0], w_arr[-1], 10000)
    intersectN = w_dense[np.argmin(np.abs(np.interp(w_dense, w_arr, D1N)))]
    intersectT = w_dense[np.argmin(np.abs(np.interp(w_dense, w_arr, D1T)))]
    
    fig, axs = plt.subplots(2, 2, figsize=(3.375, 2.2))
    axs[0, 0].plot(np.arange(P), Mall_1, 'k', label=r'$\langle w \rangle$')
    axs[0, 0].axhline((1-fs)*intersectN+fs*intersectT, c='gray', ls='--')
    axs[0, 0].plot(np.arange(P), np.sqrt(np.asarray(Vall_1)), 'orange', 
                    label=r'$\sqrt{\langle\langle w ^2 \rangle\rangle}$')
    axs[0, 0].set_ylabel(r'$ w $')
    axs[0, 0].set_title('Convergence of weights')
    axs[0, 0].set_xlabel(r'$k$')
    axs[0, 0].legend()

    axs[0, 1].plot(w_arr, D1T/1e-3, color='#006d2c', label=r'$1\to 1$')
    axs[0, 1].plot(w_arr, D1N/1e-3, color='#bd0026', label=r'$1\to 0$')
    axs[0, 1].set_xlabel(r'$w$')
    axs[0, 1].set_ylabel(r'$D^{(1)}/10^{-3}$')
    axs[0, 1].axvline(Mall_1[-1], color='gray', ls='--')
    axs[0, 1].axhline(0, color='gray', lw=.5, zorder=0)
    axs[0, 1].set_title('Drift coefficient')
    axs[0, 1].legend()


    axs[1, 0].plot(np.arange(P), DeltaTs_1, color='#66c2a4', label=r'$1\to 1$')
    axs[1, 0].plot(np.arange(P), DeltaNs_1, color='#fd8d3c', label=r'$1\to 0$')
    axs[1, 0].set_ylabel(r'$\Delta w$')
    axs[1, 0].set_title('Population-wise weight shift')
    axs[1, 0].set_xlabel(r'$k$')
    axs[1, 0].legend()

    axs[1, 1].plot(np.arange(P), 1-afp_1, 'k')
    axs[1, 1].set_xticks([0, 200, 400], labels=[0, r'$k^\ast=200$', 400])
    axs[1, 1].set_ylabel(r'$a_k$')
    axs[1, 1].set_xlabel(r'$k$')
    axs[1, 1].set_title(r'Testing $\bm{q}_{k^\ast}\rightarrow\bm{p}_{k^\ast}$')


    axs[0, 0].set_title('(a)', loc='left', x=-.21)
    axs[0, 1].set_title('(b)', loc='left', x=-.21)
    axs[1, 0].set_title('(c)', loc='left', x=-.32)
    axs[1, 1].set_title('(d)', loc='left', x=-.15)
    
    fig.savefig('figs/revision_PRXlife1_training_without_homeostasis.pdf')

    print(meta_1)

    plt.show()


def fig8_LowNoiseDrift():
    """
    Test the theory for the drift coefficient of a single synapse at
    small values of background noise intensity D.
    As demonstrated here, the theory breaks down for D->0, but slowly.
    """
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle') 

    fig, axs = plt.subplots(1, 5, figsize=(2*3.575, 1.2), sharex=True)
    num_dict = {'marker': 'o', 'c': 'darkorange', 'markerfacecolor': 'none', 'ls': '', 'marker': 'o'}

    # PARAMETERS
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20
    nuc, vt, vr = .1, 1, 0

    # Different noise intensities
    D_arr = [0, 1e-3, 1e-2, 1e-1, 1]

    for i in range(len(D_arr)):
        # Determine mu such that rate = 0.1 at w=0.1
        D = D_arr[i]
        D_tot = D + .5*0.1**2*nuc
        rate = .1
        def zero(mu_):
            mu_tot = mu_ + 0.1*nuc
            return rate-siegert_rateD(mu_tot, D_tot, vr, vt)
        mu = brentq(zero, -3, 1.5)
        
        # EVALUATE THEORY
        w_rough = 0.01*(np.arange(21)+1)
        D1D2 = partial(drift_and_diffusion_coefficient, vr, vt, mu, 
                        D, nuc, Dc, rac, tauc, tauac, w_rough)
        D1_rate, D2_rate, r0_rate = D1D2(contributions='rates')
        D1_mean, D2_mean, r0_mean = D1D2(contributions='mean_response')
        D1_rate_and_mean = D1_rate + D1_mean
        D1_full, D2_full, r0_full = D1D2(contributions='all')

        # SIMULATION
        wmin, wmax = 0., 0.251  # Sample ICs for the synaptic weights uniormly in [wmin, wmax]
        dt, T, Twarm = 1e-4, 100, 50  # Simulation parameters
        w_edges = 0.005 * np.arange(44)  # Histogram edges for KM coeff estimation
        args = {'nuc': nuc, 'wmin': wmin, 'wmax': wmax, 'mu': mu, 'D': D, 'vt': vt, 'vr': vr,
            'Dc': Dc, 'rac': rac, 'tauc': tauc, 'tauac': tauac, 
            'dt': dt, 'T': T, 'Twarm': Twarm, 'boundary': 1}
        trials = 100
        D1, D2, D3, D4 = km_coeffs(trials, args, w_edges, Dt=1e-2, redo=False)

        axs[i].plot(w_rough, D1_rate_and_mean, color='purple')
        axs[i].set_xlabel(r'$w$')
        axs[i].plot(w_edges[1:-1], D1[1:-1], **num_dict) # outer edges in digitization unreliable
        axs[i].plot(w_rough, D1_full, 'k')

    axs[0].set_ylabel(r'$D^{(1)}$')

    axs[0].set_title(r'$\,\,\,\,D=0\,\,\,\,$')
    axs[1].set_title(r'$D=10^{-3}$')
    axs[2].set_title(r'$D=10^{-2}$')
    axs[3].set_title(r'$D=10^{-1}$')
    axs[4].set_title(r'$\,\,\,\,D=1\,\,\,\,$')

    axs[0].set_title('(a)', loc='left')
    axs[1].set_title('(b)', loc='left')
    axs[2].set_title('(c)', loc='left')
    axs[3].set_title('(d)', loc='left')
    axs[4].set_title('(e)', loc='left')

    # fig.savefig('figs/single_synapse_low_noise.pdf')
    plt.show()


if __name__ == '__main__':
    # fig7_TrainingWithoutHomeostasis()
    fig8_LowNoiseDrift()
    # fig6_tradeoffs()
    # fig3_D1slope()