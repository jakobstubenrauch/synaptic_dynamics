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
Produce Figure 2 of the manuscript.
Present sample trajectories of single synapses and compute their
Kramers-Moyal coefficients.
"""
import numpy as np
from math import factorial
import multiprocess as mp
from functools import partial
from cy_code.bipoo_one_synapse_gaussian_ic import integrate as integrate_osgic
from cy_code.bipoo_one_synapse_gaussian_ic import integrate_uniform_ic
from cy_code.integrate_langevin_interp import integrate as integrate_langevin
from single_synapse_theory import drift_and_diffusion_coefficient, cumulated_diffusion
from utilities.tools import check, load, save, makename, makehash
from response_functions import siegert_cv_unitless


def core_km_coeffs(seed, args, edges, Dt=0):
    """
    Compute the first four Kramers-Moyal coefficients from a single trajectory
    with parameters in args and initial condition sampled uniformly in 
    [edges[0], edges[-1]].
    To this end, the weight is digitized according to edges.
    The Kramers-Moyal coefficients are computed as scaled moments of the 
    stochastic increments across time Dt.
    """
    # integrate the synaptic weight
    w, rate = integrate_uniform_ic(seed, **args)
    if Dt != 0:
        w = w[::int(np.round(Dt/args['dt']))]
    Dw = np.diff(w)  # stochastic increment
    w = w[:-1]  # previous weight (used as condition)

    # identify bins of weights
    bin_inds = np.digitize(w, edges, right=True)  
    # right=True: if edges[0]=0 and w[j]=0 then bin_inds[j]=0

    # number of samples per bin (needed for averaging)
    num_samples = np.array([len(Dw[bin_inds==i]) for i in range(len(edges))])

    # unscaled moments (i.e., simply sums), to be scaled by number of samples later
    S1 = np.array([np.sum(Dw[bin_inds==i]) for i in range(len(edges))])
    S2 = np.array([np.sum(Dw[bin_inds==i]**2) for i in range(len(edges))])
    S3 = np.array([np.sum(Dw[bin_inds==i]**3) for i in range(len(edges))])
    S4 = np.array([np.sum(Dw[bin_inds==i]**4) for i in range(len(edges))])
    return num_samples, S1, S2, S3, S4


def km_coeffs(trials, args, edges, Dt=0, redo=False):
    """
    Wrapper of the Kramers-Moyal coefficient computation over multiple trials.
    The results are cached on disk.
    """
    # check if already computed
    name = f'ukm_coeffs_{trials}' + makename(args) + makehash(edges)
    assert Dt >= args['dt'] or Dt == 0, "Invalid Dt"
    if Dt > args['dt']:
        name += f'Dt{Dt:.2e}'
    print(name, check(name))
    if not redo and check(name):
        return load(name, 4)

    # compute the unscaled moments in parallel
    with mp.Pool(mp.cpu_count()-2) as pool:
        workers = [pool.apply_async(core_km_coeffs, (),
            {'seed': seed, 'args': args, 'edges': edges, 'Dt': Dt}) for seed in range(trials)]
        res = np.array([wrk.get() for wrk in workers])

    # aggregate results and scale to get Kramers-Moyal coefficients
    t_num_smpls = np.sum(res[:, 0, :], axis=0)
    if Dt == 0:
        D1 = np.sum(res[:, 1, :], axis=0) / t_num_smpls / args['dt'] / factorial(1)
        D2 = np.sum(res[:, 2, :], axis=0) / t_num_smpls / args['dt'] / factorial(2)
        D3 = np.sum(res[:, 3, :], axis=0) / t_num_smpls / args['dt'] / factorial(3)
        D4 = np.sum(res[:, 4, :], axis=0) / t_num_smpls / args['dt'] / factorial(4)
    else:
        D1 = np.sum(res[:, 1, :], axis=0) / t_num_smpls / Dt / factorial(1)
        D2 = np.sum(res[:, 2, :], axis=0) / t_num_smpls / Dt / factorial(2)
        D3 = np.sum(res[:, 3, :], axis=0) / t_num_smpls / Dt / factorial(3)
        D4 = np.sum(res[:, 4, :], axis=0) / t_num_smpls / Dt / factorial(4)
    save(name, [D1, D2, D3, D4])
    return D1, D2, D3, D4


def FigSingleSynapse():
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle') 

    # Create figure
    fig = plt.figure(figsize=(2*3.375, 2))
    sfigs = fig.subfigures(1, 3, width_ratios=[1, 1.8, 1.8])
    axs = np.concatenate((
        sfigs[0].subplots(2, 1, sharex=True, sharey=True).flat,
        sfigs[1].subplots(2, 2, sharex=True).flat,
        sfigs[2].subplots(2, 2, sharex=True).flat
    ))
    num_dict = {'marker': 'o', 'c': 'darkorange', 'markerfacecolor': 'none', 'ls': '', 'marker': 'o'}

    # A1: SAMPLE TRAJECTORIES of the true jump process
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20
    nuc, m0, V0, mu, D, vt, vr = .1, 0.1, 0.001**2, .6, .2, 1, 0
    dt, T, Twarm = 1e-4, 100, 4
    t = dt * np.arange(int(np.round(T/dt)))
    args = {'nuc': nuc, 'm0': m0, 'V0': V0, 'mu': mu, 'D': D, 'vt': vt, 'vr': vr,
        'Dc': Dc, 'rac': rac, 'tauc': tauc, 'tauac': tauac, 
        'dt': dt, 'T': T, 'Twarm': Twarm, 'boundary': 1}
    print('call')
    for seed in 2*np.arange(10):
        warr, rate = integrate_osgic(seed, **args)
        axs[0].plot(t, warr, c='k', alpha=.8)

    axs[0].set_title('Samples of Eq. (1)')
    axs[0].set_ylabel(r'$w$')
    axs[0].set_xlim(0, T)
    axs[0].set_ylim(0.096, 0.108)


    # A2: plot samples of Langevin equation
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20
    nuc, m0, V0, mu, D, vt, vr = .1, 0.1, 0.001**2, .6, .2, 1, 0
    dt, T, Twarm = 1e-4, 100, 4
    t = dt * np.arange(int(np.round(T/dt)))

    w_arr = 0.001*(1+np.arange(350))
    D1, D2, _ = drift_and_diffusion_coefficient(vr, vt, mu, 
        D, nuc, Dc, rac, tauc, tauac, w_arr, contributions='all')
    for seed in 50*np.arange(10):
        print(seed)
        warr = integrate_langevin(dt, T, m0, V0, w_arr, D1, D2, seed)
        if np.max(warr) <= 0.108:
            axs[1].plot(t, warr, c='k', alpha=.8)
    axs[1].set_title('Samples of Eq. (8)')
    axs[1].set_xlabel(r'time / $\tau_m$')
    axs[1].set_ylabel(r'$w$')
    axs[1].set_xlim(0, T)
    axs[1].set_ylim(0.096, 0.108)

    # COMPUTE AND PLOT KRAMERS-MOYAL COEFFICIENTS

    # parameters
    nuc, vt, vr = .1, 1, 0
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20

    # different parameter combinations
    D_arr = [.1, .2]
    mu_arr = [.0, .9, 1.5]

    idx = np.array([[2, 3], [4, 5]])
    scales = np.array([[1e6, 1e5], [1e5, 1e4]])
    scales2 = np.array([[1e8, 1e8], [1e7, 1e7]])

    # compute KM coefficients for 2 times 2 combinations of mu and D
    for i in range(2):
        for j in range(2):
            mu = mu_arr[i]
            D = D_arr[j]
            scale = scales[i, j]

            w_rough = 0.01*(np.arange(31))

            # Evaluate theory
            D1D2 = partial(drift_and_diffusion_coefficient, vr, vt, mu, 
                           D, nuc, Dc, rac, tauc, tauac, w_rough)

            D1_rate, D2_rate, r0_rate = D1D2(contributions='rates')
            D1_mean, D2_mean, r0_mean = D1D2(contributions='mean_response')
            D1_noise, D2_noise, r0_noise = D1D2(contributions='noise_intensity_response')
            D1_full, D2_full, r0_full = D1D2(contributions='all')

            mu_tot = mu + 0.1*nuc
            D_tot = D + .5*0.1**2*nuc
            CV_01 = siegert_cv_unitless(mu_tot, np.sqrt(2*D_tot), vr=vr, vt=vt)

            r0_01 = r0_full[np.argmin(np.abs(w_rough-0.1))]

            # SIMULATION
            wmin = 0.
            wmax = 0.31
            dt, T, Twarm = 1e-4, 100, 50
            w_edges = 0.005 * np.arange(100)
            args = {'nuc': nuc, 'wmin': wmin, 'wmax': wmax, 'mu': mu, 'D': D, 'vt': vt, 'vr': vr,
                'Dc': Dc, 'rac': rac, 'tauc': tauc, 'tauac': tauac, 
                'dt': dt, 'T': T, 'Twarm': Twarm, 'boundary': 1}
            trials = 100000
            # trials = 100000
            D1, D2, D3, D4 = km_coeffs(trials, args, w_edges, Dt=1e-2, redo=False)

            # Plot theory and simulation
            axs[idx[i, j]].plot([-1], [D1_rate[2]*scale], ls='', 
                                label=r'$r\approx$ '+f'{r0_01:.2f}\n' +
                                r'$C_v\approx$ '+f'{CV_01:.2f}')

            axs[idx[i, j]].plot(w_rough, D1_rate*scale, 'gray')
            axs[idx[i, j]].plot(w_rough, D1_mean*scale, color='lightgreen')
            axs[idx[i, j]].plot(w_rough, D1_noise*scale, color='blue')
            # axs[idx[i, j]].axhline(0, color='k', ls='--', zorder=0)
            axs[idx[i, j]].set_xlabel(r'$w$')
            axs[idx[i, j]].plot(w_edges, D1*scale, **num_dict)
            axs[idx[i, j]].plot(w_rough, D1_full*scale, 'k')
            axs[idx[i, j]].set_xlim(0, 0.251)
            axs[idx[i, j]].legend(loc='upper center')

            scale2 = scales2[i, j]
            axs[idx[i, j]+4].set_xlabel(r'$w$')
            axs[idx[i, j]+4].set_xlim(0, 0.251)
            # T, trials, Dt = 1000, 10000, 10
            T, trials, Dt = 1000, 10000, 10
            CD = [cumulated_diffusion(vr, vt, mu, D, nuc, Dc, rac, tauc, tauac, m0, Dt) for m0 in w_rough]
            D1_slo, D2_slo, D3, D4 = km_coeffs(trials, args, w_edges, Dt=Dt, redo=False)
            axs[idx[i, j]+4].plot(w_edges, D2_slo*scale2, **num_dict)
            axs[idx[i, j]+4].plot(w_rough, np.asarray(CD)*scale2, 'k')

    axs[idx[0, 0]].set_ylim(-2.5e-1, 5)
    axs[idx[0, 1]].set_ylim(bottom=-1e-1)
    axs[idx[1, 0]].set_ylim(bottom=-4)
    axs[idx[1, 1]].set_ylim(bottom=-7e-1)

    axs[idx[0, 0]+4].set_ylim(-1e-1, 1.1)
    axs[idx[0, 1]+4].set_ylim(-.1, 4.5)
    axs[idx[1, 0]+4].set_ylim(-.1, 1.5)
    axs[idx[1, 1]+4].set_ylim(-.1, 2.2)


    axs[6].set_ylabel(r'$D^{(2)}/10^{-7}$')
    axs[7].set_ylabel(r'$D^{(2)}/10^{-7}$')
    axs[7].set_xlabel(r'$w$')

    axs[2].set_ylabel(r'$\mu=0$'+'\n'+r'$D^{(1)}/10^{-6}$')
    axs[4].set_ylabel(r'$\mu=0.9$'+'\n'+r'$D^{(1)}/10^{-5}$')
    axs[3].set_ylabel(r'$D^{(1)}/10^{-5}$')
    axs[5].set_ylabel(r'$D^{(1)}/10^{-4}$')

    axs[2+4].set_ylabel(r'$\mu=0$'+'\n'+r'$D^{(2)}/10^{-8}$')
    axs[4+4].set_ylabel(r'$\mu=0.9$'+'\n'+r'$D^{(2)}/10^{-7}$')
    axs[3+4].set_ylabel(r'$D^{(2)}/10^{-8}$')
    axs[5+4].set_ylabel(r'$D^{(2)}/10^{-7}$')

    axs[2].set_title(r'$D=0.1$')
    axs[3].set_title(r'$D=0.2$')
    axs[6].set_title(r'$D=0.1$')
    axs[7].set_title(r'$D=0.2$')
    sfigs[1].suptitle(r'Drift coefficient for different $\mu$, $D$')
    sfigs[2].suptitle(r'Diffusion coefficient for different $\mu$, $D$')

    fig.savefig('figs/v2_Figure_SINGLEsynapse_morediff.pdf')
    plt.show()


if __name__ == '__main__':
    FigSingleSynapse()