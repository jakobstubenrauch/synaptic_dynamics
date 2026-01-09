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
Produce Figure 5 of the manuscript.
Evolution of populations of synaptic weights during training a sequence
of associations and memory properties.
"""
import numpy as np
from network_training_scenario_theory_and_wrappers import fp_traces_and_afp, wrap_training_return_confusion
from utilities.tools import check, load, save
import time


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('syndyn.mplstyle')

    fig, axs = plt.subplot_mosaic([['A', 'B'],
                                   ['C', 'C'],
                                   ['D', 'E']], figsize=(3.375, 1.5*2.085864712))

    # Parameters
    vr, vt, muE, muI, DE, DI = 0, 1, 0, .5, .1, .05  # neuron
    NE, NI, CE, CI, J, g, h = 4000, 1000, 200, 50, 0.01, 5., 2.  # network
    M, fc, fs, nu, m0, nus, Js = 200, .05, .1, .5, .05, 64, .0125  # input
    Dc, rac, tauc, tauac = 0.002, -0.008, 16.8/20, 33.7/20  # STDP
    dt, T, Twarm = 1e-4, 50, 10  # simulation
    nu0, nu1 = 0, 1  #0.8

    P = 300
    start = time.time()
    
    scV, Kk, Gk, afp_ana = fp_traces_and_afp(
        vr, vt, muE, DE, muI, DI,  # neuron
        CE, CI, J, g, h,  # network
        m0, M, nu0, nu1, fc, fs, Js, nus,  # input
        Dc, rac, tauc, tauac, T,  # training
        P)  # number of associations to consider
    rscV, rKk, rGk, rafp_ana = fp_traces_and_afp(
        vr, vt, muE, DE, muI, DI,  # neuron
        CE, CI, J, g, h,  # network
        m0, M, nu0, nu1, fc, fs, Js, nus,  # input
        Dc, rac, tauc, tauac, T,  # training
        P, contributions=['rates'])  # number of associations to consider
    print(f'Eval. theory took {time.time()-start:.1f}s')
    assert nu0 == 0
    targs = dict(zip(  # parameters relevant for theory
        ['vr', 'vt', 'muE', 'muI', 'DE', 'DI', 'CE', 'CI', 'J', 'g', 'h', 
        'm0', 'M', 'nu', 'nus', 'fc', 'fs', 'Js', 'Dc', 'rac', 'tauc', 
        'tauac', 'T'],
        [vr, vt, muE, muI, DE, DI, CE, CI, J, g, h, m0, M, nu1, nus, fc, fs,
        Js, Dc, rac, tauc, tauac, T]))
    k = 5
    sargs = targs | dict(zip(  # add simulation parameters
        ['NE', 'NI', 'dt', 'Twarm', 'k', 'P', 'V0', 'Trecall'],
        [NE, NI, dt, Twarm, k, P, scV, np.inf]))
    
    MTs_1, MNs_1, GTs_1, GNs_1, DeltaTs_1, DeltaNs_1, VTs_1, VNs_1, V0s_1, rTs_1, rNs_1, rIs_1, afp_1, meta_1 = wrap_training_return_confusion(
        sargs, 1, redo=False)
    
    # print(f'Simulation took {meta_1[1]/3600:.1f}h on {meta_1[0]} cpus for single realization')
    # print(meta_1)
    # print(np.shape(MTs_1))
    # exit()

    MTs, MNs, GTs, GNs, DeltaTs, DeltaNs, VTs, VNs, V0s, rTs, rNs, rIs, afp = wrap_training_return_confusion(
        sargs, 100, redo=False)
    

    if check('capa_vary_fc'):
        fc_dense, capa_vary_fc, capa_vary_fc_rates = load('capa_vary_fc', 3)
    else:
        fc_dense = 0.04 + 0.01 * np.arange(16)
        capa_vary_fc = []
        capa_vary_fc_rates = []
        for fc_ in fc_dense:
            print(fc_)
            _, _, _, afp_ana_ = fp_traces_and_afp(
                vr, vt, muE, DE, muI, DI,  # neuron
                CE, CI, J, g, h,  # network
                m0, M, nu0, nu1, fc_, fs, Js, nus,  # input
                Dc, rac, tauc, tauac, T,  # training
                P)  # number of associations to consider
            capa_vary_fc.append(np.argmin(np.abs(np.asarray(afp_ana_) - 0.5)))
            _, _, _, afp_ana_ = fp_traces_and_afp(
                vr, vt, muE, DE, muI, DI,  # neuron
                CE, CI, J, g, h,  # network
                m0, M, nu0, nu1, fc_, fs, Js, nus,  # input
                Dc, rac, tauc, tauac, T,  # training
                P, contributions=['rates'])  # number of associations to consider
            capa_vary_fc_rates.append(np.argmin(np.abs(np.asarray(afp_ana_) - 0.5)))

        save('capa_vary_fc', [fc_dense, capa_vary_fc, capa_vary_fc_rates])

    if check('capa_vary_M'):
        M_dense, capa_vary_M, capa_vary_M_rates = load('capa_vary_M', 3)
    else:
        M_dense = 100 + 50*np.arange(16)
        capa_vary_M = []
        capa_vary_M_rates = []
        for M_ in M_dense:
            fc_ = fc*M/M_
            _, _, _, afp_ana_ = fp_traces_and_afp(
                vr, vt, muE, DE, muI, DI,  # neuron
                CE, CI, J, g, h,  # network
                m0, M_, nu0, nu1, fc_, fs, Js, nus,  # input
                Dc, rac, tauc, tauac, T,  # training
                P)  # number of associations to consider
            capa_vary_M.append(np.argmin(np.abs(np.asarray(afp_ana_) - 0.5)))
            _, _, _, afp_ana_ = fp_traces_and_afp(
                vr, vt, muE, DE, muI, DI,  # neuron
                CE, CI, J, g, h,  # network
                m0, M_, nu0, nu1, fc_, fs, Js, nus,  # input
                Dc, rac, tauc, tauac, T,  # training
                P, contributions=['rates'])  # number of associations to consider
            capa_vary_M_rates.append(np.argmin(np.abs(np.asarray(afp_ana_) - 0.5)))
        save('capa_vary_M', [M_dense, capa_vary_M, capa_vary_M_rates])

    numdict = {'ls': '', 'marker': 'o', 'markerfacecolor': 'none'}
    skip = 10
    axs['A'].plot(np.arange(P)[k:][::skip]-k, MTs[k:][::skip], color='#006d2c', **numdict)
    axs['A'].plot(np.arange(P)[k:][::skip]-k, MNs[k:][::skip], color='#bd0026', **numdict)
    axs['A'].plot(np.arange(P)[k:][::skip]-k, MTs_1[k:][::skip], color='#66c2a4', **numdict)
    axs['A'].plot(np.arange(P)[k:][::skip]-k, MNs_1[k:][::skip], color='#fd8d3c', **numdict)
    axs['A'].plot(np.arange(P), Kk[:, 1, 1], 'k')
    axs['A'].plot(np.arange(P), Kk[:, 0, 1], 'k')
    axs['A'].axhline(m0, color='gray')
    axs['A'].set_ylabel(r'$K^{(k)}_{a1}$')
    axs['A'].set_xlabel(r'$k$')

    axs['A'].set_title('Trace degradation')

    scale = 1e-4
    axs['B'].plot(np.arange(P)[k:][::skip]-k, GTs[k:][::skip]/scale, color='#006d2c', **numdict)
    axs['B'].plot(np.arange(P)[k:][::skip]-k, GNs[k:][::skip]/scale, color='#bd0026', **numdict)
    axs['B'].plot(np.arange(P)[k:][::skip]-k, GTs_1[k:][::skip]/scale, color='#66c2a4', **numdict)
    axs['B'].plot(np.arange(P)[k:][::skip]-k, GNs_1[k:][::skip]/scale, color='#fd8d3c', **numdict)
    axs['B'].plot(np.arange(P), Gk[:, 1, 1]/scale, 'k')
    axs['B'].plot(np.arange(P), Gk[:, 0, 1]/scale, 'k')
    axs['B'].axhline(scV/scale, color='gray')
    axs['B'].set_title('Trace variance')
    axs['B'].set_ylabel(r'$G^{(k)}_{a1}/10^{-4}$')
    axs['B'].set_xlabel(r'$k$')

    skip = 5
    axs['C'].plot(np.arange(P)[k:][::skip]-k, 1-afp[k:][::skip], color='#810f7c', **numdict)
    axs['C'].plot(np.arange(P)[k:][::skip]-k, 1-afp_1[k:][::skip], color='#8c96c6', **numdict)
    axs['C'].plot(np.arange(P), 1-afp_ana, 'k')
    axs['C'].plot(np.arange(P), 1-rafp_ana, 'k:')
    axs['C'].plot([0, np.argmin(np.abs(afp_ana-0.5))], [.5, .5], color='#253494', lw=3)
    axs['C'].set_title('Fraction of correctly activated neurons')
    axs['C'].set_ylabel(r'$a_k$')
    # axs['C'].set_ylabel(r'$a^\mathrm{fp}_k$')
    axs['C'].set_xlabel(r'$k$')

    axs['D'].plot(fc_dense, capa_vary_fc, c='#253494')
    axs['D'].plot(fc_dense, capa_vary_fc_rates, c='#253494', ls=':')
    axs['D'].plot([fc], [np.argmin(np.abs(afp[k:]-0.5))], c='k', **numdict)
    axs['D'].set_ylabel(r'capacity')
    axs['D'].set_title(r'Vary $f_c$, fix $M=200$')
    axs['D'].set_xlabel(r'Input activation ratio $f_c$')

    axs['E'].plot(M_dense[:10], capa_vary_M[:10], c='#253494')
    axs['E'].plot(M_dense[:10], capa_vary_M_rates[:10], c='#253494', ls=':')
    axs['E'].plot([M], [np.argmin(np.abs(afp[k:]-0.5))], c='k', **numdict)
    axs['E'].set_ylabel(r'capacity')
    axs['E'].set_title(r'Vary $M$, fix $f_c M=10$')
    axs['E'].set_xlabel(r'Input width $M$')

    axs['A'].set_xlim(0, 300)
    axs['B'].set_xlim(0, 300)
    axs['C'].set_xlim(0, 300)
    axs['C'].set_ylim(0, 1)
    axs['C'].set_yticks([0, 0.5, 1])
    axs['D'].set_xlim(fc_dense[0], fc_dense[-1])
    axs['E'].set_xlim(M_dense[0], 520)
    axs['E'].set_ylim(top=capa_vary_M[9]-30)
    axs['E'].set_xticks([100, 300, 500])


    keys = ['A', 'B', 'C', 'D', 'E']
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for key, label in zip(keys, labels):
        if key == 'C':
            axs[key].set_title(label, loc='left', x=-.125)
        else:
            axs[key].set_title(label, loc='left', x=-.3)

    fig.savefig('figs/v2_Figure_Forgetting.pdf')
    plt.show()