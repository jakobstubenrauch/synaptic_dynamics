from scipy.special import pbdv, roots_legendre, roots_laguerre, erfcx, dawsn
import numpy as np
import multiprocess as mp


def siegert_rateD(mu, D, vr=0, vt=1, Nquad=100):
    """
    Firing rate of an LIF neuron with mean mu and noise intensity D = sigma^2/2
    Follows the method in Layer et al. 2016 Front. Comput. Neurosci. "NNMT"
    """
    args = np.vstack(np.broadcast_arrays(mu, D, vr, vt, Nquad)).T
    if len(args) > 1:
        return siegert_rateD_vect(args)
    sigma = np.sqrt(2*D)
    u, w = roots_legendre(Nquad)

    vrt = (vr - mu) / sigma
    vtt = (vt - mu) / sigma

    if not vrt <= vtt: print(sigma, vrt, vtt, mu)

    try:
        assert vrt <= vtt
    except AssertionError:
        print('Assertion Error, vrt =', vrt, ', vtt =', vtt)
    if vrt == vtt:
        return np.inf

    if vrt > 0:
        # print('inh')
        # Strong inhibition case
        I = (vtt-vrt)/2 * np.sum(w * erfcx((vtt-vrt)/2*u + (vtt+vrt)/2))
        emvt2 = np.exp(-vtt**2)
        evr2 = np.exp(vrt**2)
        if evr2 == np.inf:
            return 0
        b = 2*dawsn(vtt) - 2*emvt2*evr2*dawsn(vrt) - emvt2*I
        return emvt2/(np.sqrt(np.pi) * b)
    elif vtt < 0:
        # print('exc')
        # Strong excitation case
        I = (vtt-vrt)/2 * np.sum(w * erfcx((vtt-vrt)/2*u - (vtt+vrt)/2))
        return 1/(np.sqrt(np.pi) * I)
    else:
        # Intermediate regime
        # print('inter')
        I = -(vrt+vtt)/2 * np.sum(w * erfcx(-(vrt+vtt)/2*u + (vtt-vrt)/2))
        emvt2 = np.exp(-vtt**2)
        b = 2*dawsn(vtt) + emvt2*I
        return emvt2/(np.sqrt(np.pi) * b)
    

def siegert_rateD_vect(args):
    """
    vectorize and parallelize the above
    """
    with mp.Pool(mp.cpu_count()-2) as pool:
        wrkrs = [pool.apply_async(siegert_rateD, arg) for arg in args]
        res = np.array([wrkr.get() for wrkr in wrkrs]).real
    return res


def siegert_cv_unitless(mu, sigma, vr=0, vt=1, nu=0, NLaguerre=150, NLegendre=160):
    """
    Coefficient of variation of ISI of LIF neuron with mean mu and std sigma.
    Note this returns CV=std/mean, wheras Brunel 2000 unconventionally 
    considers CV=std**2/mean**2.
    """
    # if provided spare the recomputation
    if nu == 0:
        nu = siegert_rateD(mu, sigma**2/2, vr, vt, NLegendre)

    u, v = roots_legendre(NLegendre)
    y, w = roots_laguerre(NLaguerre)

    vrt = (vr - mu) / sigma
    vtt = (vt - mu) / sigma
    diff = (vtt-vrt)/2
    mean = (vtt+vrt)/2

    prefactor = 2*np.pi*nu**2*diff
    weights = np.outer(v, w)
    u = u[:, np.newaxis]
    exp = np.exp((diff*u+mean)**2 - (y-diff*u-mean)**2)
    _erfcx = erfcx(y-diff*u-mean)**2
    return np.sqrt(prefactor * np.sum(weights*exp*_erfcx))


def alpha(mu, D, vr, vt, iomega, r0=0):
    """
    Evaluate the mean response function alpha from Lindner & Schimansky-Geier
    2001 Eq. (5) at imaginary frequency arguments.
    If the response was to be evaluated at real frequencies, one would need to
    use the complex parabolic cylinder function, e.g. via mpmath.
    """

    if r0 == 0:
        r0 = siegert_rateD(mu, D, vr, vt)

    pre = np.array(r0*iomega/(iomega - 1)/np.sqrt(D))
    Delta = np.array((vr**2-vt**2+2*mu*(vt-vr))/(4*D))
    vtt = np.array((mu-vt)/np.sqrt(D))
    vrt = np.array((mu-vr)/np.sqrt(D))

    dwm1 = np.array(pbdv(iomega-1, vrt)[0])
    dw = np.array(pbdv(iomega, vrt)[0])
    num = np.array(pbdv(iomega-1, vtt)[0])
    den = np.array(pbdv(iomega, vtt)[0])
    num[dwm1>0] -= np.exp(Delta[dwm1>0]+np.log(dwm1[dwm1>0]))
    den[dw>0] -= np.exp(Delta[dw>0]+np.log(dw[dw>0]))

    return pre*num/den


def beta(mu, D, vr, vt, iomega, r0=0):
    """
    Evaluate the noise intensity response function beta from Lindner & Schimansky-Geier
    2001 Eq. (6) at imaginary frequency arguments.
    If the response was to be evaluated at real frequencies, one would need to
    use the complex parabolic cylinder function, e.g. via mpmath.
    """

    if r0 == 0:
        r0 = siegert_rateD(mu, D, vr, vt)

    pre = np.array(r0*iomega*(iomega-1)/(D*(2-iomega)))
    Delta = np.array((vr**2-vt**2+2*mu*(vt-vr))/(4*D))
    vtt = np.array((mu-vt)/np.sqrt(D))
    vrt = np.array((mu-vr)/np.sqrt(D))

    dwm2 = np.array(pbdv(iomega-2, vrt)[0])
    dw = np.array(pbdv(iomega, vrt)[0])
    num = np.array(pbdv(iomega-2, vtt)[0])
    den = np.array(pbdv(iomega, vtt)[0])
    num[dwm2>0] -= np.exp(Delta[dwm2>0]+np.log(dwm2[dwm2>0]))
    den[dw>0] -= np.exp(Delta[dw>0]+np.log(dw[dw>0]))

    return pre*num/den