import numpy as np
import os
import pylab

from scipy.optimize import curve_fit
from scipy.odr import odrpack

def sample_statistics(sample):
    ''' Calculate the sample mean and (unbaised) sample standard deviation using np. Also return mean standard deviation.
    ''' 
    m = sample.mean()
    s = np.sqrt(sample.var(ddof = 1))
    
    return m, s, s/np.sqrt(len(sample))
    
def weighted_average(y, dy):
    ''' Fit with constant model, i.e. weighted average of a list of mesurements.
    '''
    if (len(y) != len(dy)):
        return print('\nErrore, array di dimensioni non corrette.\n')
    w = (1/dy)**2
    S = w.sum()
    q = (w*y).sum()/S
    dq = np.sqrt(1/S)
    chisq = ((((y - q)/dy))**2).sum()
    chisqn = chisq/(len(y) - 1)
    return q, dq, chisqn
    
def linearOLS(x, y, dy):
    ''' Ordinary least square for linear model y = m*x + q.
    '''
    w = (1/dy)**2
    Sx0 = w.sum()
    Sx1 = (w*x).sum()
    Sx2 = (w*x*x).sum()
    Sxy0 = (w*y).sum()
    Sxy1 = (w*x*y).sum()
    D = Sx0*Sx2 - Sx1*Sx1
    
    m = (Sxy1*Sx0 - Sxy0*Sx1)/D
    q = (Sxy0*Sx2 - Sxy1*Sx1)/D
    popt = np.array([m, q])
    
    dm2 = Sx0/D
    dq2 = Sx2/D
    cov_mq = - Sx1/D
    pcov = np.array(([dm2, cov_mq], [cov_mq, dq2]))
    
    return popt, pcov
    
def OLSev(f, dfdx, x, y, dx, dy, par, cov, absolute_sigma = True, conv_diff = 1e-7, max_cycles = 10, **kw):
    cycles = 1
    while True:
        if cycles >= max_cycles:
            cycles = -1
            break
        dyeff = np.sqrt(dy**2 + (dfdx(x, *par) * dx)**2)
        npar, ncov = curve_fit(f, x, y, p0 = par, sigma = dyeff, absolute_sigma = absolute_sigma, **kw)
        perror = abs(npar - par) / npar
        cerror = abs(ncov - cov) / ncov
        par = npar
        cov = ncov
        cycles += 1
        if (perror < conv_diff).all() and (cerror < conv_diff).all():
            break
    print(cycles)
    return par, cov

def fitScipyNLLS(x, y, dy, func, pars):
    """ Fit a series of data points with scipy (function must be in from f(x, p0, ...))
    """
    popt, pcov = curve_fit(func, x, y, p0 = pars, sigma = dy, absolute_sigma = True)
    perr = np.sqrt(pcov.diagonal())
    return popt, perr

def fitScipyODR(x, y, dx, dy, func, pars):
    """ Fit a series of data points with ODR (function must be in from f(beta[n], x))
    """
    model = odrpack.Model(func)
    data = odrpack.RealData(x, y, sx = dx, sy = dy)
    odr = odrpack.ODR(data, model, beta0 = pars)
    out = odr.run()
    popt = out.beta
    pcov = out.cov_beta
    out.pprint()
    print('Chi square = %f' % out.sum_square)
    return popt, pcov

def fit(engine, x, y, dx, dy, f, dfdx, pars, cov = None):
    """ Common interface to all the fit engines.
    """
    if engine == 'scipy.curve_fit':
        return fitScipyNLLS(x, y, dy, f, pars)
    elif engine == 'scipy.odr':
        print('Function must be in from f(beta[n], x)')
        return fitScipyODR(x, y, dx, dy, f, pars)
    elif engine == 'OLSev':
        return OLSev(f, dfdx, x, y, dx, dy, pars, cov)
    else:
        sys.exit('Unkown fit engine %s.' % engine)

def pol1(x, m, q):
    """ Linear model in curve_fit form
    """
    return m*x + q

def dplo1dx(x, m, q):
    return m

def pol1ODR(p, x):
    """ Linear model in ODR from
    """
    return p[0]*x + p[1]

def dpol1ODRdx(p, x):
    return p[0]

def fitLinear(engine, x, y, dx, dy, pars):
    """ Common interface to all the fit engines.
    """
    if engine == 'linearOLS':
        return linearOLS(x, y, dy)
    elif engine == 'scipy.odr':
        return fitScipyODR(x, y, dx, dy, pol1ODR, pars)
    else:
        sys.exit('Unkown fit engine %s.' % engine)
        
        
