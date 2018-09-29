# !/usr/local/bin/python
# -*- coding: utf-8 -*-

## Import libraries
import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import odrpack


##
def sample_statistics(sample):
    ''' Calculate the sample mean and (unbaised) sample standard deviation using np. Also return mean standard deviation.
    ''' 
    m = sample.mean()
    s = np.sqrt(sample.var(ddof = 1))
    
    return m, s, s/np.sqrt(len(sample))
  
    
## Media pesata e errore sulla media
def weighted_average(data, dataerr):
    ''' Fit with constant model, i.e. weighted average of a list of mesurements.
    '''
    if (len(data) != len(dataerr)):
        return print('\nErrore, array di dimensioni non corrette.\n')
    w = (1/dataerr)**2
    S = w.sum()
    q = (w*data).sum()/S
    dq = np.sqrt(1/S)
    chisq = ((((data - q)/dataerr))**2).sum()
    return q, dq, chisq
 
    
## Ordinalry least squares analitico per modello lineare
def linearOLS(xdata, ydata, yerr):
    ''' Ordinary least square for linear model y = q + m*x. Popt is [q, m]
    '''
    w = (1/yerr)**2
    Sx0 = w.sum()
    Sx1 = (w*xdata).sum()
    Sx2 = (w*xdata*xdata).sum()
    Sxy0 = (w*ydata).sum()
    Sxy1 = (w*xdata*ydata).sum()
    D = Sx0*Sx2 - Sx1*Sx1
    
    m = (Sxy1*Sx0 - Sxy0*Sx1)/D
    q = (Sxy0*Sx2 - Sxy1*Sx1)/D
    popt = np.array([q, m])
    
    dm2 = Sx0/D
    dq2 = Sx2/D
    cov_mq = - Sx1/D
    pcov = np.array(([dq2, cov_mq], [cov_mq, dm2]))
    
    return popt, pcov

    
## Curve_fit iterato con errori efficaci
def curve_fitEff(f, dfdx, xdata, ydata, xerr, yerr, pInit, absolute_sigma=True, conv_diff=1e-7, max_cycles=10, **kw):
    cycles = 1
    pInit, pcovInit = curve_fit(f, xdata, ydata, p0=pinit, sigma=dy, absolute_sigma=absolute_sigma, **kw)
    while True:
        if cycles >= max_cycles:
            cycles = -1
            break
        dyeff = np.sqrt(yerr**2 + (dfdx(xdata, *pinit) * xerr)**2)
        popt, pcov = curve_fit(f, xdata, ydata, p0=pInit, sigma=dyeff, absolute_sigma=absolute_sigma, **kw)
        poptError = abs(popt - pInit) / popt
        pcovError = abs(pcov - pcovInit) / pcov
        pInit = popt
        pcovInit = pcov
        cycles += 1
        if (poptError < conv_diff).all() and (pcovError < conv_diff).all():
            break
    print(cycles)
    return popt, pcov


## Fit con orthogonal distance regressione (ODR)
def fitScipyODR(xdata, ydata, xerr, yerr, func, pInit):
    """ Fit a series of data points with ODR (function must be in from f(beta[n], x))
    """
    model = odrpack.Model(func)
    data = odrpack.RealData(xdata, ydata, sx=xerr, sy=yerr)
    odr = odrpack.ODR(data, model, beta0=pInit)
    out = odr.run()
    popt = out.beta
    pcov = out.cov_beta
    out.pprint()
    print('Chi square = %f' % out.sum_square)
    return popt, pcov


'''
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
'''        
        
