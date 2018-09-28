# !/usr/local/bin/python
# -*- coding: utf-8 -*-

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Disclaimer #
print('Ricordati le le variabili siano compatibili con il file *.txt\n')

## Load data and name variables #
data = np.loadtxt('data/data.txt', unpack = True)

x = data[0]
dx = data[1]
y = data[2]
dy = data[3]

N = len(x)

'''
## Data plot
figData, axData = plt.subplots(1, 1)
axData.errorbar(x, y, yerr=dy, xerr=dx, fmt='', ecolor='k', capsize=None, linestyle='', label='data')
axData.set_title("Test")
axData.set_xlabel("$ x \ [\mathrm{au}] $")
axData.set_ylabel("$ y \ [\mathrm{au}] $")
axData.legend()
axData.minorticks_on()
axData.grid(which = 'major', linestyle = '--')
#plt.show(axData)
'''

## Model function
def f(x, p0, p1):
    return p0 + p1*x
    
def gradf(x, p0, p1):
    return [p1, 1, x]       # dfdx, dfdp0, dfdp

def sigmaf2(x, popt, pcov):
    dfdx, dfdp0, dfdp1 = gradf(x, *popt)
    return pcov[0][0]*(dfdp0**2) + pcov[1][1]*(dfdp0**2) + 2*dfdp0*dfdp1*pcov[0][1]
    
## Initial parameters
pinit = [2., 1.]

'''
## Data plot with f(x; pinit)
figInit = figData
axInit = axData
t = np.linspace(x.min(), x.max(), 10*N)
axInit.plot(t, f(t, *pinit), linewidth = 0.7, color='b', label='$ f(x; \mathrm{p}_{\mathrm{init}}) $')
axInit.set_title("Test with $ f(x; \mathrm{p}_{\mathrm{init}}) $")
axInit.legend()
axInit.minorticks_on()
axInit.grid(which = 'major', linestyle = '--')
#plt.show(axInit)
'''

## Fitting data with curve_fit
""" Fit a series of data points with scipy (function must be in from f(x, p0, ...))"""
popt, pcov = curve_fit(f, x, y, p0=pinit, sigma=dy, absolute_sigma=True)

perr = np.sqrt(np.diag(pcov))
pcovn = np.copy(np.asarray(pcov, dtype='float64'))
s = np.sqrt(np.diag(pcovn))
for i in range(len(s)):
    for j in range(i + 1):
        p = s[i] * s[j]
        if p != 0:
            pcovn[i, j] /= p
        elif i != j:                # if s=0 and i!=j set pcov[i, j] == nan
            pcovn[i, j] = np.nan
        pcovn[j, i] = pcovn[i, j]   # pcov and pcovn should be symmetric matrices

chi2 = (((y - f(x, *popt))/dy)**2).sum()
dof = N - len(popt)
chi2n = chi2/dof


## Vertical residuals
res = y - f(x, *popt)
resMean = res.mean()


## Output fit results
print('\n## FIT RESULTS ##\n')

print('Best fit parameters')
print(popt, '\n')
print('Error on best fit parameters (i.e. sqrt of diagonal of cov matrix)')
print(perr, '\n')

print('Covariance matrix of best fi parameters')
print(pcov, '\n')
print('Normalized covariance matrix of best fi parameters')
print(pcovn, '\n')

print('Chisquare = %f, degrees of freedom = %f' % (chi2, dof))
print('Normalized chi square = %f' % chi2n)

print('\nResiduals mean = %f' %resMean)

print('\n## END ##\n')


## Data plot with f(x; pinit)
#epsilon = np.abs(x.max() - x.min())/(len(x)) * 1e-1
#X = np.linspace(x.min() - epsilon, x.max() + epsilon, len(x)*1000)
t = np.linspace(x.min(), x.max(), 10*N)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

figFit, axFit = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3, 1]})
figFit.subplots_adjust(hspace=0)

axFit[0].errorbar(x, y, yerr=dy, xerr=dx, fmt='', ecolor='k', capsize=None, linestyle='', label='data')
axFit[0].plot(t, f(t, *popt), linewidth = 0.7, color='b', label='$ f(x; \mathrm{p}_{\mathrm{opt}}) $')
axFit[0].plot(t, f(t, *popt) - np.sqrt(sigmaf2(t, popt, pcov)), linewidth = 0.7, color='r', linestyle='--')
axFit[0].plot(t, f(t, *popt) + np.sqrt(sigmaf2(t, popt, pcov)), linewidth = 0.7, color='r', linestyle='--')
axFit[0].set_title("Test with $ f(x; \mathrm{p}_{\mathrm{opt}}) $")
axFit[0].set_xlabel("$ x \ [\mathrm{au}] $")
axFit[0].set_ylabel("$ y \ [\mathrm{au}] $")
axFit[0].legend(bbox_to_anchor=(1, 1))
axFit[0].minorticks_on()
axFit[0].grid(which = 'major', linestyle = '--')

axFit[1].errorbar(x, res, yerr=dy, xerr=None, fmt='', ecolor='k', capsize=None, linestyle='', label='residuals')
axFit[1].plot(t, np.full(len(t), resMean), linewidth = 0.7, color='r', label='$ res $ mean')
axFit[1].set_xlabel("$ x \ [\mathrm{au}] $")
axFit[1].set_ylabel("$ res \ [\mathrm{au}] $")
axFit[1].legend(bbox_to_anchor=(1, 1))
axFit[1].minorticks_on()
axFit[1].grid(which = 'major', linestyle = '--')

figFit.savefig('dataFit.png', dpi = 1200, bbox_inches='tight')
plt.show()

