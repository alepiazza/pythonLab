# !/usr/local/bin/python
# -*- coding: utf-8 -*-

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from fitting import *

# Disclaimer #
print('Ricordati le le variabili siano compatibili con il file *.txt\n')

## Load data and name variables #
data = np.loadtxt('...', unpack = True)

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
    return [p1, 1., x]       # dfdx, dfdp0, dfdp
    
def dfdx(x, p0, p1):
    return gradf(x, p0, p1)[0]

def sigmaf2(x, popt, pcov):
    dfdx, dfdp0, dfdp1 = gradf(x, *popt)
    return abs(pcov[0][0]*(dfdp0**2) + pcov[1][1]*(dfdp0**2) + 2*dfdp0*dfdp1*pcov[0][1])
    
## Initial parameters
pinit = [2., 1.]

'''
## Data plot with f(x; pinit)
t = np.linspace(x.min(), x.max(), 10*N)
figInit, axInit = plt.subplots(1, 1)
axInit.errorbar(x, y, yerr=dy, xerr=dx, fmt='', ecolor='k', capsize=None, linestyle='', label='data')
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
#chi2Eff = (((y - f(x, *popt))**2)/(dy**2 + dfdx(x, *popt)**2 * dx**2)).sum()

dof = N - len(popt)
chi2n = chi2/dof
#chi2Effn = chi2Eff/dof


## Vertical residuals
resn = (y - f(x, *popt))/dy
resnMean = resn.mean()


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
#print('Chisquare = %f, degrees of freedom = %f' % (chi2Eff, dof))
#print('Normalized chi square = %f' % chi2Effn)

print('\nNormalized residuals mean = %f' %resnMean)

print('\n## END ##\n')



## Data plot with f(x; pinit)
''' Plot best fit function slightly before and after max and min data '''
epsilon = (np.abs(x.max() - x.min())/N) * 0.5

''' LaTeX style, makes data plot very slow '''
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif', size=11)

figFit, axFit = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3, 1]})
figFit.subplots_adjust(hspace=0)

# Plot data
axFit[0].errorbar(x, y, yerr=dy, xerr=dx, fmt='', ecolor='k', capsize=None, linestyle='', label='data')

# Plot best fit curve (and confidence band)
t = np.linspace(x.min() - epsilon, x.max() + epsilon, 10*N)
axFit[0].plot(t, f(t, *popt), linewidth = 0.7, color='b', label='$ I = \hat{I}_0 + \Delta V / \hat{R} $')
#axFit[0].plot(t, f(t, *popt) - np.sqrt(sigmaf2(t, popt, pcov)), linewidth = 0.7, color='r', linestyle='--')
#axFit[0].plot(t, f(t, *popt) + np.sqrt(sigmaf2(t, popt, pcov)), linewidth = 0.7, color='r', linestyle='--')
axFit[0].set_title("Data plot with best-fit curve and normalized residuals")
axFit[0].set_xlabel("$ \Delta V \ [\mathrm{V}] $")
axFit[0].set_ylabel("$ I \ [\mathrm{mA}] $")
axFit[0].legend(bbox_to_anchor=(1, 1))
axFit[0].minorticks_on()
axFit[0].grid(which = 'major', linestyle = '--')

# Plot residuals
axFit[1].errorbar(x, resn, yerr=dy, xerr=None, fmt='', ecolor='k', capsize=None, linestyle='', label='norm. residuals')
axFit[1].plot(t, np.full(len(t), resnMean), linewidth = 0.7, color='r', label='$ resn $ mean')
axFit[1].set_xlabel("$ \Delta V \ [\mathrm{V}] $")
axFit[1].set_ylabel("$ resn \ [-] $")
axFit[1].legend(bbox_to_anchor=(1, 1))
axFit[1].minorticks_on()
axFit[1].grid(which = 'major', linestyle = '--')

#figFit.savefig('dataFit.png', dpi = 1200, bbox_inches='tight')
plt.show()
