# Import libraries
import pylab as plt
import numpy as np
from __fitting__ import *

# Disclaimer
print('Ricordati le le variabili siano compatibili con il file *.txt\n')

# Load data and name variables
x_i, x_r = np.loadtxt('../Data/Data.txt', unpack = True)

x = x_r
y = x_i
n = len(x)

dx = np.full(len(x), 1./np.sqrt(12))
dy = np.full(len(x), 1./np.sqrt(12))

# Fit
def f(x, p0):
    return p0*x
def dfdx(x, p0):
    return p0
    
pars = np.array([1.48])              # PARAMETRI INIZIAI
cov = np.array([ 1.])
    
popt, pcov = fit('OLSev', x, y, dx, dy, f, dfdx, pars, cov)

dyeff = np.sqrt(dy**2 + dfdx(x, *popt)**2 * dx**2)
chi2 = ((y - f(x, *popt))**2/dyeff**2).sum()
nu = len(y) - len(popt)
chi2n = chi2/nu

# Residui
dev = y - f(x, *popt)                                       # Verticali
#dev = (f(x, *popt) - y)/np.sqrt(1 + (dfdx(x, *popt))**2)      # Ortogonali
dev_mean = dev.mean()

# Numerical values
print('\n###### FIT RESULTS ######\n')
print('Best fit parameters')
print(popt, '\n')
print('Error on best fit parameters (i.e. sqrt of diagonal of cov matrix)')
print(np.sqrt(pcov.diagonal()), '\n')
print('Covariance matrix of best fi parameters')
print(pcov, '\n')
print('Chisquare = %f, expected = %f \pm %f' % (chi2, nu, np.sqrt(4/5 * nu)))
print('Normalized chi square = %f' % chi2n)
print('\nMedia dei residui = %f' %dev_mean)
print('\n###### END ######\n')


# Data plot
epsilon = np.abs(x.max() - x.min())/(len(x)) * 1e-1
X = np.linspace(x.min() - epsilon, x.max() + epsilon, len(x)*1000)
F = f(X, *popt)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, axs = plt.subplots(2, 1, sharex = True, gridspec_kw = {'height_ratios':[3, 1]})
fig.subplots_adjust(hspace = 0)

axs[0].errorbar(x, y, yerr = dy, xerr = dx, color = 'black', linestyle = '')
axs[0].plot(X, F, linewidth = 0.7)
axs[0].set_ylabel("$ x_{i} [\mathrm{au}] $")
axs[0].minorticks_on()
axs[0].grid(which = 'major', linestyle = '--')
#axs[0].grid(which = 'minor', linestyle = '--')

axs[1].errorbar(x, dev, yerr = dyeff, color = 'black', linestyle = '')
axs[1].plot([x.min() - epsilon, x.max() + epsilon], [dev_mean, dev_mean], color = 'b', linewidth = 0.7)
axs[1].set_xlabel("$x_{r} [\mathrm{au}] $")
axs[1].set_ylabel("$d [\mathrm{au}] $")
axs[1].minorticks_on()
axs[1].grid(which = 'major', linestyle = '--')
#axs[1].grid(which = 'minor', linestyle = '--')

plt.savefig('../plex.png', dpi = 1200)
plt.show()


'''
x_t = np.linspace(0., 30., 100)
y_t = 4.7*4/3*np.pi*(x_t/2)**3
plt.xscale('log')
plt.yscale('log')
plt.errorbar(x, y, yerr=dy, linestyle = '', fmt = '+')
plt.plot(x, y)
plt.minorticks_on()
plt.grid(which = 'major', linestyle = '--')
plt.grid(which = 'minor', linestyle = '--')
'''
plt.show()


