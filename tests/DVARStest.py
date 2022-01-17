# DVARStest.py
# flake8: noqa

'''
Implements the D-VARS algorithm as dscribed in:
Sheikholeslami, Razi, and Saman Razavi. "A fresh look at variography: measuring
dependence and possible sensitivities across geophysical systems from any given
data." Geophysical Research Letters 47.20 (2020): e2020GL089829.

Multi-SQP is replaced with scipy's minimize function implementing L-BFGS-B.
'''

import monaco as mc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.optimize import minimize
from numba import jit
np.set_printoptions(suppress=True, precision=6)

def vars_preprocess(mccase):
    x1 = mccase.mcinvals['x1'].val
    x2 = mccase.mcinvals['x2'].val
    x3 = mccase.mcinvals['x3'].val
    x4 = mccase.mcinvals['x4'].val
    x5 = mccase.mcinvals['x5'].val
    x6 = mccase.mcinvals['x6'].val
    return (x1, x2, x3, x4, x5, x6)

def g1(x):
    return -np.sin(np.pi*x) - 0.3*np.sin(3.33*np.pi*x)
def g2(x):
    return -0.76*np.sin(np.pi*(x-0.2)) - 0.315
def g3(x):
    return -0.12*np.sin(1.05*np.pi*(x-0.2)) - 0.2*np.sin(95.24*np.pi*x) - 0.96
def g4(x):
    return -0.12*np.sin(1.05*np.pi*(x-0.2)) - 0.96
def g5(x):
    return -0.05*np.sin(np.pi*(x - 0.2)) - 1.02
def g6(x):
    return -1.08

def vars_run(x1, x2, x3, x4, x5, x6):
    f = g1(x1) + g2(x2) + g3(x3) + g4(x4) + g5(x5) + g6(x6)
    #f = np.sin(np.pi*(0.2-x1))
    return (f,)

def vars_postprocess(mccase, f):
    mccase.addOutVal(name='f', val=f)


def main():
    fcns ={'preprocess' :vars_preprocess,   \
           'run'        :vars_run,          \
           'postprocess':vars_postprocess}

    ndraws = 512
    sim = mc.MCSim(name='dvars', ndraws=ndraws, fcns=fcns, firstcaseismedian=False, seed=3462356, cores=1, samplemethod='random', savecasedata=False, savesimdata=False, verbose=True, debug=True)

    varnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    d = len(varnames)
    for varname in varnames:
        sim.addInVar(name=varname, dist=uniform, distkwargs={'loc':0, 'scale':1})
    sim.runSim()

    #for varname in varnames:
    #    plt.scatter(sim.mcinvars[varname].nums, sim.mcoutvars['f'].nums)

    dh = 1e-3
    Hj_min = 0.1
    phi_opt = mc.calc_phi_opt(sim, Hj_min)
    variance = np.var(np.array([sim.mcoutvars['f'].nums]))
    for Hj in np.arange(0.1, 1.1, 0.1):
        Gamma = []
        for j in range(d):
            Gamma.append(mc.calc_Gammaj(Hj, dh, phi_opt[j], variance))
        Gamma = np.array(Gamma)/sum(Gamma)
        print(f'Hj = {Hj:0.2f}, Γ = {Gamma}')

    #'''
    '''
    nL = 1000
    phis = np.zeros((nL, sim.ninvars))
    Ls = np.zeros((nL, 1))
    for j in range(sim.ninvars):
        phis[:,j] = phi_min + (phi_max-phi_min)*mc.mc_sampling(ndraws=nL, method='random', ninvar=j, ninvar_max=sim.ninvars, seed=j)
    for i in range(nL):
        Ls[i] = L_runner(phis[i,:], X, Y)

    plt.hist(Ls, bins=20)
    plt.show(block=True)
    #'''

    return sim

if __name__ == '__main__':
    sim = main()