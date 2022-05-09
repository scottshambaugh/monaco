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
np.set_printoptions(suppress=True, precision=6)

def vars_preprocess(case):
    x1 = case.invals['x1'].val
    x2 = case.invals['x2'].val
    x3 = case.invals['x3'].val
    x4 = case.invals['x4'].val
    x5 = case.invals['x5'].val
    x6 = case.invals['x6'].val
    return (x1, x2, x3, x4, x5, x6)

def g1(x):
    return -np.sin(np.pi*x) - 0.3*np.sin(3.33*np.pi*x)
def g2(x):
    return -0.76*np.sin(np.pi*(x-0.2)) - 0.315
def g3(x):
    return -0.12*np.sin(1.05*np.pi*(x-0.2)) - 0.02*np.sin(95.24*np.pi*x) - 0.96
def g4(x):
    return -0.12*np.sin(1.05*np.pi*(x-0.2)) - 0.96
def g5(x):
    return -0.05*np.sin(np.pi*(x - 0.2)) - 1.02
def g6(x):
    return 0*x - 1.08

def vars_run(x1, x2, x3, x4, x5, x6):
    f = g1(x1) + g2(x2) + g3(x3) + g4(x4) + g5(x5) + g6(x6)
    return (f,)

def vars_postprocess(case, f):
    case.addOutVal(name='f', val=f)


def plot_gs():
    dx = 0.001
    x = np.arange(0, 1+dx, dx)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, g1(x))
    ax.plot(x, g2(x))
    ax.plot(x, g3(x))
    ax.plot(x, g4(x))
    ax.plot(x, g5(x))
    ax.plot(x, g6(x))
    plt.legend(['g1', 'g2', 'g3', 'g4', 'g5', 'g6'])
    plt.show(block=False)


def main():
    fcns ={'preprocess' :vars_preprocess,   \
           'run'        :vars_run,          \
           'postprocess':vars_postprocess}

    ndraws = 512
    sim = mc.Sim(name='dvars', ndraws=ndraws, fcns=fcns, firstcaseismedian=False,
                 seed=3462356, singlethreaded=True, daskkwargs=dict(), samplemethod='random',
                 savecasedata=False, savesimdata=False, verbose=True, debug=True)

    varnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    d = len(varnames)
    for varname in varnames:
        sim.addInVar(name=varname, dist=uniform, distkwargs={'loc':0, 'scale':1})
    sim.runSim()

    # for varname in varnames:
    #     mc.plot(sim.invars[varname], sim.outvars['f'])

    sim.calcSensitivities('f')
    sim.outvars['f'].plotSensitivities()

    return sim

if __name__ == '__main__':
    plot_gs()
    sim = main()
