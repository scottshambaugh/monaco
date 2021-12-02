# test_MCCase.py

import pytest
from monaco.MCCase import MCCase
from monaco.MCVar import MCInVar

@pytest.fixture
def mccase():
    seed = 74494861
    from scipy.stats import norm
    invar = {'Test': MCInVar('Test', ndraws=10, ninvar=1,
                             dist=norm, distkwargs={'loc': 10, 'scale': 4},
                             seed=seed, firstcaseismedian=True)}
    mccase = MCCase(ncase=0, ismedian=False, mcinvars=invar, constvals=dict(), seed=seed)
    return mccase

def test_mccase_gen(mccase):
    assert mccase.mcinvals['Test'].val == pytest.approx(10)

def test_mccase_addoutval(mccase):
    mccase.addOutVal('TestOut', [[0, 0], [0, 0], [0, 0]])
    assert mccase.mcoutvals['TestOut'].val == [[0, 0], [0, 0], [0, 0]]
    assert mccase.mcoutvals['TestOut'].size == (3, 2)

def test_mccase_addoutval_with_valmap(mccase):
    valmap = {'a': 0, 'b': -1, 'c': -2, 'd': -3, 'e': -4, 'f': -5}
    mccase.addOutVal('TestOut2', [['a', 'b'], ['c', 'd'], ['e', 'f']], valmap=valmap)
    assert mccase.mcoutvals['TestOut2'].num == [[0, -1], [-2, -3], [-4, -5]]
