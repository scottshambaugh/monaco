# test_MCCase.py

import pytest
from scipy.stats import norm
from Monaco.MCCase import MCCase
from Monaco.MCVar import MCInVar

@pytest.fixture
def case():
    seed = 74494861
    invar = {'Test':MCInVar('Test', ndraws=10, ninvar=1, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=seed)}
    case = MCCase(ncase=0, isnom=False, mcinvars=invar, constvals=dict(), seed=seed)
    return case

def test_mccase_gen(case):
    assert case.mcinvals['Test'].val == pytest.approx(10)

def test_mccase_addoutval(case):
    case.addOutVal('TestOut', [[0,0],[0,0],[0,0]])
    assert case.mcoutvals['TestOut'].val == [[0, 0], [0, 0], [0, 0]]
    assert case.mcoutvals['TestOut'].size == (3, 2)

def test_mccase_addoutval_with_valmap(case):
    valmap = {'a':0,'b':-1,'c':-2,'d':-3,'e':-4,'f':-5}
    case.addOutVal('TestOut2', [['a','b'],['c','d'],['e','f']], valmap = valmap)
    assert case.mcoutvals['TestOut2'].num == [[0, -1], [-2, -3], [-4, -5]]

### Inline Testing ###
'''
if __name__ == '__main__':
    from scipy.stats import norm
    from Monaco.MCVar import MCInVar
    seed = 74494861
    invar = {'Test':MCInVar('Test', ndraws=10, ninvar=1, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=seed)}
    case = MCCase(ncase=0, isnom=False, mcinvars=invar, constvals=dict(), seed=seed)
    print(case.mcinvals['Test'].val)      # expected: 10.000000000000002
    
    case.addOutVal('TestOut', [[0,0],[0,0],[0,0]])
    print(case.mcoutvals['TestOut'].val)  # expected: [[0, 0], [0, 0], [0, 0]]
    print(case.mcoutvals['TestOut'].size) # expected: (3, 2)
    valmap = {'a':0,'b':-1,'c':-2,'d':-3,'e':-4,'f':-5}
    case.addOutVal('TestOut2', [['a','b'],['c','d'],['e','f']], valmap = valmap)
    print(case.mcoutvals['TestOut2'].num) # expected: [[0, -1], [-2, -3], [-4, -5]]
#'''
