from Monaco.MCVar import MCVar
from Monaco.MCVal import MCOutVal
from typing import Dict, Union, Any
import numpy as np

class MCCase():
    def __init__(self, 
                 ncase     : int, 
                 isnom     : bool, 
                 mcinvars  : Dict[str, MCVar], 
                 constvals : Dict[str, Any] = dict(),
                 seed      : int = np.random.get_state()[1][0],
                 ):
        
        self.ncase = ncase
        self.isnom = isnom
        self.mcinvars = mcinvars 
        self.constvals = constvals
        self.mcoutvars = dict()
        self.seed = seed
        
        self.starttime = None
        self.endtime = None
        self.runtime = None
        
        self.filepath = None
        self.runsimid = None
        self.hasrun = False
        self.haspostprocessed = False
        
        self.mcinvals = self.getMCInVals()
        self.mcoutvals = dict()
        
        self.siminput = None
        self.simrawoutput = None
        

    def getMCInVals(self):
        mcvals = dict()
        for mcvar in self.mcinvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals


    def getMCOutVals(self):
        mcvals = dict()
        for mcvar in self.mcoutvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals
    
    
    def addOutVal(self, 
                  name   : str, 
                  val, # unconstrained type
                  split  : bool = True, 
                  valmap : Union[None, Dict[Any, int]] = None
                  ):
        self.mcoutvals[name] = MCOutVal(name=name, ncase=self.ncase, val=val, valmap=valmap, isnom=self.isnom)
        if split:
            self.mcoutvals.update(self.mcoutvals[name].split())


'''
### Test ###
if __name__ == '__main__':
    import numpy as np
    from scipy.stats import norm
    from MCVar import MCInVar
    seed = 74494861
    invar = {'Test':MCInVar('Test', ndraws=10, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=seed)}
    case = MCCase(ncase=0, isnom=False, mcinvars=invar, constvals=dict(), seed=seed)
    print(case.mcinvals['Test'].val)      # expected: 10.000000000000002
    
    case.addOutVal('TestOut', [[0,0],[0,0],[0,0]])
    print(case.mcoutvals['TestOut'].val)  # expected: [[0, 0], [0, 0], [0, 0]]
    print(case.mcoutvals['TestOut'].size) # expected: (3, 2)
    valmap = {'a':0,'b':-1,'c':-2,'d':-3,'e':-4,'f':-5}
    case.addOutVal('TestOut2', [['a','b'],['c','d'],['e','f']], valmap = valmap)
    print(case.mcoutvals['TestOut2'].num) # expected: [[0, -1], [-2, -3], [-4, -5]]
#'''
