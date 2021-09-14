# mcsim_testing_fcns.py
from Monaco.MCSim import MCFunctions

def testing_preprocess(mccase):
    return ([True,])

def testing_run(inputs):
    return (True)

def testing_postprocess(mccase, output):
    mccase.addOutVal('casenum', mccase.ncase)

def fcns():
    fcns ={MCFunctions.PREPROCESS :testing_preprocess,   \
           MCFunctions.RUN        :testing_run,          \
           MCFunctions.POSTPROCESS:testing_postprocess}
    return fcns

def dummyfcn(*args):
    return 1

