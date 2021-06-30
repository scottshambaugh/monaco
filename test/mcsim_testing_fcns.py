# mcsim_testing_fcns.py

from time import sleep

def testing_preprocess(mccase):
    return ([True,])

def testing_run(inputs):
    return (True)

def testing_postprocess(mccase, output):
    mccase.addOutVal('casenum', mccase.ncase)

def fcns():
    fcns ={'preprocess' :testing_preprocess,   \
           'run'        :testing_run,          \
           'postprocess':testing_postprocess}
    return fcns

def dummyfcn(*args):
    return 1

def slowfcn(n):
    sleep(1.0)

