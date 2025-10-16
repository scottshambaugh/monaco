# sim_testing_fcns.py
from monaco.mc_enums import SimFunctions


def testing_preprocess(case):
    return [
        True,
    ]


def testing_run(inputs):
    return True


def testing_postprocess(case, output):
    case.addOutVal("casenum", case.ncase)


def fcns():
    fcns = {
        SimFunctions.PREPROCESS: testing_preprocess,
        SimFunctions.RUN: testing_run,
        SimFunctions.POSTPROCESS: testing_postprocess,
    }
    return fcns


def dummyfcn(*args):
    return 1
