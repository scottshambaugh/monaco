# pickling_size_testing.py

import pickle
from scipy.stats import randint, rv_discrete
import monaco as mc

# Diagnostic functions
def pickle_sizes(o, *, top_n=10, protocol=5):
    pairs = []
    if hasattr(o, "__dict__"):
        items = o.__dict__.items()
    elif isinstance(o, (list, tuple)):
        items = enumerate(o)
    elif isinstance(o, dict):
        items = o.items()
    else:                           # fall back – whole object only
        items = [("<self>", o)]

    for k, v in items:
        try:
            pairs.append((k, len(pickle.dumps(v, protocol=protocol))))
        except Exception:
            pairs.append((k, float("inf")))   # not picklable

    pairs.sort(key=lambda kv: kv[1], reverse=True)
    for k, sz in pairs[:top_n]:
        print(f"{k:>20}: {sz:>10,} bytes")
    print(f'Total size: {sum(sz for _, sz in pairs):>10,} bytes')


# Sim setup
def preprocess(case):
    flip = case.invals['flip'].val
    flipper = case.invals['flipper'].val
    coin = 'quarter'
    return (flip, flipper, coin)

def run(flip, flipper, coin):
    if flip == 0:
        headsortails = 'heads'
    elif flip == 1:
        headsortails = 'tails'
    simulation_output = {'headsortails': headsortails,
                         'flipper': flipper,
                         'coin': coin}
    return (simulation_output, )

def postprocess(case, simulation_output):
    valmap = {'heads': 0, 'tails': 1}
    case.addOutVal(name='Flip Result', val=simulation_output['headsortails'], valmap=valmap)
    case.addOutVal(name='Flip Number', val=case.ncase)


FCNS = {'preprocess' : preprocess,
        'run'        : run,
        'postprocess': postprocess}

def monte_carlo_sim(ndraws, seed):
    sim = mc.Sim(name='Coin Flip', ndraws=ndraws, fcns=FCNS,
                 firstcaseismedian=False, seed=seed,
                 singlethreaded=True, usedask=False,
                 savecasedata=False, savesimdata=False,
                 verbose=True, debug=False)
    nummap = {0: 'Sam', 1: 'Alex'}
    sim.addInVar(name='flipper', dist=randint, distkwargs={'low': 0, 'high': 2}, nummap=nummap)
    flip_dist = rv_discrete(name='flip_dist', values=([0, 1], [0.7, 0.3]))
    sim.addInVar(name='flip', dist=flip_dist, distkwargs=dict())
    sim.runSim()

    return sim


if __name__ == '__main__':
    ndraws = 500
    seed = 12362398
    sim = monte_carlo_sim(ndraws=ndraws, seed=seed)
    pickle_sizes(sim[0])
