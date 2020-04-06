from scipy.stats import uniform
from MCSim import MCSim
from MCPlot import MCPlot

from rocket_example_sim import rocket_example_sim
from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
fcns ={'preprocess' :rocket_example_preprocess,   \
       'run'        :rocket_example_sim,          \
       'postprocess':rocket_example_postprocess}

ndraws = 25
seed=123098

sim = MCSim('Rocket', ndraws, fcns, seed)

sim.addInVar('windazi', uniform, (0, 360))
sim.addInVar('windspd', uniform, (0, 2))

sim.runSim()

print(sim.runtime)

print(sim.mcoutvars['Landing Dist'].stats())
MCPlot(sim.mcoutvars['Time'], sim.mcoutvars['Distance'])
MCPlot(sim.mcoutvars['Landing Dist'])
MCPlot(sim.mcoutvars['Landing E'], sim.mcoutvars['Landing N'])
MCPlot(sim.mcoutvars['Position'])
