from MCSim import MCSim
from scipy.stats import uniform, describe

from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
from rocket_example_sim import rocket_example_sim

ndraws = 5
seed=123098

sim = MCSim('Rocket', ndraws, seed)

sim.addInVar('windazi', uniform, (0, 360))

sim.genCases()

for i in range(sim.ncases):
    sim_input = rocket_example_preprocess(sim.mccases[i])
    sim_raw_output = rocket_example_sim(*sim_input)
    rocket_example_postprocess(sim.mccases[i], *sim_raw_output)

sim.genOutVars()
print(sim.mcoutvars['Landing Dist'].vals)
print(describe(sim.mcoutvars['Landing Dist'].vals))
