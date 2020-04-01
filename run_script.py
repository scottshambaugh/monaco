from MCSim import MCSim
from scipy.stats import *

from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
from rocket_example_sim import rocket_example_sim

ncases = 5
sim = MCSim('Rocket', 5)

sim.addVar('windazi', uniform, (0, 360))

sim.genCases()

for i in range(ncases):
    (sequence, massprops, propulsion, aero, launchsite) = rocket_example_preprocess(sim.mccases[i])
    (t, m, T, pos, vel, acc) = rocket_example_sim(sequence, massprops, propulsion, aero, launchsite)
    landing_dist = rocket_example_postprocess(t, m, T, pos, vel, acc)
    print(landing_dist)
