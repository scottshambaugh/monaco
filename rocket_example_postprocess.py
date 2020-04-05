import numpy as np

def rocket_example_postprocess(mccase, t, m, T, pos, vel, acc):
    mccase.addOutVal('Time', t)
    mccase.addOutVal('Mass', m)
    mccase.addOutVal('Thrust', T)
    mccase.addOutVal('Position', pos)
    mccase.addOutVal('Velocity', vel)
    mccase.addOutVal('Acceleration', acc)

    mccase.addOutVal('Landing E', pos[-1,1])
    mccase.addOutVal('Landing N', pos[-1,2])

    landing_dist = np.sqrt(pos[-1,1]**2 + pos[-1,2]**2)
    mccase.addOutVal('Landing Dist', landing_dist)
