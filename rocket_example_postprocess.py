import numpy as np

def rocket_example_postprocess(mccase, t, m, T, pos, vel, acc):
    mccase.addOutVal('Time', t)
    mccase.addOutVal('Mass', m)
    mccase.addOutVal('Thrust', T)
    mccase.addOutVal('Position', pos)
    mccase.addOutVal('Velocity', vel)
    mccase.addOutVal('Acceleration', acc)

    mccase.addOutVal('Landing E', pos[0,-1])
    mccase.addOutVal('Landing N', pos[1,-1])

    distance = np.sqrt(pos[0,:]**2 + pos[1,:]**2)
    mccase.addOutVal('Distance', distance)
    mccase.addOutVal('Landing Dist', distance[-1])
    
    mccase.addOutVal('Alt', pos[2,:])

