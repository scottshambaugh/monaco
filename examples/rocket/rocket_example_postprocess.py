import numpy as np

def rocket_example_postprocess(mccase, t, m, flightstage, T, pos, vel, acc):
    
    valmap = {'prelaunch':0, 'ignition':1, 'poweredflight':2, 'coastflight':3, 'parachute':4, 'landed':5}
    mccase.addOutVal('Flight Stage', flightstage, valmap=valmap)

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
    
    quadrant = None
    if pos[0,-1] > 0 and pos[1,-1] > 0:
        quadrant = 'I'
    elif pos[0,-1] < 0 and pos[1,-1] > 0:
        quadrant = 'II'
    elif pos[0,-1] < 0 and pos[1,-1] < 0:
        quadrant = 'III'
    elif pos[0,-1] > 0 and pos[1,-1] < 0:
        quadrant = 'IV'
    mccase.addOutVal('Landing Quadrant', quadrant)
    
    mccase.addOutVal('Alt', pos[2,:])

