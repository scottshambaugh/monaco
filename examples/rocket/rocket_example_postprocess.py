import numpy as np

def rocket_example_postprocess(mccase, t, m, flightstage, T, pos, vel, acc):
    
    valmap = {'prelaunch':0, 'ignition':1, 'poweredflight':2, 'coastflight':3, 'parachute':4, 'landed':5}
    mccase.addOutVal('Flight Stage', flightstage, valmap=valmap)

    mccase.addOutVal('Time [s]', t)
    mccase.addOutVal('Mass [kg]', m)
    mccase.addOutVal('Thrust [N]', T)
    mccase.addOutVal('Position [m]', pos)
    
    mccase.addOutVal('Northing [m]', pos[0,:])
    mccase.addOutVal('Easting [m]', pos[1,:])
    mccase.addOutVal('Altitude [m]', pos[2,:])

    mccase.addOutVal('|Velocity| [m/s]', np.sqrt(vel[0,:]**2 + vel[1,:]**2 + vel[2,:]**2))
    mccase.addOutVal('Velocity [m/s]', vel)
    mccase.addOutVal('Acceleration [m/s^2]', acc)

    mccase.addOutVal('Landing E [m]', pos[0,-1])
    mccase.addOutVal('Landing N [m]', pos[1,-1])

    distance = np.sqrt(pos[0,:]**2 + pos[1,:]**2)
    mccase.addOutVal('Distance [m]', distance)
    mccase.addOutVal('Landing Dist [m]', distance[-1])
    
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
\
