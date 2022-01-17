import numpy as np

def rocket_example_postprocess(case, t, m, flightstage, T, pos, vel, acc):

    valmap = {'prelaunch': 0, 'ignition': 1, 'poweredflight': 2,
              'coastflight': 3, 'parachute': 4, 'landed': 5}
    case.addOutVal('Flight Stage', flightstage, valmap=valmap)

    case.addOutVal('Time [s]', t)
    case.addOutVal('Mass [kg]', m)
    case.addOutVal('Thrust [N]', T)
    case.addOutVal('Position [m]', pos)

    case.addOutVal('Northing [m]', pos[0, :])
    case.addOutVal('Easting [m]',  pos[1, :])
    case.addOutVal('Altitude [m]', pos[2, :])

    case.addOutVal('|Velocity| [m/s]',
                     np.sqrt(vel[0, :]**2 + vel[1, :]**2 + vel[2, :]**2))
    case.addOutVal('Velocity [m/s]', vel)
    case.addOutVal('Acceleration [m/s^2]', acc)

    case.addOutVal('Landing E [m]', pos[0, -1])
    case.addOutVal('Landing N [m]', pos[1, -1])

    distance = np.sqrt(pos[0, :]**2 + pos[1, :]**2)
    case.addOutVal('Distance [m]', distance)
    case.addOutVal('Landing Dist [m]', distance[-1])

    quadrant = None
    if pos[0, -1] >= 0 and pos[1, -1] >= 0:
        quadrant = 'I'
    elif pos[0, -1] <= 0 and pos[1, -1] >= 0:
        quadrant = 'II'
    elif pos[0, -1] <= 0 and pos[1, -1] <= 0:
        quadrant = 'III'
    elif pos[0, -1] >= 0 and pos[1, -1] <= 0:
        quadrant = 'IV'
    case.addOutVal('Landing Quadrant', quadrant)
