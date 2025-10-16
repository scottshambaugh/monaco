import numpy as np
from baseball_example_run import outfield_x


def baseball_example_postprocess(case, t, pos, vel, acc):
    case.addOutVal("Time [s]", t)
    case.addOutVal("Position [m]", pos)

    case.addOutVal("X [m]", pos[0, :])
    case.addOutVal("Y [m]", pos[1, :])
    case.addOutVal("Z [m]", pos[2, :])

    case.addOutVal("Speed [m/s]", np.sqrt(vel[0, :] ** 2 + vel[1, :] ** 2 + vel[2, :] ** 2))
    case.addOutVal("Velocity [m/s]", vel)
    case.addOutVal("Acceleration [m/s^2]", acc)

    case.addOutVal("Max Height [m]", np.max(pos[2, :]))

    landing_x = pos[0, -1]
    landing_y = pos[1, -1]
    landing_ang = np.rad2deg(np.arctan2(landing_y, landing_x))
    case.addOutVal("Landing X [m]", pos[0, -1])
    case.addOutVal("Landing Y [m]", pos[1, -1])
    case.addOutVal("Landing Angle [deg]", landing_ang)

    foul = landing_ang >= 45 or landing_ang <= -45
    homerun = landing_x > outfield_x(landing_y) and not foul
    case.addOutVal("Foul Ball", foul)
    case.addOutVal("Home Run", homerun)

    distance = np.sqrt(pos[0, :] ** 2 + pos[1, :] ** 2)
    case.addOutVal("Distance [m]", distance)
    case.addOutVal("Landing Dist [m]", distance[-1])
