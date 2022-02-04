import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def baseball_example_run(initial_conditions, mass_props, aero):

    # Constants
    g = 9.81   # gravitational acceleration [m/s^2]
    d2r = np.pi/180  # degrees to radians conversion [rad/deg]
    rpm2radps = 2*np.pi/60  # rpm to rad per second conversion [rad/rev*min/sec]
    tmax = 10  # max flight time [s]

    # Time histories
    t = np.array([0])
    vel_x_init = initial_conditions['speed'] * np.cos(d2r*initial_conditions['launch_angle']) \
                                             * np.cos(d2r*initial_conditions['side_angle'])
    vel_y_init = initial_conditions['speed'] * np.cos(d2r*initial_conditions['launch_angle']) \
                                             * np.sin(d2r*initial_conditions['side_angle'])
    vel_z_init = initial_conditions['speed'] * np.sin(d2r*initial_conditions['launch_angle'])
    pos = np.array([[0, initial_conditions['y_init'], initial_conditions['z_init']]])
    vel = np.array([[vel_x_init, vel_y_init, vel_z_init]])
    acc = np.array([[0, 0, -g]])

    # Derived constants
    windvel = aero['windspd']*np.array([-np.sin(d2r*aero['windazi']),
                                        -np.cos(d2r*aero['windazi']),
                                        0])
    area = np.pi/4*mass_props['diameter']**2

    # Main calculation loop
    i = 0
    while t[i] <= tmax:

        dt = 0.1  # simulation timestep [s]
        # increase time resolution for landing
        if pos[i-1][2] < 5:
            dt = 0.020
        if pos[i-1][2] < 1:
            dt = 0.005
        if pos[i-1][2] < 0.1:
            dt = 0.001

        i = i+1
        t = np.append(t, t[i-1] + dt)

        # Calculate and add up forces
        velfreestream = vel[i-1]-windvel
        rho = calcRho(pos[i-1][2] + initial_conditions['altitude'])
        Faero = -0.5*rho*(velfreestream**2)*np.sign(velfreestream)*aero['cd']*area

        cl = 0.319*(1-np.exp(-2.48e-3*initial_conditions['topspin']*rpm2radps))
        w_vec = np.cross(np.array([0, 0, 1]), velfreestream)
        Fm_vec = np.cross(w_vec, velfreestream)
        Fm_vec = Fm_vec/np.linalg.norm(Fm_vec)
        Fmagnus = 0.5*rho*(np.linalg.norm(velfreestream)**2)*cl*area*Fm_vec

        Fg = np.array([0, 0, -mass_props['mass']*g])
        Ftot = Faero + Fmagnus + Fg

        # Integrate equations of motion
        acc = np.append(acc, [Ftot/mass_props['mass']], axis=0)
        vel = np.append(vel, [vel[i-1] + acc[i]*dt], axis=0)
        pos = np.append(pos, [pos[i-1] + vel[i]*dt], axis=0)

        # Check for landing
        if pos[i][2] <= 0:
            pos[i][2] = 0
            break

    # Backfill initial acceleration
    acc[0] = acc[1]

    return (t, pos.transpose(), vel.transpose(), acc.transpose())


def calcRho(alt):  # alt is altitude above sea level [m]
    # Constants
    p0 = 101325  # sea-level standard pressure [Pa]
    T0 = 288.15  # sea-level standard temperature [K]
    g = 9.81     # gravitational acceleration [m/s^2]
    L = 0.0065   # [K/m]
    R = 8.3145   # [J/(mol*K)]
    M = 0.02897  # [kg/mol]

    # Calculations
    # See: https://en.wikipedia.org/wiki/Density_of_air#Variation_with_altitude
    T = T0 - L*alt  # Temperature at alititude [Pa]
    p = p0*(1-(L*alt/T0))**(g*M/(R*L) - 1)  # Air pressure at alititude [Pa]
    rho = p*M/(R*T)
    return rho


# Generate a baseball field in a 3d plot
def plot_baseball_field(ax):
    d2r = np.pi/180  # degrees to radians conversion [rad/deg]

    ax.set_ylim([-80, 80])
    ax.set_xlim([-10, 150])
    ax.set_zlim([0, 45])

    # Pitchers mount
    circle_angs = np.arange(-180, 180+1, 1)*d2r
    pitchers_mound = np.array([5.47/2*np.cos(circle_angs) + 18.39, 5.47/2*np.sin(circle_angs)]).T
    # ax.plot(pitchers_mound[:, 0], pitchers_mound[:, 1], 0*pitchers_mound[:, 0], c='k', zorder=5)
    verts = [list(zip(pitchers_mound[:, 0], pitchers_mound[:, 1], 0*pitchers_mound[:, 0]))]
    coll = Poly3DCollection(verts, color='wheat')
    coll.set_sort_zpos(-1)
    ax.add_collection3d(coll)

    # Base Diamond
    ang = np.cos(d2r*45)
    diamond = np.array([[0, 0],
                        [ang, -ang],
                        [2*ang, 0],
                        [ang, ang],
                        [0, 0]]) * 27.43
    # ax.plot(diamond[:, 0], diamond[:, 1], 0*diamond[:, 0], c='k', zorder=5)
    verts = [list(zip(diamond[:, 0], diamond[:, 1], 0*diamond[:, 0]))]
    coll = Poly3DCollection(verts, color='forestgreen')
    coll.set_sort_zpos(-2)
    ax.add_collection3d(coll)

    # Infield Boundary
    angs = np.arange(-45, 45+0.1, 0.1)*d2r
    infield_ys = np.append(np.arange(-27.524, 27.524+0.1, 0.1), 27.524)
    infield_xs = np.sqrt(29**2-infield_ys**2) + 18.39
    infield = np.array([[0, 0]])
    infield = np.append(infield, np.array([infield_xs, infield_ys]).T, axis=0)
    infield = np.append(infield, np.array([[0, 0]]), axis=0)
    # ax.plot(infield[:, 0], infield[:, 1], 0*infield[:, 0], c='k', zorder=5)
    verts = [list(zip(infield[:, 0], infield[:, 1], 0*infield[:, 0]))]
    coll = Poly3DCollection(verts, color='wheat')
    coll.set_sort_zpos(-3)
    ax.add_collection3d(coll)

    # Outfield Boundary
    outfield_ys = np.sin(angs)*100
    outfield_xs = outfield_x(outfield_ys)
    outfield = np.array([[0, 0]])
    outfield = np.append(outfield, np.array([outfield_xs, outfield_ys]).T, axis=0)
    outfield = np.append(outfield, np.array([[0, 0]]), axis=0)
    ax.plot(outfield[:, 0], outfield[:, 1], 0*outfield[:, 0], c='k', zorder=5)
    verts = [list(zip(outfield[:, 0], outfield[:, 1], 0*outfield[:, 0]))]
    coll = Poly3DCollection(verts, color='forestgreen')
    coll.set_sort_zpos(-4)
    ax.add_collection3d(coll)

    return


# Equation for the outfield boundary (350 ft x 400 ft)
def outfield_x(outfield_y):
    return 125 - 0.01085786 * outfield_y**2


# '''
### Test ###
if __name__ == '__main__':

    mph2mps = 0.447
    initial_conditions = {
        'z_init': 1,
        'y_init': 0,
        'speed': 80*mph2mps,
        'launch_angle': 30,
        'side_angle': 10,
        'topspin': -2000,
        'altitude': 0,
    }

    mass_props = {
        'mass': 0.145,
        'diameter': 0.074,
    }

    aero = {
        'windspd': 10,
        'windazi': 0,
        'cd' : 0.3,
    }

    (t, pos, vel, acc) \
        = baseball_example_run(initial_conditions, mass_props, aero)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[0, :], pos[1, :], pos[2, :], c='k', zorder=20)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plot_baseball_field(ax)

    plt.show()
# '''
