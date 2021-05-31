import numpy as np

def rocket_example_run(sequence, massprops, propulsion, aero, launchsite):
    
    # Constants
    g = 9.81   # gravitational acceleration [m/s^2]
    dt = 0.1  # simulation timestep [s]
    d2r = np.pi/180  # degrees to radians conversion [rad/deg]
    flightstage = ['prelaunch']  # prelaunch, ignition, poweredflight, coastflight, parachute, or landed
    tapogee = None  # time at apogee [s]
    timeout = 1000 # simulation timeout [s]
    tmax = 45
    
    # Time histories
    t = np.array([0])  # simulation time [t]
    m = np.array([massprops['wetmass']])  # total mass [kg]
    T = np.array([0])  # thrust [N]
    pos = np.array([[0,0,0]])  # position [x,y,z] [m]: x = easting, y = northing, z = alt above sea level
    vel = np.array([[0,0,0]])  # velocity [vx,vy,vz] [m/s]: vx = easterly, vy = northerly, vz = vertically
    acc = np.array([[0,0,0]])  # acceleration [ax,ay,az] [m/s^2]: vx = easterly, vy = northerly, vz = vertically
    
    windvel = aero['windspd']*np.array([-np.sin(d2r*aero['windazi']), -np.cos(d2r*aero['windazi']), 0])
    thrustvec = np.array([np.sin(d2r*launchsite['launchangle']), np.sin(d2r*launchsite['launchangle']), np.cos(d2r*launchsite['launchangle'])]) \
                * np.array([np.sin(d2r*launchsite['launchazi']), np.cos(d2r*launchsite['launchazi']), 1])
    
    # Main calculation loop
    i = 0 
    while t[i] <= tmax:
        
        i = i+1
        t = np.append(t, t[i-1] + dt)   
        flightstage.append(flightstage[i-1])   

        # Check for events and state transitions
        if flightstage[i] == 'prelaunch':
            if t[i] >= sequence['ignition']:
                flightstage[i] = 'ignition'
        elif flightstage[i] == 'poweredflight':
            if T[i-1] <= 0:
                flightstage[i] = 'coastflight'
        elif flightstage[i] == 'coastflight':
            if vel[i-2][2] >= 0 and vel[i-1][2] <= 0:
                tapogee = t[i-1]
            if tapogee != None and t[i] >= (tapogee + sequence['parachute_delay']) and not sequence['parachute_failure']:
                flightstage[i] = 'parachute'
        
        # Look up aero data based on if parachute is deployed
        if flightstage[i] == 'parachute':
            area = np.array([aero['area_ax_parachute'], aero['area_lat_parachute'], aero['area_lat_parachute']])
            cd = np.array([aero['cd_ax_parachute'], aero['cd_lat_parachute'], aero['cd_lat_parachute']])
        else:
            area = np.array([aero['area_ax_flight'], aero['area_lat_flight'], aero['area_lat_flight']])
            cd = np.array([aero['cd_ax_flight'], aero['cd_lat_flight'], aero['cd_lat_flight']])

        # Look up thrust and mass flow data
        mdot = np.interp(t[i]-sequence['ignition'], propulsion['prop_time'], propulsion['prop_massflow'])
        m = np.append(m, m[i-1] + mdot*dt)
        T = np.append(T, np.interp(t[i]-sequence['ignition'], propulsion['prop_time'], propulsion['prop_thrust']))
        
        # Calculate and add up forces
        velfreestream = vel[i-1]-windvel
        Faero = -0.5*calcRho(pos[i-1][2])*(velfreestream**2)*np.sign(velfreestream)*cd*area
        Fg = np.array([0, 0, -m[i]*g])
        Fthrust = T[i]*thrustvec       
        Ftot = Faero + Fg + Fthrust
        
        # Check for liftoff, restrict vehicle if still on ground
        if flightstage[i] == 'ignition' and Ftot[2] > 0:
            flightstage[i] = 'poweredflight'
        elif flightstage[i] in ('prelaunch', 'ignition'):
            Ftot = np.array([0,0,0])

        # Integrate equations of motion
        acc = np.append(acc, [Ftot/m[i]], axis=0)
        vel = np.append(vel, [vel[i-1] + acc[i]*dt], axis=0)
        pos = np.append(pos, [pos[i-1] + vel[i]*dt], axis=0)
        
        # Check for landing
        if flightstage[i] not in ('prelaunch', 'ignition') and pos[i][2] <= 0:
            acc[i] = np.array([0,0,0])
            vel[i] = np.array([0,0,0])
            pos[i][2] = 0
            flightstage[i] = 'landed'
        
        # Check for timeout
        if t[i] > timeout:
            print('Timeout')
            return None

    return (t, m, flightstage, T.transpose(), pos.transpose(), vel.transpose(), acc.transpose())
    
    
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
    
    
    
'''
### Test ###
if __name__ == '__main__':
    sequence = {
        'ignition' : 1,             # time that engines are lit [s]
        'parachute_delay' : 3,      # time after apogee to open parachute [s]
        'parachute_failure' : True, # whether the parachute fails to deploy (boolean)
        }
    
    massprops = {
        'drymass': 2,  # dry mass of rocket [kg]
        'wetmass': 4,  # wet mass of rocket [kg]
        }
    
    propulsion = {
        'prop_time':      [0,  0.1,  9.9, 10], # thrust curve time [s]
        'prop_thrust' :   [0,  100,  100,  0], # thrust curve thrust [N]
        'prop_massflow' : [0, -0.2, -0.2,  0], # thrust curve mass flow [kg/s]
        }
    
    aero = {
        'windspd' : 2,                 # wind speed [m/s]
        'windazi' : 10,                # azimuth from north wind is blowing from [deg]
        'area_ax_flight' :     0.01,   # axial area during flight [m^2]
        'area_lat_flight' :    0.10,   # lateral area during flight [m^2]
        'area_ax_parachute' :  0.40,   # axial area during parachute [m^2]
        'area_lat_parachute' : 0.25,   # lateral area during parachute [m^2]
        'cd_ax_flight' :     0.3,      # axial drag coefficient during flight []
        'cd_lat_flight' :    1.0,      # lateral drag coefficient during flight []
        'cd_ax_parachute' :  1.5,      # axial drag coefficient during parachute []
        'cd_lat_parachute' : 1.2,      # lateral drag coefficient during parachute []
        }
    
    launchsite = {
        'launchangle' : 2,  # launch angle from vertical [deg]
        'launchazi' : 0,    # launch azimuth from north [deg]
        }
    
    (t, m, flightstage, T, pos, vel, acc) = rocket_example_run(sequence, massprops, propulsion, aero, launchsite)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[0,:], pos[1,:], pos[2,:])
    ax.set_xlabel('E')
    ax.set_ylabel('N')
    ax.set_zlabel('Alt')
    
    fig = plt.figure()
    plt.plot(t,vel[2,:],'b')
    plt.plot(t,acc[2,:],'r')
    plt.xlabel('Time [s]')
    plt.legend(('Vertical Vel [m/s]','Vertical Accel [m/s^s]'))
#'''
