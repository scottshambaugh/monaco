import numpy as np

def rocket_example(sequence, massprops, propulsion, aero, launchsite):
    
    # Constants
    g = 9.81   # gravitational acceleration [m/s^2]
    dt = 0.01  # simulation timestep [s]
    flightstage = 'prelaunch'  # prelaunch, poweredflight, coastflight, parachute, or landed
    
    # Time histories
    t = [0,]  # simulation time [t]
    m = [massprops.wetmass,]  # total mass [kg]
    T = [0,]  # thrust [N]
    pos = [[0,0,0],]  # position [x,y,z] [m]: x = alt above sea level, y = easting, z = northing
    vel = [[0,0,0],]  # position [vx,vy,vz] [m/s]: x = vertical, y = easterly, z = northerly
    acc = [[0,0,0],]  # acceleration [ax,ay,az] [m/s^2]: x = vertical, y = easterly, z = northerly
    
    
    return False
    
    
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
sequence = {
    'ignition' : 1, # time that engines are lit [s]
    'parachute_delay' : 3, # time after apogee to open parachute [s]
    }

massprops = {
    'drymass': 1,  # dry mass of rocket [kg]
    'wetmass': 3,  # wet mass of rocket [kg]
    }

propulsion = {
    'prop_time':      [0, 0.1,   5,  9.9, 10], # thrust curve time [s]
    'prop_thrust' :   [0,   5,   5,    4,  0], # thrust curve thrust [N]
    'prop_massflow' : [0, 0.1, 0.1, 0.08,  0], # thrust curve mass flow [kg/s]
    }

aero = {
    'windspd' : 2,                # wind speed [m/s]
    'windazi' : 0,                # azimuth from north wind is blowing from [deg]
    'area_ax_flight' :     0.01,  # axial area during flight [m^2]
    'area_lat_flight' :    0.05,  # lateral area during flight [m^2]
    'area_ax_parachute' :  0.25,  # axial area during parachute [m^2]
    'area_lat_parachute' : 0.15,  # lateral area during parachute [m^2]
    'cd_ax_flight' :     0.1,     # axial drag coefficient during flight []
    'cd_lat_flight' :    1.0,     # lateral drag coefficient during flight []
    'cd_ax_parachute' :  2.0,     # axial drag coefficient during parachute []
    'cd_lat_parachute' : 1.2,     # lateral drag coefficient during parachute []
    }

launchsite = {
    'launchangle' : 10, # launch angle from vertical [deg]
    'launchazi' : 0,    # launch azimuth from north [deg]
    }

rocket_example(sequence, massprops, propulsion, aero, launchsite)
#'''