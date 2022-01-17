import numpy as np

def rocket_example_preprocess(case):
    sequence = {
        'ignition': 1,         # time that engines are lit [s]
        'parachute_delay': 3,  # time after apogee to open parachute [s]
        'parachute_fails': case.invals['Parachute Failure'].val,  # chute fails to deploy [bool]
    }

    massprops = {
        'drymass': 2,  # dry mass of rocket [kg]
        'wetmass': 4,  # wet mass of rocket [kg]
    }

    # Thrust factor is applied such that total impulse is constant
    T_fac = case.invals['Thrust Factor'].val  # thrust factor []
    propulsion = {
        'prop_time':     np.array([0,  0.1,  9.9, 10])/T_fac,  # thrust curve time [s]
        'prop_thrust':   np.array([0,   50,   50,  0])*T_fac,  # thrust curve thrust [N]
        'prop_massflow': np.array([0, -0.2, -0.2,  0])*T_fac,  # thrust curve mass flow [kg/s]
    }

    aero = {
        'windspd': case.invals['Wind Speed [m/s]'].val,  # wind speed [m/s]
        'windazi': case.invals['Wind Azi [deg]'].val,    # azimuth north wind blows from [deg]
        'area_ax_flight':     0.01,                 # axial area during flight [m^2]
        'area_lat_flight':    0.10,                 # lateral area during flight [m^2]
        'area_ax_parachute':  0.40,                 # axial area during parachute [m^2]
        'area_lat_parachute': 0.25,                 # lateral area during parachute [m^2]
        'cd_ax_flight':     0.2,                    # axial drag coefficient during flight []
        'cd_lat_flight':    1.0,                    # lateral drag coefficient during flight []
        'cd_ax_parachute':  1.5,                    # axial drag coefficient during parachute []
        'cd_lat_parachute': 1.2,                    # lateral drag coefficient during parachute []
    }

    launchsite = {
        'launchangle': 2,  # launch angle from vertical [deg]
        'launchazi': 0,    # launch azimuth from north [deg]
    }

    return (sequence, massprops, propulsion, aero, launchsite)
