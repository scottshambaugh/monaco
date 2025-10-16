def baseball_example_preprocess(case):
    initial_conditions = {
        "z_init": case.invals["Z Init [m]"].val,  # Height off the ground inside the strike zone [m]
        "y_init": case.invals["Y Init [m]"].val,  # Side-to-side location inside the strike zone [m]
        "speed": case.invals["Speed Init [m/s]"].val,  # Initial speed [m/s]
        "launch_angle": case.invals["Launch Angle [deg]"].val,  # Vertical launch angle [deg]
        "side_angle": case.invals["Side Angle [deg]"].val,  # Side-to-side launch angle [deg]
        "topspin": case.invals["Topspin [rpm]"].val,  # Topspin [rpm]
        "altitude": 0,  # Baseball field altitude above sea level [m]
    }

    mass_props = {
        "mass": case.invals["Mass [kg]"].val,  # Baseball mass [kg]
        "diameter": case.invals["Diameter [m]"].val,  # Baseball Diameter [m]
    }

    aero = {
        "windspd": case.invals["Wind Speed [m/s]"].val,  # Wind speed [m/s]
        "windazi": case.invals["Wind Azi [deg]"].val,  # Azimuth from +Y wind blows from [deg]
        "cd": case.invals["CD"].val,  # Coefficient of drag for baseball []
    }

    return (initial_conditions, mass_props, aero)
