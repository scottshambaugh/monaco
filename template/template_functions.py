# template_functions.py

# This file contains the three functions passed into the fcns dict used by Sim.
# It is up to you whether to put all these functions in one file, or break it up
# into multiple files as the other examples in the examples directory do.


# The preprocessing function should only take in an Case object, and extract the
# values from inside it in order to build the inputs for the run function.
def template_preprocess(case):

    # For all random variables that you initialized in your run script and will
    # be passed to your function, access them like so:
    flip = case.invals['flip'].val
    flipper = case.invals['flipper'].val

    # Import or declare any other unchanging inputs that your function needs as well.
    coin = 'quarter'

    # Structure your data, and return all the arguments to your sim function
    # packaged in a tuple. This tuple will be unpacked when your sim function
    # is called.
    return (flip, flipper, coin)


# The run function input arguments need to match up with the outputs in the unpacked
# tuple from your preprocessing function
def template_run(flip, flipper, coin):

    # Your simulation calculations will happen here. This function can also be
    # a wrapper to call another function or non-python code.

    if flip == 0:
        headsortails = 'heads'
    elif flip == 1:
        headsortails = 'tails'

    simulation_output = {'headsortails': headsortails,
                         'flipper': flipper,
                         'coin': coin}

    # The outputs should be returned in a tuple, which will be unpacked when your
    # postprocessing function is called. Note the trailing comma to make this
    # tuple iterable.
    return (simulation_output, )


# For your postprocessing function, the first argument must be the case, and
# all input arguments after case need to match up with the outputs in the unpacked
# tuple from your run function.
def template_postprocess(case, simulation_output):

    # Simulation outputs may be huge, and this is where postprocessing can be
    # done to extract the information you want. For example, you may only want
    # to know the last point in a timeseries.

    # It is good practice to provide a dictionary to map any non-number values
    # to numbers via a known valmap. If needed this will be auto-generated, but
    # manually assigning numbers will give greater control over plotting.
    valmap = {'heads': 0, 'tails': 1}

    # Add output values from this case's simulation results, case information,
    # or from other data processing that you do in this file.
    # The name supplied here will become the outvar's name.
    case.addOutVal(name='Flip Result', val=simulation_output['headsortails'], valmap=valmap)
    case.addOutVal(name='Flip Number', val=case.ncase)
