# This file contains the three functions passed into the fcns dict used by MCSim.
# It is up to you whether to put all these functions in one file, or break it up
# into multiple files as the other examples in the examples directory do


# The preprocessing function should only take in an MCCase object, and extract the
# values from inside it in order to build the inputs for the run function
def template_preprocess(mccase):
    
    # For all random variables that you initialized in your run script and will 
    # be passed to your function, access them like so:
    flip = mccase.mcinvals['flip'].val
    flipper = mccase.mcinvals['flipper'].val
    
    # Import or declare any other unchanging inputs that your function needs as well
    coin = 'quarter'
    
    # Structure your data, and return all the arguments to your sim function packaged in a tuple
    # This tuple will be unpacked when your sim function is called
    return (flip, flipper, coin)



# The run function input arguments need to match up with the outputs in the unpacked
# tuple from your preprocessing function
def template_run(flip, flipper, coin):
    
    # Your simulation calculations will happen here. This function can also be 
    # a wrapper to call another function or non-python code
    
    if flip == 0:
        headsortails = 'heads'
    elif flip == 1:
        headsortails = 'tails'

    simulation_output = {'headsortails':headsortails,  \
                         'flipper':flipper,            \
                         'coin':coin,                  \
                        }
    
    # The outputs should be returned in a tuple, which will be unpacked when your postprocessing 
    # function is called
    return (simulation_output, )



# All input arguments after mccase need to match up with the outputs in the unpacked
# tuple from your preprocess function
def template_postprocess(mccase, simulation_output):
    
    # Simulation outputs may be huge, and this is where postprocessing can be
    # done to extract the information you want. For example, you may only want
    # to know the last point in a timeseries.
    
    # It is good practice to provide a dictionary to map any non-number values
    # to numbers via a known valmap. If needed this will be auto-generated, but
    # manually assigning numbers will give greater control over plotting
    valmap = {'heads':0, 'tails':1}
    
    # Add output values from this case's simulation results, case information, 
    # or from other data processing that you do in this file. 
    # The name supplied here will become the var's name
    mccase.addOutVal(name='Flip Result', val=simulation_output['headsortails'], valmap=valmap)
    mccase.addOutVal(name='Flip Number', val=mccase.ncase)
