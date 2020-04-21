# The input arguments need to match up with the outputs from your preprocess function
def template_sim(flip, flipper, coin):
    
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
    
    # The outputs should be returned in a tuple
    return (simulation_output, )
