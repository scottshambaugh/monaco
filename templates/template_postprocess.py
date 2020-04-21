# All input arguments after mccase need to match up with the outputs from your simulation
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
