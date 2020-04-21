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
