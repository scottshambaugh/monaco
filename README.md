# Monaco
This library is a work-in-progress under heavy development. Not recommended for 
outside use at this time.     
Originally created in 2020 by Scott Shambaugh during Coronavirus quarantine.

### Overview

At the heart of all serious forecasting, whether that be of elections, the 
spread of pandemics, weather, or the path of a rocket on its way to Mars, is a 
statistical tool known as the 
[Monte-Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method).
The Monte-Carlo method, named for the rolling of the dice at the famous Monte 
Carlo casino located in Monaco, allows you to quantify uncertainty by 
introducing randomness to otherwise deterministic processes, and seeing what 
the range of results is.

Monaco is a python library for setting up, running, and analyzing Monte-Carlo 
simulations. Users can define random input variables drawn from any of SciPy's 
continuous or discrete distributions (including custom distributions), 
preprocess and structure that data as needed to feed to their main simulation, 
run that simulation anywhere from 1 to millions of times, and postprocess 
the simulation outputs to obtain meaningful, statistically significant 
conclusions. Plotting and statistical functions specific to use cases that 
might be encountered are provided.

<p float="left" align="center">
<img width="440" height="300" src="examples/rocket/rocket_trajectory.png">  
<img width="360" height="270" src="examples/rocket/wind_vs_landing.png">
</p>

### Directory Structure:

* [build](build/)
* [examples](examples/)
    - [early_retirement_investment_portfolio](examples/early_retirement_investment_portfolio/)
    - [election](examples/election/)
    - [evidence_based_scheduling](examples/evidence_based_scheduling/) (TODO)
    - [integration](examples/integration/) (TODO)
    - [pandemic](examples/pandemic/)
    - [rocket](examples/rocket/)
* [Monaco](Monaco/)
* [templates](templates/)
* [test](test/)


### TODO:
**Before Release:**
* Flesh out READMEs and documentation
* Examples:
    * Integration
* Break out tests
* 2D Scatter Statistics
* Run on remote server (AWS)
* Parallelize preprocessing

**Future Work:**
* Run on remote server (Azure, Google Cloud)
* Get custom distributions working
* Linear trend lines on scatter plots
* Convergence plots
* Examples:
    * Evidence-Based Scheduling?
* 2D/3D Line statistics?
* Make pip installable?
* Correlation matrix input?
* Ability to plot derived data in addition to mcvars?
* 2D sensitivity contour plots?

**Done:**
* ~~Examples:~~
    * ~~Early Retirement Investment Portfolio~~
    * ~~Election Modeling~~
    * ~~Pandemic Modeling~~
    * ~~Rocket Flight~~
* ~~Save/load results to file~~
* ~~3sig / X% / Mean / Min / Max statistics~~
* ~~Make template files~~
* ~~Set up directory structure~~
* ~~Make parallelism repeatable~~
* ~~Correlation matrix output for scalars~~
* ~~Put in license~~
* ~~Map custom discrete distributions to keys that pass to functions~~
* ~~Rug plots~~
* ~~Scatter-histogram multiplots~~
* ~~Automatically split outvars~~
* ~~Highlight specific case on plot~~
* ~~Specify axis to plot on~~
* ~~Seed invars with name hash~~
* ~~Plot specific cases and highlighted cases~~
* ~~Order statistics tolerance interval and percentiles~~
* ~~Get parallelism working fast~~
* ~~Separate postprocessing from running functions~~
* ~~Progress bar~~
* ~~Continue partial results~~
* ~~Dataframe support~~
* ~~Plot tolerance intervals as shaded regions~~
* ~~Get keyboard interrupt working~~


### License:

This software is distributed under [the GPLv3.0 license](LICENSE.md).    
Please contact Scott Shambaugh for licensing this software for distribution in 
proprietary applications.

