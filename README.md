# monaco
![Release](https://img.shields.io/github/v/release/scottshambaugh/monaco?sort=semver)
![Builds](https://github.com/scottshambaugh/monaco/actions/workflows/builds.yml/badge.svg)
![Tests](https://github.com/scottshambaugh/monaco/actions/workflows/tests.yml/badge.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/monaco)

### Overview

At the heart of all serious forecasting, whether that be of elections, the spread of pandemics, weather, or the path of a rocket on its way to Mars, is a statistical tool known as the [Monte-Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The Monte-Carlo method, named for the rolling of the dice at the famous Monte Carlo casino located in Monaco, allows you to quantify uncertainty by introducing randomness to otherwise deterministic processes, and seeing what the range of results is.

`monaco` is a python library for setting up, running, and analyzing Monte-Carlo simulations. Users can define random input variables drawn using chosen sampling methods from any of SciPy's continuous or discrete distributions (including custom distributions), preprocess and structure that data as needed to feed to their main simulation, run that simulation in parallel anywhere from 1 to millions of times, and postprocess the simulation outputs to obtain meaningful, statistically significant conclusions. Plotting and statistical functions specific to use cases that might be encountered are provided, and repeatability of results is ensured through careful management of random seeds.

<p float="left" align="center">
<img width="293.08" height="270" src="https://raw.githubusercontent.com/scottshambaugh/monaco/master/examples/rocket/rocket_trajectory.png">  
<img width="384.94" height="270" src="https://raw.githubusercontent.com/scottshambaugh/monaco/master/examples/rocket/wind_vs_landing.png">
</p>

### Installation:
General usage:
```
pip install monaco
```

Installing from source and running tests:
```
git clone https://github.com/scottshambaugh/monaco.git
cd monaco
pip install poetry
poetry install
poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m
```

### Quick Start:
Copy the two files from the [template directory](https://github.com/scottshambaugh/monaco/tree/master/template), which contains a simple, well commented Monte Carlo simulation of flipping coins.    
Then, check out the other [examples](https://github.com/scottshambaugh/monaco/tree/master/examples) for inspiration and more in-depth usage of `monaco`'s features.

### Basic Architecture:
At the center of a Monte Carlo simulation is a program which you wish to run with randomized inputs. Around this, monaco's Monte Carlo architecture is structured like a sandwich. At the top, you generate a large number of randomized values for your input variables. These input values are preprocessed into the form that your program expects, your program is run, and at the bottom the results are postprocessed to extract values for select output variables. You can then plot, collect statistics about, or otherwise use all the input and output variables from your sim. The sandwich is sliced vertically into individual cases, which are run in parallel to massively speed up computation.

<p float="left" align="left">
<img width="500" height="400" src="https://raw.githubusercontent.com/scottshambaugh/monaco/master/docs/val_var_case_architecture.png">  
</p>

### Basic Workflow:
To run a Monte Carlo simulation, you must first have a function which you wish to run with randomized inputs. This python function can be a standalone script, a script which sets up and calls additional routines, or a wrapper for code written in other languages. We will call this your `run` function.

The workhorse of the Monte Carlo Simulation which you will run is an `MCSim` object. To instantiate this object, you need to pass it two things: the number of random cases `ncases` which you want to run with, and a dict `fcns` of the handles for three functions which you need to create: `preprocess`, `run`, and `postprocess`. The processing functions will be explained in more detail in a moment. You can also choose [a random sampling method, explained here](docs/sampling_methods.md).

Once you have your MCSim object, you need to generate the randomized values which your `run` function will use. To do this, use the `MCSim.addInVar()` method to generate `MCInVar` objects. This method takes in the handle to any of SciPy's continuous or discrete probability distributions, as well as the required arguments for that probability distribution. See [info for some common distributions here](docs/statistical_distributions.md). It then randomly draws `ncases` different numbers from that distribution, and assigns them to `MCInVal` objects. The raw value is stored in `MCInVal.num`. Optionally if nonnumeric random inputs are desired, the method can also take in a `nummap` dictionary which maps the randomly drawn integers to values of other types, stored in `MCInVal.val`.

Once all input variables have been added, the sim can be run with `MCSim.runSim()`. The first thing that this does is generate `ncases` number of `MCCase` objects. Each of these objects stores the n'th value of each of the input variables. A repeatably random seed for each case is also generated for use by your `run` function, if you use additional randomness inside your function that needs seeding.

Your function will now be run for all the individual cases via the call chain shown below. This should give some light into what the three functions you passed to `MCSim` are doing. `preprocess` needs to take in an `MCCase` object, extract its random values, and package that along with any other data into the structure that `run` expects for its input arguments. The `run` function then executes on those inputs arguments and returns its outputs. The `postprocess` function then needs to take in the original `mccase` as well as those outputs. Within that function, you will need to perform any postprocessing you wish to do, and choose what data to log out by calling `MCCase.addOutVal(val)` on those values. The outputs of `preproccess` and `run` both need to be packaged into tuples for unpacking later.

    postprocess( mccase, *run( *preprocess( mccase ) ) )

A progress bar in the terminal will show the progress as the results for all cases are calculated. After it is complete, all the output values will be compiled into `MCOutVar` objects, and you can now interrogate the combined results. Several things which you might want to do are built in:

1) Calculate statistics on the ensemble of input or output values, via the `MCVar.addVarStat` method.

2) Plot the ensemble of results for a value, either on its own or against other values. The `mc_plot` function intelligently chooses how to plot the variables you pass it, whether that data are singular values, timeseries, or 2D/3D vectors. Histograms, cumulative density plots, 2D/3D scatter plots, and 2D/3D line plots are manually callable for more fine grained control. Specific cases can be highlighted, and the statistics calculated in (1) above can be added to the plots as well.

3) Calculate correlations between input and output variables. This shows the linear sensitivity of the output variables to the change in input variables, and can also be plotted.

See the [examples](examples/) folder for some examples you can step through or use as templates. Some details and several more features here have been passed over, and this documentation will be fleshed out in the future. Of note, saving and loading results to file, 'nominal' cases, running on remote servers, using order statistics, and additional plotting options will need to be explained.


### License / Citation:
Copyright 2020-2021 Scott Shambaugh, distributed under [the GPLv3.0 (or later) license](LICENSE.md).    

If you use `monaco` to do research that gets published, please cite [the monaco github page](https://github.com/scottshambaugh/monaco).
