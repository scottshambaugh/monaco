# Basic Workflow

To run a Monte Carlo simulation, you must first have a function which you wish to run with randomized inputs. This python function can be a standalone script, a script which sets up and calls additional routines, or a wrapper for code written in other languages. We will call this your `run` function.

The workhorse of the Monte Carlo Simulation which you will run is an `MCSim` object. To instantiate this object, you need to pass it two things: the number of random cases `ncases` which you want to run with, and a dict `fcns` of the handles for three functions which you need to create: `preprocess`, `run`, and `postprocess`. The processing functions will be explained in more detail in a moment. You can also choose [a random sampling method, explained here](https://monaco.readthedocs.io/en/latest/sampling_methods.html).

Once you have your MCSim object, you need to generate the randomized values which your `run` function will use. To do this, use the `MCSim.addInVar()` method to generate `MCInVar` objects. This method takes in the handle to any of SciPy's continuous or discrete probability distributions, as well as the required arguments for that probability distribution. See [info for some common distributions here](https://monaco.readthedocs.io/en/latest/statistical_distributions.html). It then randomly draws `ncases` different numbers from that distribution, and assigns them to `MCInVal` objects. The raw value is stored in `MCInVal.num`. Optionally if nonnumeric random inputs are desired, the method can also take in a `nummap` dictionary which maps the randomly drawn integers to values of other types, stored in `MCInVal.val`.

Once all input variables have been added, the sim can be run with `MCSim.runSim()`. The first thing that this does is generate `ncases` number of `MCCase` objects. Each of these objects stores the n'th value of each of the input variables. A repeatably random seed for each case is also generated for use by your `run` function, if you use additional randomness inside your function that needs seeding.

Your function will now be run for all the individual cases via the call chain shown below. This should give some light into what the three functions you passed to `MCSim` are doing. `preprocess` needs to take in an `MCCase` object, extract its random values, and package that along with any other data into the structure that `run` expects for its input arguments. The `run` function then executes on those inputs arguments and returns its outputs. The `postprocess` function then needs to take in the original `mccase` as well as those outputs. Within that function, you will need to perform any postprocessing you wish to do, and choose what data to log out by calling `MCCase.addOutVal(val)` on those values. The outputs of `preproccess` and `run` both need to be packaged into tuples for unpacking later.

    postprocess( mccase, *run( *preprocess( mccase ) ) )

A progress bar in the terminal will show the progress as the results for all cases are calculated. After it is complete, all the output values will be compiled into `MCOutVar` objects, and you can now interrogate the combined results. Several things which you might want to do are built in:

1) Calculate statistics on the ensemble of input or output values, via the `MCVar.addVarStat` method.

2) Plot the ensemble of results for a value, either on its own or against other values. The `mc_plot` function intelligently chooses how to plot the variables you pass it, whether that data are singular values, timeseries, or 2D/3D vectors. Histograms, cumulative density plots, 2D/3D scatter plots, and 2D/3D line plots are manually callable for more fine grained control. Specific cases can be highlighted, and the statistics calculated in (1) above can be added to the plots as well.

3) Calculate correlations between input and output variables. This shows the linear sensitivity of the output variables to the change in input variables, and can also be plotted.

See the [examples](https://github.com/scottshambaugh/monaco/tree/main/examples) folder on github for some examples you can step through or use as templates. Some details and several more features here have been passed over, and this documentation will be fleshed out in the future. Of note, saving and loading results to file, 'median' cases, running on remote servers, using order statistics, and additional plotting options will need to be explained.
