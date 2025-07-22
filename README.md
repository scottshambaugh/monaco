<p float="center" align="center">
<img width="570" height="150" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/docs/images/monaco_logo.png">  
</p>

![Release](https://img.shields.io/github/v/release/scottshambaugh/monaco?sort=semver)
[![Downloads](https://static.pepy.tech/badge/monaco)](https://pepy.tech/project/monaco)
![Builds](https://github.com/scottshambaugh/monaco/actions/workflows/builds.yml/badge.svg)
![Tests](https://github.com/scottshambaugh/monaco/actions/workflows/tests.yml/badge.svg)
[![Docs](https://readthedocs.org/projects/monaco/badge/?version=latest)](https://monaco.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/scottshambaugh/monaco/graph/badge.svg?token=FAL47REO7V)](https://codecov.io/gh/scottshambaugh/monaco)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/monaco)

Quantify uncertainty and sensitivities in your computer models with an industry-grade Monte Carlo library.

### Overview

At the heart of all serious forecasting, whether that be of elections, the spread of pandemics, weather, or the path of a rocket on its way to Mars, is a statistical tool known as the [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The Monte Carlo method, named for the rolling of the dice at the famous Monte Carlo casino located in Monaco, allows you to quantify uncertainty by introducing randomness to otherwise deterministic processes, and seeing what the range of results is.

<p float="left" align="center">
<img width="500" height="250" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/docs/images/analysis_process.png">
</p>

`monaco` is a python library for analyzing uncertainties and sensitivities in your computational models by setting up, running, and analyzing a Monte Carlo simulation wrapped around that model. Users can define random input variables drawn using chosen sampling methods from any of SciPy's continuous or discrete distributions (including custom distributions), preprocess and structure that data as needed to feed to their main simulation, run that simulation in parallel anywhere from 1 to millions of times, and postprocess the simulation outputs to obtain meaningful, statistically significant conclusions. Plotting and statistical functions specific to use cases that might be encountered are provided, and repeatability of results is ensured through careful management of random seeds.

<p float="left" align="center">
<img width="350" height="350" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/examples/baseball/baseball_trajectory.png">
<img width="440" height="330" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/examples/baseball/launch_angle_vs_landing.png">
</p>

### Quick Start
First, install `monaco`:
```
pip install monaco
```
Then, copy the two files from the [template directory](https://github.com/scottshambaugh/monaco/tree/main/template), which contains a simple, well commented Monte Carlo simulation of flipping coins. That link also contains some exercises for you to do, to help you familiarize yourself with how `monaco` is structured.

After working through the template exercises, check out the other [examples](https://github.com/scottshambaugh/monaco/tree/main/examples) for inspiration and more in-depth usage of `monaco`'s features.

### Documentation / API Reference / SciPy 2022 Talk

The documentation at https://monaco.readthedocs.io includes:

- A complete [API reference](https://monaco.readthedocs.io/en/latest/api_reference.html)
- More detailed [installation, test, and publishing](https://monaco.readthedocs.io/en/latest/installation.html) instructions
- An overview of the [basic architecture](https://monaco.readthedocs.io/en/latest/basic_architecture.html) and [basic workflow](https://monaco.readthedocs.io/en/latest/basic_workflow.html)
- Information on [single-threaded and parallel computing](https://monaco.readthedocs.io/en/latest/processing_methods.html)
- Some details on [statistical distributions](https://monaco.readthedocs.io/en/latest/statistical_distributions.html) and [sampling methods](https://monaco.readthedocs.io/en/latest/sampling_methods.html)

Monaco was presented at the SciPy 2022 Conference, and the conference resources should give another good overview of the library. Check out [the paper](http://conference.scipy.org.s3-website-us-east-1.amazonaws.com/proceedings/scipy2022/pdfs/scott_shambaugh.pdf), [the video of the talk](https://www.youtube.com/watch?v=yB539OIol_s), and [the talk's slides and notebooks](https://github.com/scottshambaugh/monaco-scipy2022).

### License / Citation
Copyright 2020 Scott Shambaugh, distributed under [the MIT license](LICENSE.md).

If you use `monaco` to do research that gets published, please cite the conference paper using the below or [monaco.bib](monaco.bib):    
> W. Scott Shambaugh (2022). Monaco: A Monte Carlo Library for Performing Uncertainty and Sensitivity Analyses. *In Proceedings of the 21st Python in Science Conference* (pp. 202 - 208).

### Further Reading
* [Hanson, J. M., and B. B. Beard. "Applying Monte Carlo simulation to launch vehicle design and requirements analysis." National Aeronautics and Space Administration, Marshall Space Flight Center, 1 September 2010.](https://ntrs.nasa.gov/citations/20100038453)
* [Razavi, S. et. al. "The future of sensitivity analysis: an essential discipline for systems modeling and policy support." Environmental Modelling & Software Volume 137, March 2021.](https://www.sciencedirect.com/science/article/pii/S1364815220310112)
* [Satelli, A. et. al. "Why so many published sensitivity analyses are false: A systematic review of sensitivity analysis practices." Environmental Modelling & Software Volume 114, April 2019.](https://www.sciencedirect.com/science/article/pii/S1364815218302822)
