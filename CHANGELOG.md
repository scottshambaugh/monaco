# Changelog

## Future Work
### Features & Maintenance:
- 2D Scatter Statistics
- Regressions (linear, poly, custom)
- Contribution to sample mean and contribution to sample variance
- Get custom distributions working
- Tornado plots (?)
- Variable contribution plots
- Example: Evidence-Based Scheduling
- 2D/3D Line statistics
- Correlated inputs
- Switch from prints to logging
- Ability to plot derived data in addition to vars
- Star Sampling
- Demo running on remote server (AWS, etc)
- Variogram Sensitivity Analysis
- repr's for all main objects
- Histogram/spaghetti plot coloring a-la [aleatory](https://github.com/quantgirluk/aleatory)
### Known Bugs:
- Cannot plot a varstat in percentile space
----

## [Unreleased]
### Added    
* `case.vars` dict to reference all invars and outvars, and `case.vals` dict to reference all invals and outvals
* `__getitem__` for cases to get a val by name, e.g. `case['valname']`
* `__getitem__` for sims to get a case by number or a var by name, e.g. `sim[i]` or `sim['varname']`
* `__getitem__` for vars to get a val by number, e.g. `var[i]`
* `datasource` attribute on invals and outvals to tell if they were loaded from file
### Changed    
* External requests blocked during testing
* `if __name__ == '__main__'` blocks excluded from code coverage percentage
### Removed    
* git `develop` branch, only using `main` from now on

## [0.15.1] - 2025-02-04
### Added    
* `cases` kwarg to `addVarStat` and `calcSensitivities` to allow specifying which cases to use to calculate the stat ([GH-14](https://github.com/scottshambaugh/monaco/issues/14))

## [0.15.0] - 2025-02-03
### Added    
* `cases` kwarg to `VarStat` to allow specifying which cases to use to calculate the stat ([GH-14](https://github.com/scottshambaugh/monaco/issues/14))
* `cases` kwarg to `dvars_sensitivity` to allow specifying which cases to use for the sensitivity analysis ([GH-14](https://github.com/scottshambaugh/monaco/issues/14))
* `cases` kwarg to `plot_2d_cov_ellipse`, `plot_integration_error`, and `plot_integration_convergence` to allow specifying which cases to use for the plot ([GH-14](https://github.com/scottshambaugh/monaco/issues/14))
### Changed    
* Switch varstat bootstrap method from 'basic' to 'BCa'
* `get_cases` function moved from `mc_plot` to `helper_functions`

## [0.14.1] - 2025-01-31
### Changed    
* More robust valmap generation ([GH-13](https://github.com/scottshambaugh/monaco/issues/13))

## [0.14.0] - 2024-12-22
### Added    
* Python 3.13 support
### Removed    
* Python 3.9 support

## [0.13.2] - 2024-07-11
### Changed    
* Fixed docs build not including docstrings

## [0.13.1] - 2024-06-25
### Added    
* Numpy 2.0 real compatibility & wheels on PyPI

## [0.13.0] - 2024-06-12
### Added    
* Testing for partial failures of cases while running with different `debug` flags ([GH-10](https://github.com/scottshambaugh/monaco/issues/10))
### Changed    
* Support newer versions of `dask.distributed` ([GH-12](https://github.com/scottshambaugh/monaco/pull/12))
### Removed    
* `tqdm_dask_distributed` supporting function, replaced by `dask.distributed.progress`

## [0.12.1] - 2024-03-19
### Added    
* Numpy 2.0 code compatibility attempt

## [0.12.0] - 2024-03-03
### Added    
* Allow specifying `dists` and `distkwargs` for invars loaded from file
### Changed    
* Fix cases not being populated with invars and invals when loading from file ([GH-9](https://github.com/scottshambaugh/monaco/issues/9))
* Fix nummaps not being per-variable when loading outvars from file

## [0.11.7] - 2024-02-19
### Changed    
* Fix histogram not plotting for imported invars ([GH-8](https://github.com/scottshambaugh/monaco/issues/8))

## [0.11.6] - 2024-02-19
### Changed    
* Fix broken docs build

## [0.11.2] - [0.11.5] - 2024-02-19
### Changed    
* Automated github publishing to PyPI
* Update github action runners


## [0.11.1] - 2024-02-19
### Changed    
* Fix underflow issue in dvars ([GH-7](https://github.com/scottshambaugh/monaco/issues/7))


## [0.11.0] - 2023-11-05
### Added    
* Python 3.12 support
### Changed    
* Documentation cleanup
* Halton sequence seeded draws changed with scipy 1.11, see scipy issue #18079
### Removed    
* Python 3.8 support


## [0.10.0] - 2023-04-23
### Added    
* More plotting test coverage.
* Allow passing in varname strings to `sim.plot`.
### Changed    
* License changed from GPLv3 to more permissive MIT (ok because single author project).
* Fix `sim.plot` scalarvars not being used ([GH-6](https://github.com/scottshambaugh/monaco/issues/6))
### Removed    


## [0.9.1] - 2023-02-16
### Added    
* Python 3.11 support
### Changed    
* `numba` is now an optional dependency
### Removed    


## [0.9.0] - 2023-02-08
### Added    
* Automated testing for plots and multiplots.
### Changed    
* Parallel processing now chains preprocessing, running, and postprocessing into
a single dask task graph via Sim.executeAllFcns(), giving large speed boost
* Sim functions that take in cases default to None (all cases)
* Fixed bug plotting histograms
### Removed    


## [0.8.3] - 2022-07-13
### Added    
### Changed    
* Fixed another bug generating and plotting varstat ranges from mixed-length data
### Removed    


## [0.8.2] - 2022-07-13
### Added    
### Changed    
* Fixed bug generating and plotting percentile ranges from mixed-length data
### Removed    


## [0.8.1] - 2022-07-13
### Added    
* `mc_multi_plot.multi_plot_grid_rect` plotting, made default for `mc.plot`
* `__repr__` for Cases and Vals
### Changed    
* `mc_multi_plot.multi_plot_2d_scatter_grid` renamed to `multi_plot_grid_tri`
* Fixed bug when plotting against simulation steps of different lengths
* Make Sims singlethreaded by default
### Removed    

## [0.8.0] - 2022-07-09
### Added    
* Export outvar nums to a csv or json file with `sim.exportOutVars`
* Import invars from a csv or json with `sim.importInVars`
* Optionally plot a contour underneath a 2D scatter plot with a third variable
* Flags to keep siminput and simrawoutput for each case (default True)
* Added Sim.vars to reference all invars and outvars
* Variance, skewness, kurtosis, and moment added as varstats
### Changed    
* `sim.exportInVarNums` renamed to `sim.exportInVars`
* `sim.importOutVals` renamed to `sim.importOutVars`
* Refined and removed upper triangle of multiplot grid
### Removed    

## [0.7.0] - 2022-06-13
### Added    
* Choose whether to sort sensisitivities while plotting.
* Added `Sim.plot()` method
* Export invar nums to a csv or json file with `sim.exportInVarNums`
* Import outvals from a csv or json with `sim.importOutVals`, and convert to outvars
* 2.5D plotting (scalar vs vectors)
### Changed    
* Prevent overwriting existing InVars or OutVals with an already used name.
* Do not save sim and case data by default
* Fix splitting pairs of variables when plotting
### Removed    

## [0.6.0] - 2022-05-09
### Added    
* DVARS fleshed out, moved out of alpha
* Can now plot sensitivity indices and ratios
### Changed    
### Removed    

## [0.5.0] - 2022-05-01
### Added    
* Added `singlethreaded` kwarg to `Sim` initialization.
* Added `daskkwargs` kwarg to `Sim` initialization.
* Added `percentile` varstat.
### Changed    
* For all datafiles, switch from `dill` to `cloudpickle` for pickling.
* Parallel processing backend moved from `pathos` to `dask`.
* `Sim.PreProcessCase`, `Sim.RunCase`, and `Sim.PostProcessCase` broken out to
`case_runners.py` functions.
### Removed    
* Removed `cores` kwarg from `Sim` initialization.
* Removed python 3.7 support, to align with dask.

## [0.4.0] - 2022-04-22
### Added    
* Plots of all the example statistical distributions
* Bootstrapping confidence interval for a VarStat
* Plot bootstrapped confidence intervals
* Copy of template in jupyter notebook format
### Changed    
* Change color palette to match matplotlib defaults
* Discrete InVar plots show stem pmfs
### Removed    

## [0.3.1] - 2022-01-29
### Added    
* Vars get their own `plot()` method as a shorthand
* Baseball example
### Changed    
### Removed    
* Rocket example

## [0.3.0] - 2022-01-22
### Added    
* Experimental D-VARS sensitivity analysis
* `Sim.scalarOutVars()` dict and `Sim.noutvars`
* You can specify a custom varstat
* You can plot plot in percentile space rather than nums
### Changed    
* Removed mc_ and MC prefixes from all functions and classes

## [0.2.3] - 2021-12-18
### Added    
* Python 3.7.0 support (pandas testing made optional)
### Changed    
* All functions are now imported into top-level package, so you don't need to dig through modules

## [0.2.2] - 2021-12-07
### Added    
* Python 3.7 support (>=3.7.1 to match pandas)
### Changed    
* `Val.num`s were made numpy arrays
* `Val`s have a `shape` attribute rather than `size`, to match with numpy
* `helper_functions.get_tuple()` changed to `get_list()`
* Simplify code for extracting valmaps
* More typing

## [0.2.1] - 2021-12-02
### Added    
* Python 3.8 support
* Pre-commit hooks for linting
### Changed    
* Dusted off the code with lots of linting
* Made all modules lowercase
* pandas is now an optional dependency

## [0.2.0] - 2021-11-30
### Added    
* Documentation is now [up on readthedocs](https://monaco.readthedocs.io/en/latest/)!
* Docstrings for all classes and functions, roughly follows the [numpy docstrings convention](https://numpydoc.readthedocs.io/en/latest/format.html)
* Analysis process diagram
* More unit tests
### Changed    
* Documentation and images were moved around.
### Removed    

## [0.1.2] - 2021-11-15
### Added    
* Project logo
* Added multi_plot_2d_scatter_grid()
* Python 3.10 support
### Changed
* `import monaco` now imports all modules
* The 'nom' case is changed to the 'median' case
### Removed    
* No more inline tests

## [0.1.1] - 2021-10-20
### Changed
* Documentation updates
* Better type consistency
* Testing on more platforms
* Small bugfixes

## [0.1.0] - 2021-10-04
### Added
* Initial release!
