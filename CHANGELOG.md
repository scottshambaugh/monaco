# Changelog

## Future Work
### Definitely
- 2D Scatter Statistics
- Migrate to dask as parallel processing backend
- Run on remote server (AWS, etc)
- Get custom distributions working
- Polynomial approximations
- Tornado plots
- Sensitivity Indices (Sobol's Method)
- Get working in jupyter notebooks
- Color plots by 3rd variable
- Strict typing
- Bootstrap statistics
### Maybe
- Example: Evidence-Based Scheduling
- 2D/3D Line statistics
- Correlation matrix input
- Ability to plot derived data in addition to mcvars
- Star Sampling
- Variogram Sensitivity Analysis

----

## [Unreleased]
### Added    
### Changed    
### Removed    


## [0.2.2] - 2021-12-07
### Added    
* Python 3.7 support (>=3.7.1 to match pandas)
### Changed    
* `MCVal.nums` were made numpy arrays
* `MCVal`s have a `shape` attribute rather than `size`, to match with numpy
* `helper_functions.get_tuple()` changed to `get_list()`
### Removed    


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
* Added mc_multi_plot_2d_scatter_grid()
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
