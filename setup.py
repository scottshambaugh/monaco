# flake8: noqa
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['monaco']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.3.0,<4.0.0',
 'numpy>=1.21,<2.0',
 'pathos>=0.2.7,<0.3.0',
 'psutil>=5.0.0,<6.0.0',
 'scipy>=1.7,<2.0',
 'tqdm>=4.0,<5.0']

extras_require = \
{'docs': ['sphinx>=4.2,<5.0',
          'sphinx_rtd_theme>=1.0,<2.0',
          'myst-parser>=0.15'],
 'pandas:python_full_version >= "3.7.1"': ['pandas>=1.3,<2.0']}

setup_kwargs = {
    'name': 'monaco',
    'version': '0.2.2',
    'description': 'Quantify uncertainty and sensitivities in your computer models with an industry-grade Monte-Carlo framework.',
    'long_description': '<p float="center" align="center">\n<img width="570" height="150" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/docs/images/monaco_logo.png">  \n</p>\n\n![Release](https://img.shields.io/github/v/release/scottshambaugh/monaco?sort=semver)\n![Builds](https://github.com/scottshambaugh/monaco/actions/workflows/builds.yml/badge.svg)\n![Tests](https://github.com/scottshambaugh/monaco/actions/workflows/tests.yml/badge.svg)\n[![Docs](https://readthedocs.org/projects/monaco/badge/?version=latest)](https://monaco.readthedocs.io/en/latest/?badge=latest)\n![PyPI - Python Version](https://img.shields.io/pypi/pyversions/monaco)\n\nQuantify uncertainty and sensitivities in your computer models with an industry-grade Monte-Carlo framework.\n\n### Overview\n\nAt the heart of all serious forecasting, whether that be of elections, the spread of pandemics, weather, or the path of a rocket on its way to Mars, is a statistical tool known as the [Monte-Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The Monte-Carlo method, named for the rolling of the dice at the famous Monte Carlo casino located in Monaco, allows you to quantify uncertainty by introducing randomness to otherwise deterministic processes, and seeing what the range of results is.\n\n<p float="left" align="center">\n<img width="500" height="250" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/docs/images/analysis_process.png">\n</p>\n\n`monaco` is a python library for analyzing uncertainties and sensitivities in your computational models by setting up, running, and analyzing a Monte-Carlo simulation wrapped around that model. Users can define random input variables drawn using chosen sampling methods from any of SciPy\'s continuous or discrete distributions (including custom distributions), preprocess and structure that data as needed to feed to their main simulation, run that simulation in parallel anywhere from 1 to millions of times, and postprocess the simulation outputs to obtain meaningful, statistically significant conclusions. Plotting and statistical functions specific to use cases that might be encountered are provided, and repeatability of results is ensured through careful management of random seeds.\n\n<p float="left" align="center">\n<img width="293.08" height="270" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/examples/rocket/rocket_trajectory.png">  \n<img width="384.94" height="270" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/examples/rocket/wind_vs_landing.png">\n</p>\n\n### Quick Start\nFirst, install `monaco`:\n```\npip install monaco\n```\nThen, copy the two files from the [template directory](https://github.com/scottshambaugh/monaco/tree/main/template), which contains a simple, well commented Monte Carlo simulation of flipping coins. That link also contains some exercises for you to do, to help you familiarize yourself with how `monaco` is structured.\n\nAfter working through the template exercises, check out the other [examples](https://github.com/scottshambaugh/monaco/tree/main/examples) for inspiration and more in-depth usage of `monaco`\'s features.\n\n### Documentation / API Reference\nDocumentation is being built up - read the docs here: https://monaco.readthedocs.io\n\nCurrently there is a complete [API reference](https://monaco.readthedocs.io/en/latest/api_reference.html), more detailed [installation, test, and publishing](https://monaco.readthedocs.io/en/latest/installation.html) instructions, an overview of the [basic architecture](https://monaco.readthedocs.io/en/latest/basic_architecture.html) and [basic workflow](https://monaco.readthedocs.io/en/latest/basic_workflow.html), and some details on [statistical distributions](https://monaco.readthedocs.io/en/latest/statistical_distributions.html) and [sampling methods](https://monaco.readthedocs.io/en/latest/sampling_methods.html). \n\n### License / Citation\nCopyright 2020-2021 Scott Shambaugh, distributed under [the GPLv3.0 (or later) license](LICENSE.md).    \nIf you use `monaco` to do research that gets published, please cite [the monaco github page](https://github.com/scottshambaugh/monaco).\n\n### Further Reading\n* [Hanson, J. M., and B. B. Beard. "Applying Monte Carlo simulation to launch vehicle design and requirements analysis." National Aeronautics and Space Administration, Marshall Space Flight Center, 1 September 2010.](https://ntrs.nasa.gov/citations/20100038453)\n* [Razavi, S. et. al. "The future of sensitivity analysis: an essential discipline for systems modeling and policy support." Environmental Modelling & Software Volume 137, March 2021.](https://www.sciencedirect.com/science/article/pii/S1364815220310112)\n* [Satelli, A. et. al. "Why so many published sensitivity analyses are false: A systematic review of sensitivity analysis practices." Environmental Modelling & Software Volume 114, April 2019.](https://www.sciencedirect.com/science/article/pii/S1364815218302822)\n',
    'author': 'Scott Shambaugh',
    'author_email': 'scott@theshamblog.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/scottshambaugh/monaco/',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.7.0,<3.11',
}


setup(**setup_kwargs)
