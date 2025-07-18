# Installation, Testing, & Publishing

## Basic Installation

```
pip install monaco
```

## Installing from Source

```
git clone https://github.com/scottshambaugh/monaco.git
cd monaco
pip install poetry
poetry install --extras "pandas numba distributed"
```

## Running Tests and Type Checking

```
poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
poetry run mypy src tests/test*
```

## Profiling

Edit the example to remove the plotting functions before profiling. Then upload the json output to [speedscope.app](https://www.speedscope.app/) to view the report.
```
pip install pyspy
py-spy record -o election.speedscope.json --format speedscope -- python examples/election/election_example_monte_carlo_sim.py
```

## Building Docs

ReadTheDocs will automatically [build](https://readthedocs.org/projects/monaco/builds/) when the `main` branch is updated.
```
pip install sphinx sphinx_rtd_theme myst_parser
cd docs
poetry run make clean && poetry run make html
```

## Releasing a New Version and Publishing to PyPI

1) Update `CHANGELOG.md`
2) Update the version in `pyproject.toml`
3) Update and install the package
    ```
    poetry update
    poetry install --extras "pandas numba distributed"
    ```
4) Run tests, type checking, and linting locally
    ```
    poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
    poetry run mypy src tests/test*
    poetry run flake8
    ```
5) Run plotting tests manually
6) Commit any changes and push up the main branch
7) Wait for [CI tests](https://github.com/scottshambaugh/monaco/actions) to pass
8) Check that the [docs are building](https://readthedocs.org/projects/monaco/builds/)
9) [Create a new release](https://github.com/scottshambaugh/monaco/releases), creating a new tag and including a changelog:    
    ```
    **Changelog**: https://github.com/scottshambaugh/monaco/blob/main/CHANGELOG.md    
    **Full Diff**: https://github.com/scottshambaugh/monaco/compare/v0.x.x...v0.x.x
    ``` 
    This will automatically publish the release to [PyPI](https://pypi.org/project/monaco/).
