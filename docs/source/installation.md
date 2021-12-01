# Installation, Testing, & Publishing

## Basic Installation

```
pip install monaco
```

## Installing from Source

Installing from source and running tests:
```
git clone https://github.com/scottshambaugh/monaco.git
cd monaco
pip install poetry
poetry install
```

## Running Tests and Type Checking

```
poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
poetry run mypy src
```

## Building Docs

Docs will automatically [build](https://readthedocs.org/projects/monaco/builds/) when the `main` branch is updated.
```
cd docs
poetry run make clean && poetry run make html
```

## Publishing to PyPi

1) Update `CHANGELOG.md`
2) Update the version in `pyproject.toml`
3) Update and install the package
    ```
    poetry update
    poetry install
    ```
4) Run tests and type checking locally
    ```
    poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
    poetry run mypy src
    ```
5) Push all changes up to the main branch
    ```
    git checkout main
    git merge develop
    git push
    ```
6) Wait for [CI tests](https://github.com/scottshambaugh/monaco/actions) to pass
7) Check that the [docs are building](https://readthedocs.org/projects/monaco/builds/)
7) [Create a new release](https://github.com/scottshambaugh/monaco/releases), creating a new tag and including a changelog:    
    `**Full Changelog**: https://github.com/scottshambaugh/monaco/compare/v0.x.x...v0.x.x`
8) Build wheels: `poetry build`
9) [Publish to PyPi](https://pypi.org/project/monaco/): `poetry publish`
10) Wait 10 minutes to check that the package has updated
