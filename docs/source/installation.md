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
poetry install --extras "pandas"
```

## Running Tests and Type Checking

```
poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
poetry run mypy src
```

## Building Docs

ReadTheDocs will automatically [build](https://readthedocs.org/projects/monaco/builds/) when the `main` branch is updated.
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
4) Run tests, type checking, and linting locally
    ```
    poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
    poetry run mypy src
    poetry run flake8
    ```
5) Commit any changes and push up the develop branch
6) Push all changes up to the main branch
    ```
    git checkout main
    git merge develop
    git push
    ```
7) Wait for [CI tests](https://github.com/scottshambaugh/monaco/actions) to pass
8) Check that the [docs are building](https://readthedocs.org/projects/monaco/builds/)
9) [Create a new release](https://github.com/scottshambaugh/monaco/releases), creating a new tag and including a changelog:    
    ```
    **Changelog**: https://github.com/scottshambaugh/monaco/blob/main/CHANGELOG.md    
    **Full Diff**: https://github.com/scottshambaugh/monaco/compare/v0.x.x...v0.x.x
    ```
10) Build wheels: `poetry build`
11) [Publish to PyPi](https://pypi.org/project/monaco/): `poetry publish`
12) Wait 10 minutes to check that the package has updated
