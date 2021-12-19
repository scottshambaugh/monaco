# Installation, Testing, & Publishing

## Basic Installation

```
pip install monaco
```

## Installing from Source
Shouldn't need this before too long, see discussions here:
https://github.com/python-poetry/poetry-core/pull/182
https://github.com/python-poetry/poetry/discussions/1135
```
git clone https://github.com/scottshambaugh/monaco.git
cd monaco
pip install poetry
poetry install --extras "pandas"
```

## Installing Local Editable

```
poetry build --format sdist
tar -xvf dist/*-`poetry version -s`.tar.gz --wildcards -O '*/setup.py' > setup.py
pip install -e .
```

## Running Tests and Type Checking

```
poetry run coverage run --source=monaco -m pytest && poetry run coverage report -m 
poetry run mypy src tests/test*
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
    poetry run mypy src tests/test*
    poetry run flake8
    ```
5) Run plotting tests manually
6) Commit any changes and push up the develop branch
7) Push all changes up to the main branch
    ```
    git checkout main
    git merge develop
    git push
    ```
8) Wait for [CI tests](https://github.com/scottshambaugh/monaco/actions) to pass
9) Check that the [docs are building](https://readthedocs.org/projects/monaco/builds/)
10) [Create a new release](https://github.com/scottshambaugh/monaco/releases), creating a new tag and including a changelog:    
    ```
    **Changelog**: https://github.com/scottshambaugh/monaco/blob/main/CHANGELOG.md    
    **Full Diff**: https://github.com/scottshambaugh/monaco/compare/v0.x.x...v0.x.x
    ```
11) Build wheels: `poetry build`
12) Publish to PyPi: `poetry publish`
13) Wait 10 minutes to check that [the package](https://pypi.org/project/monaco/) has updated
