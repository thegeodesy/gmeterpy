.. _install:

Installing
==========

Current source version
----------------------

The current source requires Python 3.12 or newer. Install it with::

    python -m pip install "git+https://github.com/thegeodesy/gmeterpy@main"

To work from an editable checkout::

    git clone https://github.com/thegeodesy/gmeterpy.git
    cd gmeterpy
    uv venv --python 3.15 .venv
    source .venv/bin/activate
    uv pip install --upgrade -e .

Published releases
------------------

Install the latest compatible release from PyPI with::

    python -m pip install gmeterpy

Published releases can differ from the current source in supported Python
versions, APIs and available commands.

Command-line tools
------------------

The current source installs five commands. Check their options before preparing
instrument input files::

    gabl --help
    gphone --help
    loop --help
    scx2 --help
    vgfit --help

Development checks
------------------

From a checkout, install the development and documentation extras::

    uv pip install --upgrade -e ".[dev,docs]"

Then run::

    .venv/bin/ruff check .
    .venv/bin/ruff format --check .
    .venv/bin/python -m pytest
    .venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
    .venv/bin/python -m build
    .venv/bin/twine check --strict dist/*
    .venv/bin/check-wheel-contents dist/*.whl

The root ``.readthedocs.yaml`` declares the hosted documentation build. Repository
and webhook settings still need to be configured in the documentation service.

Dependencies
------------

Runtime dependencies are declared in ``pyproject.toml`` and installed by pip.
They include NumPy, SciPy, pandas, Matplotlib and Astropy.
