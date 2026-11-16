========
gMeterPy
========

.. image:: https://img.shields.io/pypi/v/gmeterpy.svg
        :target: https://pypi.org/project/gmeterpy/
        :alt: Latest version on PyPI

.. image:: https://github.com/thegeodesy/gmeterpy/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/thegeodesy/gmeterpy/actions/workflows/ci.yml
        :alt: GitHub Actions build status

.. image:: https://img.shields.io/pypi/pyversions/gmeterpy.svg
        :alt: Compatible Python versions
        :target: https://pypi.org/project/gmeterpy/

.. image:: https://readthedocs.org/projects/gmeterpy/badge/?version=latest
        :target: https://gmeterpy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Gravity (small `g`) measurement data processing framework written in Python.

The goal of the gMeterPy is to create an open source software to process 
raw data from all types of gravimeters.

**The project is under active development; APIs may change between releases.**

About
-----

*gMeterPy* is an open source and free Python library for processing 
measurement data from gravimeters, which are used for gravity field surveys 
in geodesy, geophysics and other branches of Earth and planetary sciences. 
The main intention is to go as deep as possible in processing **raw** data 
from scratch, as well as to support all types of instruments, 
corrections, common file formats and processing procedures.

Install
-------

You will need **Python 3.12 or greater**.

Install the current source version:

.. code:: bash

    python -m pip install "git+https://github.com/thegeodesy/gmeterpy@main"

Or clone the git repository locally and run:

.. code:: bash

    git clone https://github.com/thegeodesy/gmeterpy.git
    cd gmeterpy
    python -m pip install .

The latest compatible published release is available with
``python -m pip install gmeterpy``. Published releases can differ from the
current source in APIs, commands and Python requirements.

The current source installs ``gabl``, ``gphone``, ``loop``, ``scx2`` and
``vgfit``. Run each command with ``--help`` to inspect its options.

Citation
--------

Citation metadata is available in ``CITATION.cff`` and through the repository's
"Cite this repository" menu.

Contributing
------------

There are many ways you can contribute! Please read 
`CONTRIBUTING.rst <https://github.com/thegeodesy/gmeterpy/blob/main/CONTRIBUTING.rst>`__
to see how you can help and give feedback.

Please note also that *gMeterPy* is released with a Code of Conduct 
(see `CODE_OF_CONDUCT.md <https://github.com/thegeodesy/gmeterpy/blob/main/CODE_OF_CONDUCT.md>`__).
By participating in this project you agree that you will follow all its terms.

License
-------

This project is licensed under the MIT license. 
Please see `LICENSE.txt <https://github.com/thegeodesy/gmeterpy/blob/main/LICENSE.txt>`__
for details.
