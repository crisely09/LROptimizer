#!/usr/bin/env python
# -*- coding: utf-8 -*-
# An experimental local optimization package 
# Copyright (C) 2018 Cristina E. Gonzalez-Espinoza <gonzalce@mcmaster.ca>
#
# This file is part of LROptimizer.
#
# LROptimizer is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# LROptimizer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""LROptimizer setup script.

If you are not familiar with setup.py, just use pip instead:

    pip install Flik --user --upgrade

Alternatively, you can install from source with

    ./setup.py install --user
"""

from __future__ import print_function

from setuptools import setup

def get_version():
    """Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable defining the
    version string with single quotes.

    """
    with open('Flik/version.py', 'r') as f:
        return f.read().split('=')[-1].replace('\'', '').strip()


def readme():
    """Load README.rst for display on PyPI."""
    with open('README.rst') as f:
        return f.read()


setup(
    name='LROptimizer',
    version=get_version(),
    description='Optimizer of Long-Range model Hamiltonians.',
    long_description=readme(),
    author='Cristina E. Gonz\'alez-Espinoza',
    author_email='gonzalce@mcmaster.ca',
    url='https://github.com/crisely09/lroptimizer',
    packages=['LROptimizer'],
    zip_safe=False,
)
