#!/usr/bin/env python

import codecs
from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='lroptimizer',
      version='0.0.0',
      description='Package for optimizing the erfgau LR term.',
      long_description=long_description,
      url='https://github.com/quantumelephant/olsens',
      license='GNU Version 3',
      author='Cristina E. Gonzalez-Espinoza',
      author_email='gonzalce@mcmaster.ca',
      classifiers=['Development Status :: 1 - Beta',
                   'Intended Audience :: Developers of new methods for Range-Separation'
                   'equation',
                   'Topic :: Range-Separation :: Smooth potentials',
                   'Programming Language :: Python :: 2.7'],
      keywords='wavefunction hamiltonian optimization',
      package_dir={'optimizer': 'optimizer'},
      packages=['optimizer', 'optimizer.minimizer'],
      # test_suite='nose.collector',
      install_requires=['numpy', 'scipy', 'gmpy2', 'LibInt', 'LibXC'],
      extras_require={'horton': [], 'fanci': [], 'pyci' : []},
      tests_requires=['nose', 'cardboardlint'],
      package_data={},
      data_files=[],
      )
