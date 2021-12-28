#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="MTNN",
      version="1.0",
      description="Multilevel methods for training neural networks",
      packages=find_packages(),
      namespace_packages=['MTNN'],
      package_dir={' ': 'MTNN'},
      install_requires=['torch',
                        'torchvision',
                        'numpy',
                        'matplotlib'],
      python_requires='>=3.6'
      )
