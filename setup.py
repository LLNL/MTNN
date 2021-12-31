#!/usr/bin/env python

from setuptools import setup, find_packages, find_namespace_packages

setup(name="MTNN",
      version="1.0",
      description="Multilevel methods for training neural networks",
      packages=find_namespace_packages(),
      package_dir={' ': 'MTNN'},
      install_requires=['torch',
                        'torchvision',
                        'numpy',
                        'matplotlib'],
      python_requires='>=3.6'
      )
