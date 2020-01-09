#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name = "MTNN",
      version = "0.1",
      description = "Module to test multigrid training on neural networks",
      author = "",
      author_email = "",
      #install_requires=
      packages = find_packages(), install_requires = ['PyYAML', 'torch', 'pytest', 'scikit-learn']
      )
