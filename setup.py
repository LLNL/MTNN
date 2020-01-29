#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name = "MTNN",
      version="0.1",
      description="Package to develop and test multigrid algorithms on neural networks",
      author="Christina Mao",
      author_email="cm@llnl.gov",
      packages=find_packages(),
      package_dir={' ':'MTNN'},
      install_requires=['PyYAML',
			 'torch',
			 'pytest', 
			'scikit-learn'],
      python_requires='>=3.7'
      )
