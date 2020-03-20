#!/usr/bin/env python

from setuptools import setup, find_packages,find_namespace_packages

setup(name="MTNN",
      version="0.1",
      description="Package to develop and test multigrid algorithms on neural networks",
      author="Christina Mao",
      author_email="mao6@llnl.gov",
      packages=find_packages(),
      namespace_packages=find_namespace_packages(),
      package_dir={' ': 'MTNN'},
      install_requires=['PyYAML',
                        'torch',
                        'pytest',
                        'tensorboard',
                        'scikit-learn'],
      python_requires='>=3.7'
      )
