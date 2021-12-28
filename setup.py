#!/usr/bin/env python

from setuptools import setup, find_packages,find_namespace_packages

setup(name="MTNN",
      version="0.1",
      description="Multilevel methods for training neural networks",
      packages=find_packages(),
      namespace_packages=['MTNN'],
      package_dir={' ': 'MTNN'},
      install_requires=['torch',
                        'torchvision',
                        'pytest',
                        'onnx',
                        'numpy'],
      python_requires='>=3.6'
      )
