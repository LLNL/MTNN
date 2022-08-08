# MTNN

MTNN (Multilevel Neural Networks) is a PyTorch-based library for the
application of multilevel algorithms to the training of neural
networks.


## Supported neural network architectures

The algorithms behind MTNN mathematically work whenever the neural
network can be decomposed into a set of operational subsets such that
each subset consists of a set of neurons, channels, or other similar
operational units. In this case, pairs of neurons (or channels or
other operational units) can be matched up and restricted into a
coarse network.

MTNN currently has software support for feedforward neural networks with

-Fully-connected layers

-One or more convolutional layers followed by 0 or more fully-connected layers.

# Installation 

Requires Python 3.6 or up

MTNN Dependencies:

* PyTorch
* Numpy
 * Matplotlib (for plotting results)

The [Anaconda Python
distribution](https://www.anaconda.com/products/individual), free for
individuals, has all of these.

To install MTNN, from the MTNN root, run the command
`pip install -e .`

# Documentation
Additional details about the algorithm and parameter settings can be found in `ABOUT.md`
Documentation can be generated via Doxygen with the commands
`doxygen docs/Doxyfile`

# Examples
## Datasets 
Datasets needed to run the Darcy and Poisson examples can be downloaded from [UC San Diego Library's Digital Collections](https://search.datacite.org/works/10.6075/J0HM58MK), or with the command
`wget https://library.ucsd.edu/dc/object/bb1852369g/_2_1.tar`

Place the data folders in `/examples/datasets`.

## Running examples

There are a number of examples in the `examples` folder which
illustrate how to set up and train hierarchies of neural networks.

-`circle_example.py` is the best starting point, and shows visually
 the regularization benefits of MTNN.
 
-`darcy_example.py` and `poisson_example.py` show using MTNN to
 effective learn function estimation based on PDEs.

-`mnist_example.py` shows using MTNN for a classification task. MTNN
 seems to work best here when the network of interest is squeezed
 between a bigger and a smaller network in the multilevel hierarchy.

## Visualizing results

MTNN includes a visualization script that parses its log files and
plots results using matplotlib. This can be found in
`MTNN/visualize_results.py`.

# Contributions and Code of Conduct 
By agreeing to contribute to MTNN, you agree to abide the rules in Code of Conduct

# Authors 
This work was produced at Lawrence Livermore National Laboratory: 
* Ruipeng Li (li50@llnl.gov)

* Christina Mao (mao6@llnl.gov)

* Colin Ponce (ponce11@llnl.gov)


# Terms of Release 
This project is licensed under the MIT License. Please see LICENSE and NOTICE for details. 
All new contributions must be made under the MIT License. 

`SPDX-License-Identifier: MIT`
`LLNL-CODE-827581`