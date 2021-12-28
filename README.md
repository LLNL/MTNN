# Introduction

MTNN (MulTilevel Neural Networks) is a PyTorch-based tool for the
application of multilevel algorithms to the training of neural
networks. Multilevel methods are algorithmic *frameworks* in which one
must make individual algorithmic choices for various components; we
have provided a number of choices here, and designed the code in a
modular way so that you can easily write and insert your own
algorithmic components.

See the paper at **[INSERT ARXIV LINK]** for a description of the
algorithms behind this software.

# Installation Instructions

MTNN Dependencies:
-PyTorch
-Numpy
-Matplotlib (for plotting results)

The [Anaconda Python
distribution](https://www.anaconda.com/products/individual), free for
individuals, has all of these.

To install MTNN, from the MTNN root, run the command
`pip install -e .`

# Documentation

Documentation can be generated via Doxygen with the commands
`doxygen gocs/Doxyfile`


# Running examples

There are a number of examples in the `examples` folder.