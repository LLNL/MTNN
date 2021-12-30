# Introduction

MTNN (MulTilevel Neural Networks) is a PyTorch-based tool for the
application of multilevel algorithms to the training of neural
networks.

## Algorithmic overview

See the paper at **[INSERT ARXIV LINK]** for a complete description of
the algorithms behind this software.

Multilevel methods, originally devloped for iterative methods for
solving linear systems of equations, work by constructing a sequence
of increasingly-"coarse" analogues to your problem of interest, each
coarse analogue a smaller (fewer variables) version of the
immediately-previous one. This sequence is typically called the
multilevel *hierarchy*.

Multilevel methods proceed by performing a bit of work on the original
problem (called *smoothing*), then passing learned information to
the next-coarser problem in the hierarchy (passing information to a
coarser level is called *restriction*), performing a bit of work on
that one (smoothing again), then passing information to the
next-coarser problem, and so on, until reaching the coarsest problem
in the hierarchy. The method then reverses the process: perform a bit
of work (smoothing) on the coarsest problem, then pass learned
information to the next-finer problem (passing information to a finer
level is called *prolongation*) where a bit of work is performed,
and so on, until reaching the original problem again. This entire
sequence is called a *V-Cycle*.

Simple code for an iterative V-Cycle solver might look like the
following:
```
for j in range(num_cycles):
    for level_ind, level in hierarchy[:-1]:
        level.smooth(problem_data)
        level.restrict(level.next_coarser_level)

    hierarchy[-1].smooth(problem_data)

    for level_ind, level in reversed(enumerate(hierarchy[:-1])):
        level.prolongate(level.next_coarser_level)
        level.smooth(problem_data)
```

The intuition behind this is that the "big idea" can be efficiently
learned at the coarse levels, with the details refined at the fine
levels. We have found that, in the case of neural networks, the value
of multilevel methods is not necessarily computational speed, but
regularization: Training an entire hierarchy of neural networks
together forces a neural network to be "similar" to those above and/or
below it in the hierarchy. Therefore, it learns to be a refined
version of a smaller network that may capture the big ideas but not
the details, and/or to be a better-regularized version of a bigger
network that may be prone to overfitting. MTNN does not replace
existing regularization methods, which remain crucial, but can
regularize in ways that existing methods do not, and so is a useful
addition to our toolbox of regularization methods.

Multilevel methods are not algorithms, must **algorithmic frameworks**
in which one must make individual algorithmic choices for various
components; we have provided a number of choices here, and designed
the code in a modular way so that you can (hopefully) easily write and
insert your own algorithmic components.

## Some algorithmic component choices

### Smoothing

Smoothing is typically just a traditional neural network optimizer
applied to that level's neural network. We've used SGD here, but one
could certainly use other optimizers.

### Restriction

We have provided one implementation of a restriction method here. In
our restriction, a coarse neural network has the same architecture as
the original, but with fewer neurons per layer. In the case of CNNs,
the coarse network has fewer channels per layer.

We use a Heavy Edge Matching procedure to identify pairs of similar
neurons (or channels) in a layer. Upon restriction, we calculate a
weighted average of the parameters of the two neurons to produce a
single "coarse neuron."

The weighted averaging is linear in the parameters of the
network. Thus, the operation can be represented as a matrix $R$, and
restriction computes a set of coarse network parameters $N_c$ from an
original network $N$ as

![equation](http://www.sciweavers.org/tex2img.php?eq=N_c%20%5Cleftarrow%20R%20N&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

### Prolongation

Prolongation is simply the reverse operation of restriction, and a
linear operator $P$ converts from coarse to fine networks. However,
the image of $P$ only lies on a subspace of the original parameter
space, and we want to maintain the selection on the orthogonal
subspace prior to restriction. Therefore, we update as follows:

![equation](http://www.sciweavers.org/tex2img.php?eq=N%20%5Cleftarrow%20N%20%2B%20P%28N_c%20-%20RN%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

### Tau correction

We leverage here a variant of the multilevel framework known as the
*Full Approximation Scheme*, originally developed for solving
nonlinear system of equations. The *tau correction* is a linear
modification to the coarse loss:

![equation](http://www.sciweavers.org/tex2img.php?eq=%5Ctilde%20L_c%28NN_c%29%20%3D%20L_c%28NN_c%29%20%2B%20%5Ctau%5ET%20NN_c&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

As we show in our ppaper, the linear modifier $\tau$ is chosen to
replace the coarse gradient with a restricted version of the fine
gradient. This has the effect of coarse-level training proceeding more
like fine-level training, at least at first before the higher-order
derivatives at the coarse level take over.

We provide a few different tau correction options:

-`none` Does not use a tau correction. This forces the training
 procedure to seek out a point in which both the fine and coarse
 networks have small gradients and a positive-definite Hessian.

-`wholeset` or `minibatch` are two provided variants of using a tau
 correction. Because it replaces the coarse gradient, this is weaker
 regularization than `none`. It seeks out a point in which both fine
 and coarse networks have positive-definite Hessians, but only
 requires that the fine network have a small gradient.

Note that `wholeset` and `minibatch` are only to be used when the
neural network to be used is the finest in the hierarchy; the loss
modifications at the coarser levels tend to shift those networks into
modes that are suboptimal if one were to use them on their own. Using
no tau correction (`none`) produces an entire hierarchy of
potentially-useful neural networks of varying sizes.

## Supported neural network architectures

The algorithms behind MTNN mathematically work whenever the neural
network can be decomposed into a set of operational subsets such that
each subset consists of a set of neurons, channels, or other similar
operational units. In this case, pairs of neurons (or channels or
other operational units) can be matched up and restricted into a
coarse network.

MTNN currently has software support for feedforward neural neural networks with

-Fully-connected layers

-One or more convolutional layers followed by 0 or more fully-connected layers.

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

# Examples

## Running examples

There are a number of examples in the `examples` folder which
illustrate how to set up and train hierarchies of neural networks.

-`circle_example.py` is the best starting point, and shows visually
 the regularization benefits of MTNN.
 
-`darcy_example.py` and `poisson_example.py` show using MTNN to
 effective learn function estimation based on PDEs.

-`mnist_example.py` shows using MTNN for a classifcation task. MTNN
 seems to work best here when the network of interest is squeezed
 between a bigger and a smaller network in the multilevel hierarchy.

## Visualizing results

MTNN includes a visualization script that parses its log files and
plots results using matplotlib. This can be found in
`MTNN/visualize_results.py`.

# Authors

-Ruipeng Li (li50@llnl.gov)

-Christina Mao (mao6@llnl.gov)

-Colin Ponce (ponce11@llnl.gov)