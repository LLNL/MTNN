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
multilevel **hierarchy*.

Multilevel methods proceed by performing a bit of work on the original
problem (called **smoothing**), then passing learned information to
the next-coarser problem in the hierarchy (passing information to a
coarser level is called **restriction**), performing a bit of work on
that one (smoothing again), then passing information to the
next-coarser problem, and so on, until reaching the coarsest problem
in the hierarchy. The method then reverses the process: perform a bit
of work (smoothing) on the coarsest problem, then pass learned
information to the next-finer problem (passing information to a finer
level is called **prolongation**) where a bit of work is performed,
and so on, until reaching the original problem again. This entire
sequence is called a **V-Cycle**.

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
together forces a neural network to be "similar" to those above and
below it in the hierarchy. Therefore, it learns to be a refined
version of a smaller network that may capture the big ideas but not
the details, and also to be a better-regularized version of a bigger
network that may have overfit.

Multilevel methods are algorithmic *frameworks* in which one must make
individual algorithmic choices for various components; we have
provided a number of choices here, and designed the code in a modular
way so that you can (hopefully) easily write and insert your own
algorithmic components.

## Some algorithmic component choices

### Smoothing

Smoothing is typically just a traditional neural network optimizer
applied to that level's neural network. We've used SGD here, but one
could certainly use other optimizers.

### Restriction

### Prolongation


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

There are a number of examples in the `examples` folder which
illustrate how to set up and train hierarchies of neural networks.

-`circle_example.py` is the best starting point, and shows visually
 the regularization benefits of MTNN.
 
-`darcy_example.py` and `poisson_example.py` show using MTNN to
 effective learn function estimation based on PDEs.

-`mnist_example.py` shows using MTNN for a classifcation task. MTNN
 seems to work best here when the network of interest is squeezed
 between a bigger and a smaller network in the multilevel hierarchy.
