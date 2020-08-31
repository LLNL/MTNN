"""
Example of cascadic multigrid with MTNN
"""
from collections import namedtuple
import time

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
# local
from MTNN.core.components import data, models
from MTNN.core.multigrid.operators import smoother, prolongation
from MTNN.core.alg import trainer, evaluator, stopping
import MTNN.core.multigrid.scheme as mg
import MTNN.utils.builder as levels


# Load Data and Model
data = data.MnistData(trainbatch_size=100, testbatch_size=10)
net = models.MultiLinearNet([784, 50, 25, 10], F.relu, F.log_softmax)


# Build Multigrid Hierarchy
prolongation_operator = prolongation.LowerTriangleProlongation(expansion_factor=3)
stopping_measure = stopping.EpochStopper(1)
SGDparams = namedtuple("SGDparams", ["lr", "momentum", "l2_decay"])
optim_params = SGDparams(lr=0.01, momentum=0.00, l2_decay=1e-2)
smoother = smoother.SGDSmoother



mg_levels = levels.build_uniform_levels(num_levels=3,
                                        presmoother = smoother(model=net, loss_fn =nn.CrossEntropyLoss(),
                                                             optim_params=optim_params, stopper=stopping_measure,
                                                             log_interval=10000),
                                        postsmoother = smoother(model=net, loss_fn =nn.CrossEntropyLoss(),
                                                              optim_params=optim_params, stopper=stopping_measure,
                                                              log_interval=10000),
                                        prolongation_operator = prolongation_operator,
                                        restriction_operator = None,
                                        coarsegrid_solver=smoother(model=net, loss_fn =nn.CrossEntropyLoss(),
                                                                   optim_params=optim_params, stopper=stopping_measure,
                                                                   log_interval=10000),

                                        )

mg_scheme = mg.Cascadic(mg_levels)

training_alg = trainer.MultigridTrainer(dataloader=data.trainloader,
                                        verbose=True,
                                        log=False,
                                        save=False,
                                        load=False)
evaluator = evaluator.CategoricalEvaluator()



# Train
print('Starting Training')
start = time.perf_counter()
trained_model = training_alg.train(model=net, multigrid=mg_scheme, cycles=1)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))

trained_model.print('high')


# Test
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

