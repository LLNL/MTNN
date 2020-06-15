"""
Example of cascadic multigrid with MTNN
"""
import time

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
# local
from MTNN.core.components import data, models
from MTNN.core.multigrid.operators import smoother, prolongation
from MTNN.core.alg import trainer, evaluator, stopping
import MTNN.core.multigrid.scheme as mg
import MTNN.builder.levels as levels



# Load Data and Model
data = data.MnistData(trainbatch_size=100, testbatch_size=10)
net = models.MultiLinearNet([784, 50, 25, 10], activation=F.relu)


# Build Multigrid Hierarchy
smoother = smoother.SGDSmoother(model=net, loss=nn.CrossEntropyLoss(),
                                lr=0.01, momentum=0.09, log_interval=10000)

prolongation_operator = prolongation.LowerTriangleProlongation(expansion_factor=3)
stopping_measure = stopping.EpochStopper(1)
#stopping_measure = stopping.CycleStopper(1, 3)



mg_levels = levels.build_uniform_levels(num_levels=3,
                                        presmoother=smoother,
                                        postsmoother=smoother,
                                        prolongation_operator =prolongation_operator,
                                        coarsegrid_solver=smoother,
                                        stopping_criteria=stopping_measure)

mg_optimizer = mg.Cascadic(mg_levels)

training_alg = trainer.MultigridTrainer(dataloader=data.trainloader,
                                        verbose=True,
                                        log=False,
                                        save=False,
                                        load=False)
evaluator = evaluator.CategoricalEvaluator()



# Train
print('Starting Training')
start = time.perf_counter()
training_alg.train(model=net, multigrid=mg_optimizer, cycles=1)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))


# Test
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

