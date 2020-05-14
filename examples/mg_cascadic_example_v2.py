"""
Example of cascadic multigrid with MTNN
"""
import time

# PyTorch
import torch.nn as nn

# local
from MTNN.core.components import data, models
from MTNN.core.optimizer.operators import smoother, prolongation
from MTNN.core.alg import trainer, evaluator, stopping
import MTNN.core.optimizer.multigrid as mg
import MTNN.builder.levels as levels



# Load Data and Model
data = data.MnistData(trainbatch_size=10, testbatch_size=10)
#net = models.SingleLayerNet(dim_in=784, hidden=10, dim_out=10)
#net = models.BasicMnistModel()
#net = models.SingleFCNet(dim_in=784, hidden=25, dim_out=10)
net = models.MultiLinearNet([784, 50, 25, 10])


# Build Multigrid Hierarchy

smoother = smoother.SGDSmoother(model=net, loss=nn.CrossEntropyLoss(),
                                lr=0.01, momentum=0.09, log_interval=10)

#prolongation_operator = prolongation.IdentityProlongation()
prolongation_operator = prolongation.LowerTriangleProlongation(expansion_factor=2)
stopping_measure = stopping.EpochStopper(1)
#stopping_measure = stopping.CycleStopper(1, 3)

mg_levels = levels.build_uniform(num_levels=2,
                                 presmoother=smoother,
                                 postsmoother=smoother,
                                 prolongation=prolongation_operator,
                                 coarsegrid_solver=smoother,
                                 stopping_criteria=stopping_measure)
mg_optimizer = mg.Cascadic(mg_levels)

training_alg = trainer.MultigridTrainer(dataloader=data.trainloader,
                                        verbose=True,
                                        save=True,
                                        load=False)
evaluator = evaluator.CategoricalEvaluator()



# Train
print('Starting Training')
start = time.perf_counter()
training_alg.train(model=net, optimizer=mg_optimizer, cycles=1)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))


# Test
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3f)' % (stop-start))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

