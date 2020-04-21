"""
Example of cascadic multigrid with MTNN
"""
import time

# PyTorch
import torch.nn as nn
import torch.optim as optim

# local
from  MTNN.core.components import datasets, models
import MTNN.core.alg.optimizer.operators.smoother as smoother
import MTNN.core.alg.optimizer.operators.prolongation as prolongation
import MTNN.core.alg.optimizer.operators.stopping as stopping
import MTNN.core.alg.optimizer.scheme.multigrid as mg
import MTNN.core.alg.trainer as trainer
import MTNN.core.alg.evaluator as evaluator

# Load Data and Model
data = datasets.CIFAR10Data(batch_size=1)
net = models.BasicCifarModel()

# Smoother/ Solver
smoother = smoother.SGDSmoother( model=net, loss=nn.CrossEntropyLoss(), lr=0.01, momentum=0.09)
prolongation_operator = prolongation.LowerTriangleInterpolator(expansion_factor=3)
stopping_measure = stopping.ByNumIterations()

mg_optimizer = mg.Cascadic(levels=3, presmoother=smoother, postsmoother= smoother,
                           prolongation=prolongation_operator,
                           coarsegrid_solver=smoother,
                           stopping_criteria=stopping_measure)


training_alg = trainer.MultigridTrainer(dataloader=data.trainloader, train_batch_size=1)
evaluator = evaluator.CategoricalEvaluator(test_batch_size=1)


# Train
print('Starting Training')
start = time.perf_counter()
net = training_alg.train(model=net, optimizer=mg_optimizer)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))

"""
# Test

print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3f)' % (stop-start))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
"""