"""
Example of cascadic multigrid with MTNN
"""
import time

# local
import MTNN.core as core
from  MTNN.core.components import datasets, models
import MTNN.core.alg.optimizer.operators.smoother as smoother
import MTNN.core.alg.optimizer.operators.prolongation as prolongation
import MTNN.core.alg.optimizer.operators.stopping as stopping
import MTNN.core.alg.optimizer.scheme.multigrid as multigrid
import MTNN.core.alg.trainer as trainer

from MTNN.core.alg.evaluator import CategoricalEvaluator


data = datasets.MnistData()
net = models.BasicMnistModel()

# Smoother
SGD = smoother.SGDSmoother()
# Prolongation
prolong_op = prolongation.LowerTriangleInterpolator()
# Stopping measure
stopping_measure = stopping.ByNumIterations()

multigrid_optimizer = multigrid.Cascadic(levels=3,
                                         postsmoother= SGD,
                                         prolongation=prolong_op,
                                         coarsegrid_solver=SGD,
                                         stopping_criteria=stopping_measure)
training_alg = trainer.Trainer(train_batch_size=10, optimizer=multigrid_optimizer)
evaluator = CategoricalEvaluator(test_batch_size=10)

# Train
print('Starting Training')
start = time.perf_counter()
net = training_alg.train(model=net)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))


# Test

print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.validate(model=net, dataloader=data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3f)' % (stop-start))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
