"""
Example of VCycle multigrid with MTNN
"""
import time

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

# local
from MTNN.core.components import data, models
from MTNN.core.multigrid.operators import smoother, prolongation, restriction, coarsener
from MTNN.core.alg import trainer, evaluator, stopping
import MTNN.core.multigrid.scheme as mg
import MTNN.builder.levels as levels


# Set-up
# Load Data and Model
data = data.TestData(imagesize=(1,28,28), trainbatch_size=10, testbatch_size=10)
net = models.MultiLinearNet([784, 10, 5, 10], F.relu, F.log_softmax)

#test_data = data.TestData(trainbatch_size=10, testbatch_size=10)

# Build Multigrid Hierarchy Components
smoother = smoother.SGDSmoother(model=net, loss=nn.CrossEntropyLoss(),
                                lr=0.01, momentum=0.09, log_interval=20)

prolongation_op = prolongation.PairwiseAggregationProlongation()
restriction_op = restriction.PairwiseAggregationRestriction()


stopping_measure = stopping.EpochStopper(1)

#stopping_measure = stopping.CycleStopper(epochs=1, cycles=1)

# Build Multigrid Hierarchy Levels/Grids
num_levels = 3
FAS_levels = levels.build_vcycle_levels(num_levels=3, presmoother=smoother, postsmoother=smoother,
                                        prolongation_operator=prolongation_op, restriction_operator=restriction_op,
                                        coarsegrid_solver=smoother, stopper=stopping_measure)


mg_scheme = mg.VCycle(FAS_levels)
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

""""
# Test
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
"""