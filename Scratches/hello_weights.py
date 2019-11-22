# Scratch code to learn about weight initialization and training on them 
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TestModel(nn.Module):
    def __init__(self, module):
        super(TestModel, self).__init__()
        self.module = module

    def register_parameter(self, new_weight):
        self.module.register_parameter()

    def forward(self, x):
        print(x)
        x = self.module(x) # do not use intermediate variables or gradients won't be passed
        print(x)
        x = F.relu(x)
        print(x)
        return x #output layer should have same number of classes as out_features


#%%
# Make a test model object
model = TestModel(nn.Linear(2,1)) # in features/weights, out_features/neurons
print("\nMODEL\n", model)
print("\nMODEL.MODULES\n", model.module)
#print("\nMODEL.PARAMETERS()\n",model.parameters()) # This just gives you the address location
for name, param in model.named_parameters():
    print("\t",name,param)

#%%

# Modify weights and bias
model.module.weight.data.fill_(0.1)
model.module.bias.data.fill_(0.5)
print("\nINITIALIZED MODEL.PARAMETERS()\n",model.module.parameters()) # This just gives you the address location
for name, param in model.named_parameters():
    print("\t",name,param)

#%%
# Original tensor x to use as weight tensor
# Filled with ones
x = torch.ones(1,2) #(row, coloumn)
print("\nOriginal Tensor X\n", x.data)


# Kaiming fills input tensor with values according to the Kaiming
KU_weight= torch.nn.init.kaiming_uniform_(x)
# Kaiming Uniform is sampled from U[-bound, bound]
print("\nKAIMING UNIFORM\n", KU_weight)

# Kaiming normal is sampled from N[0, std**2]
# Using a normal distribution
KN_weight = torch.nn.init.kaiming_normal_(x)
print("\nKAIMING NORMAL\n", KN_weight)
#%%

# Make transformed tensor into a nn.parameter
# Parameters are Tensor subclasses
# Will be automatically added to a Module's list of parameters(model.parameters())
# Assigning a Tensor will not have the same effect
KN_weight_param = nn.Parameter(data=KN_weight, requires_grad=True)
KU_weight_param = nn.Parameter(data=KU_weight, requires_grad=True )
print("\nKaiming Normal Weight as a parameter\n", KN_weight_param)

# Use Register_parameter() method to make it trainable 
model.module.register_parameter("kaiming_weight", KN_weight_param) # (name, param)


print("\n\n**NEW MODEL PARAMETERS**")
for name, param in model.named_parameters():
    print("\t",name, param)

#%%
"""
Forward Pass 1
"""
# Input
model_in = torch.ones(1,2)
#model_in = torch.tensor([2,2]) # Throws an error: Expected object of type Long but got Scalar Float
print("\nMODEL INPUT\n", model_in)
# Forward pass
print("\nDOING FORWARD PASS")
output = model(model_in)
print("\nPREDICTION", output)
optimizer = optim.SGD(model.parameters(), lr= 0.01, momentum= 0.5)
#optimizer.zero_grad() # not needed because first pass

#%%
"""
Backward pass 1
"""
# Target
target = torch.FloatTensor([3])
# Reshape target size
# Target/output size must have same size as the input size ex. [1,1]
target = target.view(-1, 1)
print(target)

loss = F.mse_loss(output,target)
print("GRADIENTS\n", model.module.weight.grad)

print("\nLOSS", loss)
loss.backward()

print("GRADIENTS\n", model.module.weight.grad)
#%%


print("\n\n**MODEL PARAMETERS AFTER ONE FULL PASS**")
for name, param in model.named_parameters():
    print("\t", name, param)

"""
# How do you train the model on new weights?
# Base class of nn.optim.optimizer has .add_param_group(dict) method
# that specifies what tensors should be optimized along with group
"""

"""
Update the optimizer
"""
# The new kaiming weight is already added to the Param groups because of register_paramter()
print("\nOPTIMIZER PARAM GROUPS\n",optimizer.param_groups)
print("\nOPTIMIZER STATEDICT\n", optimizer.state_dict())
#optimizer.add_param_group({'params': KU_weight_param})

# Update optimizer.
print("\n**UPDATE OPTIMIZER**")
optimizer.step()

print("\nOPTIMIZER PARAM GROUPS\n",optimizer.param_groups)



model_in = torch.tensor((2,2))
print(model_in)



