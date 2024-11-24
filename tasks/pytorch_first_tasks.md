```
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118A 
```

--------------------------------------------------------- 

```
import torch

x = torch.tensor([2.0,3.0,4.0])
print("Tensor: ", x)
print("Multiplied tensor: ", (x*2))
```

--------------------------------------------------------- 
```
import torch

def euclidian_norm(tensor):
    return torch.sqrt(torch.sum(tensor**2))

euclidian_norm(torch.tensor([3.0,4.0]))
```

--------------------------------------------------------- 
```
import torch
# https://pytorch.org/docs/stable/nn.html
import torch.nn as nn 

# https://pytorch.org/docs/stable/optim.html   
import torch.optim as optim

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, init_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        # Applies an affine linear transformation to the incoming data
        # Parameters:
        # 
        # in_features (int) – size of each input sample
        # out_features (int) – size of each output sample
        # bias (bool) – If set to False, the layer will not learn an additive bias. Default: True

        self.hidden_layer = nn.Linear(init_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#relu
        # Applies the rectified linear unit function element-wise
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

input_size = 4
hidden_size = 8
output_size = 3
learning_rate = 0.01
epochs = 100

model = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
# This criterion computes the cross entropy loss between input logits and target.
#
# It is useful when training a classification problem with C classes. If provided, the optional argument
# weight should be a 1D Tensor assigning weight to each of the classes.
criterion = nn.CrossEntropyLoss()

# https://arxiv.org/abs/1412.6980
# Adam, is an algorithm for first-order gradient-based optimization of stochastic
# objective functions, based on adaptive estimates of lower-order moments
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand
# Returns a tensor filled with random numbers from a uniform
# distribution on the interval [0,1)[0,1)
inputs = torch.rand(5, input_size)

# https://pytorch.org/docs/stable/generated/torch.randint.html#torch-randint
# Returns a tensor filled with random integers generated uniformly
# between low (inclusive) and high (exclusive)
targets = torch.randint(0, output_size,(5,))

for epoch in range(epochs):

    # Creates model instance and executes the 'forward' method
    ouputs = model(inputs)

    # Executes the CrossEntropyLoss
    loss = criterion(ouputs, targets)

    # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad
    # Reset the gradients of all optimized torch.Tensor
    optimizer.zero_grad()

    # https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward
    loss.backward()

    # Perform a single optimization step.
    optimizer.step()

    if(epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("Training completed.")

```
