> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118A 


--------------------------------------------------------- 

import torch

x = torch.tensor([2.0,3.0,4.0])
print("Tensor: ", x)
print("Multiplied tensor: ", (x*2))

--------------------------------------------------------- 
import torch

def euclidian_norm(tensor):
    return torch.sqrt(torch.sum(tensor**2))

euclidian_norm(torch.tensor([3.0,4.0]))

--------------------------------------------------------- 

import torch
import torch.nn as nn    
import torch.optim as optim

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, init_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()

        self.hidden_layer = nn.Linear(init_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

inputs = torch.rand(5, input_size)
targets = torch.randint(0, output_size,(5,))

for epoch in range(epochs):
    ouputs = model(inputs)
    loss = criterion(ouputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("Training completed.")
