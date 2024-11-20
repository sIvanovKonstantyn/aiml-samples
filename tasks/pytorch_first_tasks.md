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
    
