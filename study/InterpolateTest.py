import torch
import torch.nn.functional as F

tensor = torch.arange(100,dtype=torch.float32)
tensor = tensor.view(10,10)
print(tensor.shape)
tensor = tensor.unsqueeze(0).unsqueeze(1)
print(tensor.shape)
newtensor = F.interpolate(tensor,(9,9),mode='bilinear',align_corners=False)
i = 0