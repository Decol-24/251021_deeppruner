import torch

x = torch.zeros(3, 3)
index = torch.tensor([
    [0, 1, 2],
    [1, 1, 1],
    [1, 1, 0]
])
x.scatter_(0, index, 1)

print(x)