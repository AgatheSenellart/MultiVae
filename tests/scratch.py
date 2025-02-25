import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
l = []
b = a.detach() * 2
l.append(b)


print(l[0].requires_grad)
print(b.requires_grad)
