import torch

x = torch.tensor(data=[2.0, 3.0], requires_grad=True)
y = x ** 2
z = 2 * y + 3
print(x, y, z)

target = torch.tensor([3.0, 4.0])
print(z - target)

loss = torch.sum(torch.abs(z - target))
loss.backward()
print(x.grad)