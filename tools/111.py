import torch


a = torch.randn(7)

# print(a, a[-3:])

b = torch.randn((3, 4))
c = b.new()
# print(c.size(), type(c), c.dtype)
# print(id(b), id(c))

x1 = [i for i in range(1, 10)]
x1 = torch.tensor(x1)
x2 = torch.clamp(x1, min=5)
# print(x2)

A = torch.arange(12).view(3, 1, 4)
# print(A.size())
A = A.expand(-1, 12, -1)
# print(A.size())
# print(A)

B = torch.normal(0, 1, (3, 4))
C = torch.normal(0, 1, (3, 4))
D = torch.max(B[:, None, :2], C[:, :2])
# print("B:{},\n C:{}, \nD:{}".format(B, C, D))

# torch.Tensor.new()的用法

# E = B.new(C.size()).normal_(0, 0.1)
# E = torch.Tensor.new(C)       # E为空的Tensor
# E = B.new(C)                  # E为和C一样内容和B一样type的变量
# print(C)
# print(E)
# print(id(C), id(E))
# print(B.type(), C.type())
a = torch.randn((3, 1))
b = torch.randn(4)
# print(a, b)
c = torch.add(a, b)
# print(c, c.size())
a = torch.randint(1, 10, (1, 10))

# print(a[0])
# print(a[0, :])
# a = a[:-1]
# print(a.size())
b = a.ge(5)
# print(b)
c = a >= 5
# print(c)
d = a[c]
# print(d)
a = torch.randn(4, 4)
print(a)
b = torch.argmax(a)
c = torch.argmax(a, dim=0, keepdim=True)
print(b)
print(b.size())
print(c)
print(c.size())



















