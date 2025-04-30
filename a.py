import torch

python_list = [[1, 2], [2, 3], [4, 5]]
p_tensor = torch.tensor(python_list)
print(p_tensor)
new = p_tensor.repeat(4, 1)
print(new)

mylist=['1','2','3']
new=[x for x in mylist for i in range(2)]
print(new)
mylist=['1','2','3']
y = []
for i in range(2):
    for x in mylist:
        y.append(x)
print(y)

y = [x for i in range(2) for x in mylist]
print(y)
