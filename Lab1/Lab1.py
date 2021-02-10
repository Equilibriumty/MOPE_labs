import random
import numpy as np

# варіант - 311 (max(Y))
a0 = 1
a1 = 3
a2 = 3
a3 = 5

# generating x1, x2, x3
x1 = [random.randrange(1, 21, 1) for _ in range(8)]
x2 = [random.randrange(1, 21, 1) for _ in range(8)]
x3 = [random.randrange(1, 21, 1) for _ in range(8)]
x_matrix = np.array([x1, x2, x3])

# calculating Y
Y = [a0 + (a1 * x1[i]) + (a2 * x2[i]) + (a3 * x3[i]) for i in range(8)]

# getting max(Y)
Y_max = max(Y)

# calculating x01, x02, x03
x01 = (max(x1) + min(x1)) / 2
x02 = (max(x2) + min(x2)) / 2
x03 = (max(x3) + min(x3)) / 2

# calculating dx1, dx2, dx3
dx1 = x01 - min(x1)
dx2 = x02 - min(x2)
dx3 = x03 - min(x3)

# calculating xn1, xn2, xn3
xn1 = [(x1[i] - x01) / dx1 for i in range(8)]
xn2 = [(x2[i] - x02) / dx2 for i in range(8)]
xn3 = [(x3[i] - x03) / dx3 for i in range(8)]
xn_matrix = np.array([xn1, xn2, xn3])

# calculating Yet
Yet = a0 + (a1 * x01) + (a2 * x02) + (a3 * x03)

print("X1 X2 X3")
print(x_matrix.transpose())
print("Y: ", Y)
print("x01, x02, x03: ", x01, x02, x03)
print("dx1, dx2, dx3: ", dx1, dx2, dx3)
print("Xn1, Xn2, xn3:")
print(xn_matrix.transpose())
print("Yет:", Yet)
print("max(Y): ", Y_max)
