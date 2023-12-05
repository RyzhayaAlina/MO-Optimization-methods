import numpy as np
from math import *

# Функция подсчета значения функции
def funcf(A, x, b):
    f=0.5* (x.transpose()) @ A @ x + np.dot(b.flatten(), x.flatten())
    return f

# Функция подсчета матрицы Якоби
def funcJ(A, x, x0, y):
    I = np.eye(4)
    matrix = np.zeros((5, 5))
    for i in range(4):
        for j in range(5):
            if j==4:
                matrix[i, j] = 2*(x[i]-x0[i])
                matrix[j, i] = matrix[i, j]
            else:
                matrix[i , j] = A[i, j] + 2* I[i, j] *y
    matrix[4, 4] = 0
    return(matrix)

A = np.array([
    [3, 5, 2, 7],
    [5, 6, 9, -1],
    [2, 9, 1, 4],
    [7, -1, 4, 8]
])
b = np.array([[1], [7], [2], [-1]])
x0 = np.array([[0], [1], [1], [0]])
r = 9  # радиус сферы (произвольное число)

arr_result = []
func_result = []

# 1
y = r
e = 10**(-6)

for i in range(8):
    x = np.zeros((5, 1))
    x[:4, :] = x0
    x[4] = y

    if i>=4:
        x[i-4]-=r
    else:
        x[i]+=r

    print("Начальное приближение №", i+1, x[:4, :].T)
    condition = False
    while not condition:
        x_new = np.copy(x)
        J = funcJ(A, x, x0, y)

        left_part = np.ones((5, 1))
        left_part[:4, :] = (A + 2*np.eye(4)*y) @ x[:4, :]+(b+2*y*x0)
        left_part[4] = (np.linalg.norm(x[:4, :] - x0))**2 -r**2

        x_new = x - np.linalg.inv(J) @ left_part
        # print(np.linalg.norm(x_new-x))
        if np.linalg.norm(x_new-x) <= e:
            condition = True
        x = x_new
        y = x[4,0]
    if x[4] > 0:
        arr_result.append(x)
        func_result.append(funcf(A, x[:4, :], b))
    print("Вычисленное значение:", "xi = ", x[:4, :].T, "y =", x[4])
    print("Значение функции в точке ", funcf(A, x[:4, :], b))
    print(" ")

# 2
y = 0
x_star = -np.linalg.inv(A) @ b
func_x_star = funcf(A, x_star, b)
if np.linalg.norm(x_star-x0)<=r:
    print("Начальное приближение №", 9, "xi =", x_star.T, " y = 0")
    print("Значение функции в точке ", funcf(A, x_star, b))
    arr_result.append(x_star)
    func_result.append(funcf(A, x_star, b))
    print("Точка при y = 0 удовлетворяет условию")
else:
    print("Точка при y = 0 не удовлетворяет условию")

min_ = min(func_result)
k = func_result.index(min_)
print("Минимальное значение функции: ", min_, "достигается в точке №", k+1,  arr_result[k].T)

