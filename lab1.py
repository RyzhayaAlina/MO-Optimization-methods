import numpy as np, matplotlib.pyplot as plt
from math import *

# Функция подсчета значения поизводной функции
def funcGrad(x, A_trans, A, b):
    f=0.5*(A_trans+A) @ x + b
    return f

# Функция получения значения Евклидовой нормы
def funcNorm(x2, x1):
   norm = np.sqrt(sum(pow(a-b, 2) for a, b in zip(x2, x1)))
   return norm

# Функция подсчета значения функции
def funcf(x, A, b):
    f=0.5* (x.transpose()) @ A @ x + np.dot(b.flatten(), x.flatten())
    return f

# Функция преобразования списка списков в один список (для графика)
def funclistmerge(mass):
    all=[]
    for l in mass:
      all.extend(l)
    return all

A = np.array([
    [2, -1, 0, 0, 0, 0],
    [-1, 2, -1, 0, 0, 0],
    [0, -1, 2, -1, 0, 0],
    [0, 0, -1, 2, -1, 0],
    [0, 0, 0, -1, 2, -1],
    [0, 0, 0, 0, -1, 2]
])

b=np.array([[1], [1], [1], [0], [2], [1]])
x0=np.array([[1], [0], [0], [2], [2], [1]])

# for row in A:
#     print(row)

# Проверка матрицы на положительную определенность

if np.all(np.linalg.eigvals(A) > 0):
    print("Матрица положительно определена")
else:
    print("Матрица не положительно определена")


# A_inv=np.linalg.inv(A)
# for row in A_inv:
#     print(row)

# точное решение
A_trans=A.transpose()
A_new=np.linalg.inv(A_trans+A)
x_tochnoye=A_new @ (-2*b)

# первичное значение x(k+1)
h=10**(-4)
x_k_new=x0-h*funcGrad(x0, A_trans, A, b)

e=10**(-6)
n=0
x_k=x0
mass_x=[]
mass_f=[]
while funcNorm(x_k_new, x_k)>e:
    mass_x.append(x_k)
    mass_f.append(funcf(x_k, A, b))
    x_k=x_k_new
    x_k_new=x_k-h*funcGrad(x_k, A_trans, A, b)
    n=n+1

print(n) 
#n=282705
print("Примерное значение при шаге 0.25 :")
print(mass_x[n//4], (mass_f[n//4]))
print("Примерное значение при шаге 0.5 :")
print(mass_x[n//2], (mass_f[n//2]))
print("Примерное значение при шаге 0.75 :")
print(mass_x[3*n//4], (mass_f[3*n//4]))
print("Примерное значение при последнем шаге:")
print(mass_x[n-1], (mass_f[n-1]))
print(" ")

print("Точное значение:")
print(x_tochnoye, funcf(x_tochnoye, A, b))
print(" ")

# Погрешность
print("Погрешность решения:")
print(abs(mass_x[n-1]-x_tochnoye))

print("Погрешность значения функции:")
print(abs((mass_f[n-1])-funcf(x_tochnoye, A, b)))
# print(funcf(x_tochnoye, A, b))

plt.plot(funclistmerge(mass_f))
plt.show()