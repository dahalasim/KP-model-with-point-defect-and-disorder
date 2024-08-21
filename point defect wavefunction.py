import numpy as np
import matplotlib.pyplot as plt

nb = 10
l = nb
v0 = 100
N = 200
n = 3
x_r = [-1/2 + r for r in range(1, nb + 1)]
b = 1 / 6

def f(k, x, l):
    return np.sin(k * np.pi * x / l) / (k * np.pi)

def Fnn(x, l, n):
    return x / l - f(2 * n, x, l)

def Fmn(x, l, m, n):
    return f(m - n, x, l) - f(m + n, x, l)

def hnn(s, b, l, n, central_reduction=False):
    current_b = b * 0.1 if central_reduction else b
    return Fnn(s + current_b / 2, l, n) - Fnn(s - current_b / 2, l, n)

def hmn(s, b, l, m, n, central_reduction=False):
    current_b = b * 0.1 if central_reduction else b
    return Fmn(s + current_b / 2, l, m, n) - Fmn(s - current_b / 2, l, m, n)

def Hnn(n, l, x_r, v0, b):
    result = (n * np.pi / l) ** 2
    for i in range(len(x_r)):
        central_reduction = (i == len(x_r) // 2)
        result += v0 * hnn(x_r[i], b, l, n, central_reduction)
    return result

def Hmn(m, n, l, x_r, v0, b):
    result = 0
    for i in range(len(x_r)):
        central_reduction = (i == len(x_r) // 2)
        result += v0 * hmn(x_r[i], b, l, m, n, central_reduction)
    return result

def Hamiltonian(N, l, x_r, v0, b):
    matrix = []
    for m in range(1, N + 1):
        row = []
        for n in range(1, N + 1):
            if m == n:
                row.append(Hnn(n, l, x_r, v0, b))
            else:
                row.append(Hmn(m, n, l, x_r, v0, b))
        matrix.append(row)
    return matrix

Hamiltonian_matrix = Hamiltonian(N, l, x_r, v0, b)
u, v = np.linalg.eig(Hamiltonian_matrix)
sorted_indices = np.argsort(u)
cm_matrix = v.T[sorted_indices]

def functions(x, l, N):
    return np.array([np.sin(n * np.pi * x / l) for n in range(1, N + 1)])

def wavefunctions(x, l, N, n):
    return np.sqrt(2 / l) * np.sum(cm_matrix[n - 1] * functions(x, l, N))

x = np.linspace(0, l, N)

for n in range(1, n + 1):
    y = np.array([wavefunctions(x_val, l, N, n) for x_val in x])
    plt.plot(x, y)
    plt.xlabel('x', fontsize=15)
    plt.ylabel(r'$\psi(x)$', fontsize=15)
    plt.title(f'n={n} Eigenfunction', fontsize=15)
    plt.show()
    
    plt.plot(x, y * y, color='g')
    plt.xlabel('x', fontsize=15)
    plt.ylabel(r'$|\psi(x)|^2$', fontsize=15)
    plt.title(f'n={n} Probability density', fontsize=15)
    plt.show()
