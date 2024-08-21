import numpy as np
import matplotlib.pyplot as plt

nb = 10  # Number of barriers (wells)
n = 21    # Number of wavefunctions to plot
l = nb   # Length of the system
v0 = 100 # Potential height
b = 1 / 6 # Width of each barrier
N = 100  # Number of basis functions

# Generate non-uniform positions for the wells
np.random.seed(0)  # For reproducibility
x_r = np.cumsum(np.random.uniform(l / (2*nb), 2 * l / nb, nb))
x_r -= x_r[0]  # Start from 0
x_r += (l - x_r[-1]) / 2  # Center the last barrier at the end

# Define the necessary functions for constructing the Hamiltonian
def f(k, x, l):
    return np.sin(k * np.pi * x / l) / (k * np.pi)

def Fnn(x, l, n):
    return x / l - f(2 * n, x, l)

def Fmn(x, l, m, n):
    return f(m - n, x, l) - f(m + n, x, l)

def hnn(s, b, l, n):
    return Fnn(s + b / 2, l, n) - Fnn(s - b / 2, l, n)

def hmn(s, b, l, m, n):
    return Fmn(s + b / 2, l, m, n) - Fmn(s - b / 2, l, m, n)

def Hnn(n, l, x_r, v0, b):
    result = (n * np.pi / l) ** 2
    for i in range(len(x_r)):
        result += v0 * hnn(x_r[i], b, l, n)
    return result

def Hmn(m, n, l, x_r, v0, b):
    result = 0
    for i in range(len(x_r)):
        result += v0 * hmn(x_r[i], b, l, m, n)
    return result

# Construct the Hamiltonian matrix
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

# Compute the eigenvalues and eigenvectors
Hamiltonian_matrix = Hamiltonian(N, l, x_r, v0, b)
u, v = np.linalg.eig(Hamiltonian_matrix)
sorted_indices = np.argsort(u)
cm_matrix = v.T[sorted_indices]

# Function to generate the sine functions for the basis
def functions(x, l, N):
    return np.array([np.sin(n * np.pi * x / l) for n in range(1, N + 1)])

# Function to compute the wavefunctions
def wavefunctions(x, l, N, n):
    return np.sqrt(2 / l) * np.sum(cm_matrix[n - 1] * functions(x, l, N))

# Define the x-axis for plotting
x = np.linspace(0, l, N)

# Plot wavefunctions and probability densities
for i in range(1, n + 1):
    y = np.array([wavefunctions(x_val, l, N, i) for x_val in x])
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f'$\psi_{i}(x)$')
plt.xlabel('x', fontsize=15)
plt.ylabel(r'$\psi(x)$', fontsize=15)
plt.title(f'n={i} Eigenfunction', fontsize=15)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x, y * y, color='g', label=r'$|\psi_{i}(x)|^2$')
plt.xlabel('x', fontsize=15)
plt.ylabel(r'$|\psi(x)|^2$', fontsize=15)
plt.title(f'n={i} Probability Density', fontsize=15)
plt.legend()
plt.show()
