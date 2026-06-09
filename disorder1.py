import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
nb = 9
v0 = 200
l = nb
b = 1/6
N = 200

disorders = [0.05, 0.15, 0.30]
colors = ['steelblue', 'darkorchid', 'darkorange']

x_r_ref = [-1/2 + r for r in range(1, nb + 1)]

# -----------------------------
# Hamiltonian matrix elements
# -----------------------------
def f(k, x, l):
    return np.sin(k * np.pi * x / l) / (np.pi * k)

def Fnn(n, x, l):
    return x / l - f(2 * n, x, l)

def Fmn(m, n, x, l):
    return f(m - n, x, l) - f(m + n, x, l)

def hnn(n, s, b, l):
    return Fnn(n, s + b/2, l) - Fnn(n, s - b/2, l)

def hmn(m, n, s, b, l):
    return Fmn(m, n, s + b/2, l) - Fmn(m, n, s - b/2, l)

def Hnn(n, l, x_r, v0_list, barrier_widths):
    result = (n * np.pi / l) ** 2
    for i in range(len(x_r)):
        result += v0_list[i] * hnn(n, x_r[i], barrier_widths[i], l)
    return result

def Hmn(m, n, l, x_r, v0_list, barrier_widths):
    result = 0
    for i in range(len(x_r)):
        result += v0_list[i] * hmn(m, n, x_r[i], barrier_widths[i], l)
    return result

def Hamiltonian(N, l, x_r, v0_list, barrier_widths):
    H = np.zeros((N, N))
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            if m == n:
                H[m-1, n-1] = Hnn(n, l, x_r, v0_list, barrier_widths)
            else:
                H[m-1, n-1] = Hmn(m, n, l, x_r, v0_list, barrier_widths)
    return H

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
nlevels = nb * 4 + 1

for ax, disorder, color in zip(axes, disorders, colors):
    np.random.seed(42)

    x_r = [xr + np.random.uniform(-disorder, disorder)
           for xr in x_r_ref]
    barrier_widths = [b + np.random.uniform(-disorder*b, disorder*b)
                      for _ in range(nb)]
    v0_list = [v0 + np.random.uniform(-disorder*v0, disorder*v0)
               for _ in range(nb)]

    H = Hamiltonian(N, l, x_r, v0_list, barrier_widths)
    energies, _ = np.linalg.eigh(H)
    energies = np.sort(energies)

    ax.scatter(
        [(n+1)*np.pi/l for n in range(nlevels)],
        energies[:nlevels],
        color=color,
        s=15
    )
    ax.set_title(rf'$D = {disorder}$', fontsize=13)
    ax.set_xlabel(r'Wave number', fontsize=11)
    ax.set_ylim(0, 190)
    ax.grid(True)

axes[0].set_ylabel('Energy', fontsize=12)

plt.suptitle('Disordered Kronig--Penney Energy Spectra', fontsize=15)
plt.tight_layout()
plt.savefig('disorder_comparison.png', dpi=300, bbox_inches='tight')
plt.show()