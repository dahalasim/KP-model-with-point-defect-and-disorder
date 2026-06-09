import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
nb = 9
v0 = 200
l = nb
b = 1/6
disorder = 0.15

np.random.seed(42)

# Ideal barrier positions (used only as reference)
x_r_ref = [-1/2 + r for r in range(1, nb + 1)]

# -----------------------------
# Generate disorder
# -----------------------------
x_r = [xr + np.random.uniform(-disorder, disorder)
       for xr in x_r_ref]

barrier_widths = [b + np.random.uniform(-disorder*b,
                                        disorder*b)
                  for _ in range(nb)]

v0_list = [v0 + np.random.uniform(-disorder*v0,
                                  disorder*v0)
           for _ in range(nb)]

# -----------------------------
# Potential
# -----------------------------
def potential(x, x_r, barrier_widths, v0_list):
    val = 0
    for i in range(len(x_r)):
        if x_r[i] - barrier_widths[i]/2 <= x <= x_r[i] + barrier_widths[i]/2:
            val = v0_list[i]
    return val

x = np.linspace(0, l, 1000)

y = [potential(xi, x_r, barrier_widths, v0_list)
     for xi in x]

# -----------------------------
# Hamiltonian matrix elements
# -----------------------------
N = 200

def f(k, x, l):
    return np.sin(k*np.pi*x/l)/(np.pi*k)

def Fnn(n, x, l):
    return x/l - f(2*n, x, l)

def Fmn(m, n, x, l):
    return f(m-n, x, l) - f(m+n, x, l)

def hnn(n, s, b, l):
    return Fnn(n, s+b/2, l) - Fnn(n, s-b/2, l)

def hmn(m, n, s, b, l):
    return Fmn(m, n, s+b/2, l) - Fmn(m, n, s-b/2, l)

def Hnn(n, l, x_r, v0_list, barrier_widths):
    result = (n*np.pi/l)**2

    for i in range(len(x_r)):
        result += v0_list[i] * hnn(
            n,
            x_r[i],
            barrier_widths[i],
            l
        )

    return result

def Hmn(m, n, l, x_r, v0_list, barrier_widths):
    result = 0

    for i in range(len(x_r)):
        result += v0_list[i] * hmn(
            m,
            n,
            x_r[i],
            barrier_widths[i],
            l
        )

    return result

def Hamiltonian(N, l, x_r, v0_list, barrier_widths):

    H = np.zeros((N, N))

    for m in range(1, N + 1):
        for n in range(1, N + 1):

            if m == n:
                H[m-1, n-1] = Hnn(
                    n,
                    l,
                    x_r,
                    v0_list,
                    barrier_widths
                )
            else:
                H[m-1, n-1] = Hmn(
                    m,
                    n,
                    l,
                    x_r,
                    v0_list,
                    barrier_widths
                )

    return H

# -----------------------------
# Diagonalize Hamiltonian
# -----------------------------
H = Hamiltonian(
    N,
    l,
    x_r,
    v0_list,
    barrier_widths
)

energies, states = np.linalg.eigh(H)
energies = np.sort(energies)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Potential
axes[0].plot(x, y, lw=1.5, color='green')
axes[0].set_xlabel("x")
axes[0].set_ylabel("V(x)")
axes[0].set_title(
    f"Disordered KP Potential"
)
axes[0].grid()

# Energy spectrum
nlevels = nb * 4 + 1

axes[1].scatter(
    [(n+1)*np.pi/l for n in range(nlevels)],
    energies[:nlevels],
    s=20,
    color='green'
)
axes[1].set_xlabel(r'Wave number')
axes[1].set_ylabel("Energy")
axes[1].set_title("Disordered Band Structure")
axes[1].grid()

plt.tight_layout(w_pad=2)
plt.show()