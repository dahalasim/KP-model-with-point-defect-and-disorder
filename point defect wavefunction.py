import numpy as np
import matplotlib.pyplot as plt
nb = 9
v0 = 200
v0_wf = 200
l = nb
b = 1/6
x_r = [-1/2 + r for r in range(1, nb+1)]
first_idx = 0
# --- Potential (v0=200) ---
def potential(x, x_r, b, v0):
    val = 0
    for i in range(nb):
        current_b = b * 0.5 if i == first_idx else b
        if x_r[i] - current_b/2 <= x <= x_r[i] + current_b/2:
            val = v0
    return val
x = np.linspace(0, l, 10000)
y = [potential(x_i, x_r, b, v0) for x_i in x]
lo = x_r[first_idx] - (b * 0.5)/2
hi = x_r[first_idx] + (b * 0.5)/2
y_normal = [v if not (lo <= x_i <= hi) else 0 for x_i, v in zip(x, y)]
y_first = [v if (lo <= x_i <= hi) else 0 for x_i, v in zip(x, y)]
x_left = [xi for xi in x if xi < lo]
x_right = [xi for xi in x if xi > hi]
# --- Hamiltonian (v0=100) ---
N = 200
barrier_widths = [b*0.01 if i == first_idx else b for i in range(nb)]
def f(k, x, l):
    return np.sin(k * np.pi * x / l) / (k * np.pi)
def Fnn(x, l, n):
    return x/l - f(2*n, x, l)
def Fmn(x, l, m, n):
    return f(m-n, x, l) - f(m+n, x, l)
def hnn(s, b_i, l, n):
    return Fnn(s + b_i/2, l, n) - Fnn(s - b_i/2, l, n)
def hmn(s, b_i, l, m, n):
    return Fmn(s + b_i/2, l, m, n) - Fmn(s - b_i/2, l, m, n)
def Hnn(n, l, x_r, v0, barrier_widths):
    result = (n * np.pi / l)**2
    for i in range(len(x_r)):
        result += v0 * hnn(x_r[i], barrier_widths[i], l, n)
    return result
def Hmn(m, n, l, x_r, v0, barrier_widths):
    result = 0
    for i in range(len(x_r)):
        result += v0 * hmn(x_r[i], barrier_widths[i], l, m, n)
    return result
def Hamiltonian(N, l, x_r, v0, barrier_widths):
    matrix = []
    for m in range(1, N+1):
        row = []
        for n in range(1, N+1):
            if m == n:
                row.append(Hnn(n, l, x_r, v0, barrier_widths))
            else:
                row.append(Hmn(m, n, l, x_r, v0, barrier_widths))
        matrix.append(row)
    return matrix
Hamiltonian_matrix = Hamiltonian(N, l, x_r, v0_wf, barrier_widths)
u, v = np.linalg.eig(Hamiltonian_matrix)
sorted_indices = np.argsort(u.real)
cm_matrix = v.T[sorted_indices]
def functions(x, l, N):
    return np.array([np.sin(n * np.pi * x / l) for n in range(1, N+1)])
def wavefunctions(x, l, N, n):
    return np.sqrt(2/l) * np.sum(cm_matrix[n-1] * functions(x, l, N))
x_plot = np.linspace(0, l, 500)
# --- Figure 1: potential + n=1 wavefunction ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
y_wf1 = np.array([wavefunctions(xi, l, N, 1) for xi in x_plot])
y_wf1 = -y_wf1
axes[0].plot(x, y_normal, color='k')
axes[0].plot(x, y_first, color='b')
axes[0].plot(x_left, [0]*len(x_left), color='k', linewidth=1.5, zorder=3)
axes[0].plot(x_right, [0]*len(x_right), color='k', linewidth=1.5, zorder=3)
axes[0].set_xlabel('x', fontsize=15)
axes[0].set_ylabel('Potential', fontsize=15)
axes[0].set_title('Kronig-Penney Model with Reduced First Barrier', fontsize=13)
axes[0].grid()
axes[1].plot(x_plot, y_wf1, color='b')
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel(r'$\psi(x)$', fontsize=12)
axes[1].set_title('n=1 Eigenfunction', fontsize=11)
axes[1].grid()
plt.tight_layout(w_pad=2)
plt.show()
# --- Figure 2: gap state wavefunctions ---
states = [9, 19, 27]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, n in enumerate(states):
    y_wf = np.array([wavefunctions(xi, l, N, n) for xi in x_plot])
    if y_wf[len(y_wf)//2] < 0:
        y_wf = -y_wf
    axes[col].plot(x_plot, y_wf, color='b')
    axes[col].set_xlabel('x', fontsize=12)
    axes[col].set_ylabel(r'$\psi(x)$', fontsize=12)
    axes[col].set_title(f'n={n} Eigenfunction', fontsize=11)
    axes[col].grid()
plt.tight_layout(w_pad=2)
plt.show()
