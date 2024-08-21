import numpy as np
import matplotlib.pyplot as plt

nb = 5  # Number of barriers
v0 = 200  # Potential height
l = nb  # Total length of the potential
b = 1 / 6  # Initial width of each barrier
x_r = [-1/2 + r for r in range(1, nb + 1)]  # Position of the centers of the barriers

def potential(x, x_r, b, v0):
    potential = 0
    for i in range(nb):
        # Reduce the width of the central barrier by 25%
        current_b = b * 0.5 if i == nb // 2 else b
        if x_r[i] - current_b / 2 <= x <= x_r[i] + current_b / 2:
            potential = v0
    return potential

x = np.linspace(0, l, 1000)
y = [potential(x_i, x_r, b, v0) for x_i in x]

plt.figure(figsize=(8, 6))
plt.plot(x, y, color='r')
plt.xlabel('x', fontsize=18)
plt.ylabel('Potential', fontsize=15)
plt.title('Kronig-Penney Model with Reduced Central Barrier', fontsize=15)
plt.grid()
plt.show()
