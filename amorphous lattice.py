import numpy as np
import matplotlib.pyplot as plt

nb = 15  # Number of barriers (wells)
l = nb  # Length of the system
b = 1 / 6  # Width of each barrier
v0 = 100  # Potential height
np.random.seed(0)  # For reproducibility

# Randomly vary the distances between the barriers
# Ensure positions are within the range [0, l]
x_r = np.cumsum(np.random.uniform(l / (2*nb), 2 * l / nb, nb))
x_r -= x_r[0]  # Start from 0
x_r += (l - x_r[-1]) / 2  # Center the last barrier at the end

# Define the potential function
def potential(x, x_r, b, v0):
    potential = 0
    for i in range(len(x_r)):
        if x_r[i] - b / 2 <= x <= x_r[i] + b / 2:
            potential = v0
    return potential

x = np.linspace(0, l, 1000)
y = [potential(xi, x_r, b, v0) for xi in x]

# Plot the potential
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='r')
plt.xlabel('x', fontsize=18)
plt.ylabel('Potential', fontsize=15)
plt.title('Amorphous Lattice Potential with Non-uniform Distances', fontsize=15)
plt.grid()
plt.show()
