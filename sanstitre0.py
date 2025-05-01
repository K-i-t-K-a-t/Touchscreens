import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Grid parameters
nx, ny = 30, 30
N = nx * ny
dx = dy = 1e-3  # 1 mm grid spacing

# Constants
eps0 = 8.854e-12
eps_air = 1 * eps0
eps_glass = 5 * eps0
eps_finger = 50 * eps0

# Region slices
air_rows = slice(0, 13)
glass_rows = slice(14, 17)
sensor_rows = slice(18, 25)

Tx_rows = (23)
Rx_rows = (23)
finger_rows = slice(0, 15)
Tx_columns = slice(0,6)
Rx_columns = slice(25, 31)
finger_columns = slice(13,19)

# Build permittivity map
epsilon = np.ones((ny, nx)) * eps_air
epsilon[glass_rows, :] = eps_glass
epsilon[finger_rows, finger_columns] = eps_finger

# Helper to flatten 2D index
def idx(i, j):
    return i * nx + j

# Set boundary conditions
tx_y = 22
rx_y = 22

V_fixed = -np.ones((ny, nx))

V_fixed[Tx_rows, Tx_columns] = 1.0
V_fixed[Rx_rows, Rx_columns] = 0.0
V_fixed[finger_rows, finger_columns] = 0.0
# Assemble system matrix
A = lil_matrix((N, N))
b = np.zeros(N)

for i in range(ny):
    for j in range(nx):
        k = idx(i, j)

        if k in V_fixed:
            A[k, k] = 1
            b[k] = V_fixed[k]
            continue

        # 5-point Laplacian with permittivity weighting
        total_weight = 0
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                nk = idx(ni, nj)
                eps_avg = 0.5 * (epsilon[i, j] + epsilon[ni, nj])
                weight = eps_avg / (dx**2)
                A[k, nk] = -weight
                total_weight += weight
        A[k, k] = total_weight

# Solve the sparse linear system
phi = spsolve(A.tocsr(), b)
V = phi.reshape((ny, nx))

# Compute electric field from potential
Ey, Ex = -np.gradient(V, dy, dx)
E2 = Ex**2 + Ey**2

# Compute energy and capacitance
energy_density = 0.5 * epsilon * E2
W = np.sum(energy_density) * dx * dy
V0 = 1.0
C = 2 * W / V0**2

print(f"Estimated capacitance: {C * 1e12:.3f} pF")

# Plot potential field
plt.figure(figsize=(8, 6))
plt.imshow(V, cmap='plasma', origin='upper',
           extent=[0, nx*dx, ny*dy, 0])  # Aligns with physical scale
plt.colorbar(label="Potential (V)")
plt.title("Electric Potential Field")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.show()

# Simplified Electric Field Vectors
plt.figure(figsize=(8, 6))
plt.imshow(V, cmap='plasma', origin='upper', extent=[0, nx*dx, ny*dy, 0], alpha=0.5)

# Using quiver directly without excessive adjustments
plt.quiver(np.arange(0, nx) * dx,  # X positions of vectors
           np.arange(0, ny) * dy,  # Y positions of vectors
           Ex,  # Ex components
           Ey,  # Ey components
           color='white')  # Scale the vectors to fit plot
plt.colorbar(label="Potential (V)")
plt.title("Electric Field Vectors (Simplified)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.show()

