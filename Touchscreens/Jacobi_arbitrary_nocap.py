# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# CONSTANTS

EPS_ZERO = 8.854e-12
EPS_air = EPS_ZERO

# SETTINGS

nx = 90
ny = 90
N = nx * ny

# GRID SETUP

class Cell:
    def __init__(self):
        self.eps = EPS_air         # Permittivity
        self.V = 0.0               # Electric potential
        self.E = np.zeros(3)       # Electric field vector
        
def create_grid(length, height):
    grid = np.empty((length, height), dtype=object)
    for i in range(length):
        for j in range(height):
            grid[i, j] = Cell()
    return grid

# BOUNDARY CONDITIONS

V_fixed = dict()

def idx(i, j):
    return i * nx + j

class Electrode:
    def __init__(self, size, center, permittivity, voltage, V_fixed):
        self.size = size
        self.center = center
        self.eps = permittivity
        self.V = voltage
        self.fixed = V_fixed
    def place(self, grid):
        for dy in range(-self.size[0] // 2, self.size[0] // 2 + 1):
            for dx in range(-self.size[1] // 2, self.size[1] // 2 + 1):
                y = self.center[0] + dy
                x = self.center[1] + dx
                
                if 0 <= y < ny and 0 <= x < nx:
                    grid[x, y].eps = self.eps
                    grid[x, y].V = self.V
                    if self.fixed:
                        V_fixed[idx(y, x)] = self.V
                    elif idx(y, x) in V_fixed.keys():
                        del V_fixed[idx(y, x)]
                        
for i in range(nx):
    V_fixed[idx(0, i)] = 0.0
    V_fixed[idx(ny - 1, i)] = 0.0
for j in range(ny):
    V_fixed[idx(j, 0)] = 0.0
    V_fixed[idx(j, nx - 1)] = 0.0

Tx = Electrode((1, 1), (2*nx//3, ny//6), EPS_ZERO, 1.0, True)
Rx = Electrode((1, 1), (2*nx//3, 5*ny//6), EPS_ZERO, 0.5, True)
finger = Electrode((30, 20), (nx//3-nx//18, ny//2), 50*EPS_ZERO, 0.0, True)
glass = Electrode((nx//10, ny), (nx//2, ny//2), 5*EPS_ZERO, 0.0, False)

# why 1.0, 0.5, and 0.0 specifically?

grid = create_grid(nx,ny)

Tx.place(grid)
Rx.place(grid)
# finger.place(grid)
# glass.place(grid)

# LAPLACIAN SOLVING

dx = 90e-3/nx
dy = 90e-3/ny
dz = (dx + dy)/2

A = lil_matrix((N,N))
b = np.zeros(N)

w_left = w_right = (dy**2)/(2*dx**2 + 2*dy**2)
w_up = w_down = (dx**2)/(2*dx**2 + 2*dy**2)

b_old = np.zeros(N)

for k in V_fixed:
    b_old[k] = V_fixed[k]

for m in range(3000):
    
    b_new = np.copy(b_old)
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            if k in V_fixed:
                continue
            sum_neighbors = 0
            count = 0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    nk = idx(ni, nj)
                    sum_neighbors += b_old[nk]
                    count += 1
            b_new[k] = sum_neighbors / count
    diff = np.max(np.abs(b_new - b_old))
    if diff < 1e-4:
        print(f"Converged after {m} iterations.")
        break
    b_old = b_new

# explain clearly

V = b_old.reshape(nx, ny)
                
# Compute electric field
Ey, Ex = np.gradient(-V, dy, dx)
E_mag = np.sqrt(Ex**2 + Ey**2)
max_E = np.max(E_mag)

# -- Compute mutual capacitance --

eps_map = np.array([[grid[y, x].eps for x in range(nx)] for y in range(ny)])

# Average eps_map to match shape of Ex, Ey
eps_avg = 0.25 * (
    eps_map[:-1, :-1] + eps_map[1:, :-1] +
    eps_map[:-1, 1:] + eps_map[1:, 1:])

E_crop = E_mag[:-1, :-1] # Then crop E_mag to match
u = 0.5 * eps_avg * E_crop**2 # Energy density
W = np.sum(u) * dx * dy * dz # Total energy
V0 = 0.5 # Potential difference between Tx and Rx
C = 2 * W / V0**2
print(f"mutual capacitance C = {C} between Tx and Rx.")

# needs fixing !!!!!

# Plot potential field
plt.figure(figsize=(8, 6))
plt.imshow(V, cmap='plasma', origin='upper', extent=[0, nx*dx, ny*dy, 0])
plt.colorbar(label="Potential (V)")
plt.title("Electric Potential Field")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.show()

# Plot electric field vectors
plt.figure(figsize=(8, 6))
skip = 2
plt.imshow(V, cmap='plasma', origin='upper', extent=[0, nx*dx, ny*dy, 0], alpha=0.95)
X, Y = np.meshgrid(
    np.arange(0, nx, skip) * dx + dx,
    np.arange(0, ny, skip) * dy)
plt.quiver(X, Y, Ex[::skip, ::skip], Ey[::skip, ::skip],
           color='white', scale=max_E * 10, pivot='tail')

plt.gca().invert_yaxis()  # Ensure alignment with imshow(origin='upper')
plt.colorbar(label="Potential (V)")
plt.title("Electric Field Vectors")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.show()