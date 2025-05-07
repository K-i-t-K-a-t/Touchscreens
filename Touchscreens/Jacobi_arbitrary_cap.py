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

electrodes = dict()

class Electrode:
    def __init__(self, size, center, permittivity, voltage, V_fixed):
        self.size = size
        self.center = center
        self.eps = permittivity
        self.V = voltage
        self.fixed = V_fixed
        
    def place(self, grid): # function to initialise an electrode of desired geometry and permittivity + voltage on the grid
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
                        
    # def charge(self, eps_avg, Ex, Ey, dx, dy):
    #     Q = 0.0
    
    #     h, w = self.size
    #     cy, cx = self.center
    #     y0 = cy - h // 2
    #     y1 = cy + h // 2
    #     x0 = cx - w // 2
    #     x1 = cx + w // 2
    
    #     # LEFT and RIGHT edges
    #     for j in range(y0, y1 + 1):
    #         Q += -eps_avg[x0 - 1, j] * Ex[x0 - 1, j] * dy
    #         Q += eps_avg[x1 + 1, j] * Ex[x1 + 1, j] * dy
    
    #     # TOP and BOTTOM edges
    #     for i in range(x0, x1 + 1):
    #         Q += -eps_avg[i, y0 - 1] * Ey[i, y0 - 1] * dx
    #         Q += eps_avg[i, y1 + 1] * Ey[i, y1 + 1] * dx
    
    #     # CORNERS with diagonal normals
    #     corner_normals = {
    #         (x0 - 1, y0 - 1): (-1, -1),
    #         (x1 + 1, y0 - 1): (1, -1),
    #         (x0 - 1, y1 + 1): (-1, 1),
    #         (x1 + 1, y1 + 1): (1, 1),
    #     }
    
    #     for (i, j), (nx, ny) in corner_normals.items():
    #         normal = np.array([nx, ny]) / np.sqrt(2)
    #         E_corner = np.array([Ex[i, j], Ey[i, j]])
    #         dA = 0.5 * dx * dy
    #         Q += eps_avg[i, j] * np.dot(E_corner, normal) * dA
    
    #     return Q
    
    def charge(self, eps_map, Ex, Ey, dx, dy):
        Q = 0.0
        
        h, w = self.size
        cy, cx = self.center
        
        bottom = cy - h // 2
        top = cy + h // 2
        left = cx - w // 2
        right = cx + w // 2
    
        # Top and Bottom Sides: integrate along the y-direction
        for i in range(left + 1, right):
            # Top side (increasing y)
            Q += eps_map[i, top] * Ey[i, top] * dy
            # Bottom side (decreasing y)
            Q += -eps_map[i, bottom] * Ey[i, bottom] * dy
    
        # Left and Right Sides: integrate along the x-direction
        for j in range(bottom + 1, top):
            # Left side (increasing x)
            Q += -eps_map[left, j] * Ex[left, j] * dx
            # Right side (decreasing x)
            Q += eps_map[right, j] * Ex[right, j] * dx
    
        # Corners: account for diagonal contributions
        corner_normals = {
            (left, bottom): (-1, -1),  # Bottom-left corner
            (right, bottom): (1, -1),  # Bottom-right corner
            (left, top): (-1, 1),      # Top-left corner
            (right, top): (1, 1),      # Top-right corner
        }
    
        # Diagonal normal vectors for the corner points
        for (i, j), (nx, ny) in corner_normals.items():
            normal = np.array([nx, ny]) / np.sqrt(2)  # Normalized diagonal direction
            E_corner = np.array([Ex[i, j], Ey[i, j]])  # Electric field vector at the corner
            dA = 0.5 * dx * dy  # Approximate area of a corner (triangle)
            Q += eps_map[i, j] * np.dot(E_corner, normal) * dA  # Dot product for energy density at corner
    
        return Q
        
for i in range(nx):
    V_fixed[idx(0, i)] = 0.0
    V_fixed[idx(ny - 1, i)] = 0.0
for j in range(ny):
    V_fixed[idx(j, 0)] = 0.0
    V_fixed[idx(j, nx - 1)] = 0.0

Tx = Electrode((3, 3), (2*nx//3, ny//6), EPS_ZERO, 1.0, True)
Rx = Electrode((3, 3), (2*nx//3, 5*ny//6), EPS_ZERO, 0.5, True)
finger = Electrode((30, 20), (nx//3-nx//18, ny//2), 50*EPS_ZERO, 0.0, True)
glass = Electrode((nx//10, ny), (nx//2, ny//2), 5*EPS_ZERO, 0.0, False)

# why 1.0, 0.5, and 0.0 specifically?

grid = create_grid(nx,ny)

Tx.place(grid)
Rx.place(grid)
finger.place(grid)
glass.place(grid)

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

for m in range(250):
    
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
# eps_avg = 0.25 * (
#     eps_map[:-1, :-1] + eps_map[1:, :-1] +
#     eps_map[:-1, 1:] + eps_map[1:, 1:])

# E_crop = E_mag[:-1, :-1] # Then crop E_mag to match

Q_Tx = Tx.charge(eps_map, Ex, Ey, dx, dy)
Q_Rx = Rx.charge(eps_map, Ex, Ey, dx, dy)
print(f'Charge Q = {Q_Tx} C on the transmitting electrode Tx')
print(f'Charge Q = {Q_Rx} C on the receiving electrode Rx')

W = 0.5 * (Tx.V * Q_Tx + Rx.V * Q_Rx)
C = 2 * W / (Tx.V - Rx.V)**2
print(f'Mutual capacitance C = {C} F between Tx and Rx')

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