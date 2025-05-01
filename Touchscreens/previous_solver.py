# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:16:04 2025

@author: PC
"""

A = lil_matrix((N, N))
b = np.zeros(N)

for i in range(ny):
    for j in range(nx):
        k = idx(i, j)

        if k in V_fixed:
            A[k, k] = 1.0
            b[k] = V_fixed[k]
            continue

        total_weight = 0.0
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                nk = idx(ni, nj)
                eps_avg = 0.5 * (grid[nj, ni].eps + grid[j, i].eps)
                weight = eps_avg / dx**2
                A[k, nk] = -weight
                total_weight += weight
        A[k, k] = total_weight

# Solve for potential
phi = spsolve(A.tocsr(), b)
V = phi.reshape((ny, nx))