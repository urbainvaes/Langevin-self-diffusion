import numpy as np
import sympy as sym

s = 200
γ = 1000

mat = np.zeros((s+1, s+1))

for i in range(s+1):
    mat[i, i] = - γ*(s-i)
    if i > 0:
        mat[i-1, i] = sym.sqrt(s + 1 - i) * sym.sqrt(i)
    if i < s:
        mat[i+1, i] = - sym.sqrt(s - i) * sym.sqrt(i + 1)

# High-frequency in q
vec = np.zeros(s+1)
vec[-1] = 1
# vec[s // 2] = 1

e, q = np.linalg.eig(mat)

decomp = np.linalg.inv(q).dot(vec)

# np.linalg.norm(np.linalg.inv((mat)))

# e, q = np.linalg.eig(mat)
# min(abs(e))

# e, q = np.linalg.eig(mat[:-1, :-1])
# min(abs(e))
# np.linalg.norm(np.linalg.inv((mat[:-1, :-1])))

# e, q = np.linalg.eig(mat)

# decomp = np.linalg.inv(q).dot(vec)
# decomp

# e
