import numpy as np
import sympy as sym

N = 20

S1, S2 = 0, 0

for i in range(1, N + 1):
    S1 += (N + 1 - i)**2 * (N + i)**2
    for j in range(i + 1, N + 1):
        S2 += (j + 1 -i)**2 * (j + i)**2

var = (18*S1 + 36*S2) / (N*(N+1)*(N+2))**2
