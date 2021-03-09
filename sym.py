import numpy as np
import sympy as sym

<<<<<<< HEAD
N = 2
=======
# N = 3000
N = 1
>>>>>>> 93b75e6626e0b767dcf63bf7541a4a0da20fedba

S1, S2 = 0, 0

for i in range(1, N + 1):
    S1 += (N + 1 - i)**2 * (N + i)**2
    for j in range(i + 1, N + 1):
        S2 += (j + 1 -i)**2 * (j + i)**2

var = (18*S1 + 36*S2) / (N*(N+1)*(N+2))**2
