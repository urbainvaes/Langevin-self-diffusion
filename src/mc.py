#!/usr/bin/env python
import numpy

# Fix seed
numpy.random.seed(0)

# Number of particles
np = 10**3

# Position and momentum
q = numpy.zeros(np)
p = numpy.zeros(np)

# Time step and final time
Δt = 1e-2
tf = 10000

# Friction and inverse temperature
γ, β = 1, 1

# Potential and its derivative
V = lambda x: (1 - cos(x))/2

for i in range(int(tf/Δt)):
    Δw = numpy.sqrt(Δt) * numpy.random.randn(np)
    q = q + Δt*p
    p = p - Δt*γ*p + numpy.sqrt(2*γ/β) * Δw

# Squared position (shorthand for this is q.*q)
q2 = q*q

# Estimation of the effective diffusion
D = numpy.mean(q2) / (2*tf)
print(D)
[200~q = numpy.zeros(np)[201~
[200~q = numpy.zeros(np)[201~
[200~np = 10**3[201~
