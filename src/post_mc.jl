#!/usr/bin/env julia

import Plots
import Random
import Statistics
import Polynomials
import QuadGK
import LinearAlgebra
import DelimitedFiles
linalg = LinearAlgebra;
include("src/lib.jl")

# PARAMETERS {{{1

# Friction and inverse temperature
γ, β = 1, 1;

listfiles = readdir("data/γ=0.001")
index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1])
qfiles = filter(s -> contains(s, r"Δt=0.01-i=.*q.txt"), listfiles)
pfiles = filter(s -> contains(s, r"Δt=0.01-i=.*p.txt"), listfiles)
ξfiles = filter(s -> contains(s, r"Δt=0.01-i=.*ξ.txt"), listfiles)
qfiles = sort(qfiles, by=index)
pfiles = sort(pfiles, by=index)
ξfiles = sort(ξfiles, by=index)

# V = q -> 0;
# Check energy conservation is Δt²
# ΔE = (V.(q) + p.^2/2) - (V.(q0) + p0.^2/2)
# Statistics.mean(abs.(ΔE))

f = Polynomials.fit(times[niter÷10:end], mean_q²[niter÷10:end], 1)
D = f.coeffs[2] / 2

Plots.plot(times, mean_q²)
Plots.plot!(f, times)
Plots.plot(times, mean_q²./(2*times))

# Squared position (shorthand for this is q.*q)
q2 = broadcast(*, q - q0, q - q0);

# Estimation of the effective diffusion
D = Statistics.mean(q2) / (2*tf)
Plots.plot(mean_q², bins=20)

Plots.histogram(q2, bins=20)

Statistics.var(q - q0) / (2*tf)
Statistics.var(q - q0 - ξ) / (2*tf)

# Estimation of the effective diffusion with control variate
D = (1/γ)*Du - (1/γ)*Statistics.mean(ξ.^2)/tf + Statistics.mean((q - q0).^2)/(2*tf)

Plots.histogram((q - q0)/sqrt(tf), bins=10)
# Plots.histogram(q0, bins=-π:(π/10):π)
# Plots.histogram(p0, bins=20)

