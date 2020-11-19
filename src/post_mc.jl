#!/usr/bin/env julia

import Random
import Statistics
import Polynomials
import QuadGK
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("src/lib.jl")

# PARAMETERS {{{1

# Friction and inverse temperature
β, γ = 1, .001

# Parse the files
datadir = "data/γ=$γ/"
listfiles = readdir(datadir);
index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1]);
q0file = filter(s -> contains(s, r"q0.txt"), listfiles)[1];
p0file = filter(s -> contains(s, r"p0.txt"), listfiles)[1];
qfiles = filter(s -> contains(s, r"Δt=0.01-i=.*q.txt"), listfiles);
pfiles = filter(s -> contains(s, r"Δt=0.01-i=.*p.txt"), listfiles);
ξfiles = filter(s -> contains(s, r"Δt=0.01-i=.*ξ.txt"), listfiles);
qfiles = sort(qfiles, by=index);
pfiles = sort(pfiles, by=index);
ξfiles = sort(ξfiles, by=index);

# Iteration indices
indices = map(index, qfiles);

# Time step
Δt = parse(Float64, match(r"Δt=([^-]+)", qfiles[1]).captures[1]);

# Initial condition
q0 = DelimitedFiles.readdlm(string(datadir, q0file));
p0 = DelimitedFiles.readdlm(string(datadir, p0file));

# Underdamped limit
Du = diff_underdamped(β);
φ₀ = solution_underdamped();

# Store diffusion coefficients

for i in 1:length(indices)
    q = DelimitedFiles.readdlm(string(datadir, qfiles[i]));
    p = DelimitedFiles.readdlm(string(datadir, pfiles[i]));
    ξ = DelimitedFiles.readdlm(string(datadir, ξfiles[i]));

    print("Iteration: ", indices[i], ". ");
    D1 = Statistics.mean((q - q0).^2) / (2*indices[i]*Δt);
    control = ξ + φ₀.(q0, p0)/γ - φ₀.(q, p)/γ;
    D2 = (1/γ)*Du + D1 - Statistics.mean(control.^2)/(2*indices[i]*Δt);
    σ1 = Statistics.std((q - q0).^2/(2*indices[i]*Δt))
    σ2 = Statistics.std(((q - q0).^2 - control.^2)/(2*indices[i]*Δt))
    println(@Printf.sprintf("D₁ = %.3E, D₂ = %.3E, σ₁ = %.3E, σ₂ = %.3E",
                            D1, D2, σ1, σ2))
end

qend = DelimitedFiles.readdlm(string(datadir, qfiles[end]));
pend = DelimitedFiles.readdlm(string(datadir, pfiles[end]));
ξend = DelimitedFiles.readdlm(string(datadir, ξfiles[end]));
control_end = ξend + φ₀.(q0, p0)/γ - φ₀.(qend, pend)/γ;
tend = indices[end]*Δt

v1 = Statistics.var((qend - q0).^2/(2*tend))
v2 = Statistics.var(((qend - q0).^2 - control_end.^2)/(2*tend))

Plots.histogram((qend - q0).^2, bins=20)
Plots.histogram((qend - q0).^2, bins=20)

# f = Polynomials.fit(times[i÷10:i], mean_q²[i÷10:i], 1)
# D2 = f.coeffs[2] / 2
# D3 = (1/γ)*Du - Statistics.mean(ξ.^2)/(2*i*Δt) + D1

import Plots
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

