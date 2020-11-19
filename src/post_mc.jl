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
# γ, β = .0001, 1;

# Potential and its derivative
V = q -> (1 - cos(q))/2;
dV = q -> sin(q)/2;

# V = q -> 0;
# dV = q -> 0;

# UNDERDAMPED LIMIT {{{1
# This is only for the case of the cosine potential!
# V(q) = (1 - cos(q))/2
import SpecialFunctions
import Elliptic

inf = 100;
Zb = (2π)^(3/2) / β^(1/2) * exp(-β/2) * SpecialFunctions.besseli(0, β/2);
S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z);
integral = QuadGK.quadgk(z -> exp(-β*z) / S(z), 1, inf)[1];
Du = diff_underdamped(β)

# Calculate limit in underdamped limit
E₀, Estep, Einf = 1, .001, 20
Es = E₀:Estep:Einf

function ∇p_φ₀(q, p)
    E = V(q) + p*p/2
    E > E₀ ? sign(p)*p*2π/S(E) : 0
end

# MONTE CARLO METHOD {{{1

# Fix seed
# Random.seed!(0);

# Number of particles
np = 1000;

# Time step and final time
Δt = .01;
tf = 1000;
# tf = 1000000;

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q0, p0 = sample_gibbs(V, β, np);
q, p, ξ = copy(q0), copy(p0), zeros(np);

# Covariance between Δw and ∫_{0}^{Δt} e^{-γ(Δt-s)} ds
α = exp(-γ*Δt)

if γ > 0
    cov = [Δt (1-α)/γ; (1-α)/γ (1-α*α)/(2γ)];
    rt_cov = (linalg.cholesky(cov).L);
elseif γ == 0
    rt_cov = sqrt(Δt)*[1 0; 1 0];
end

# Track q2 at each iteration
mean_q² = zeros(niter)
times = Δt*(1:niter) |> collect

# Integrate the evolution
for i = 1:niter
    global p, q
    method = "geometric_langevin"

    # Generate Gaussian increments
    gaussian_increments = rt_cov*Random.randn(2, np)
    Δw, gs = gaussian_increments[1, :], gaussian_increments[2, :]

    ξ += ∇p_φ₀.(q, p) .* Δw
    p += - (Δt/2)*dV.(q);
    q += Δt*p;
    p += - (Δt/2)*dV.(q);
    p = α*p + sqrt(2γ/β)*gs

    mean_q²[i] = Statistics.mean((q-q0).^2)

    if i % (niter ÷ nsave) == 0
        DelimitedFiles.writedlm("Δt=$Δt-i=$i-γ=$γ-p.txt")
        DelimitedFiles.writedlm("Δt=$Δt-i=$i-γ=$γ-q.txt")
    end

    if i % (niter ÷ 1000) == 0
        print("Pogress: ", (1000*i) ÷ niter, "‰. ")
        D1 = Statistics.mean((q - q0).^2) / (2*i*Δt)
        f = Polynomials.fit(times[i÷10:i], mean_q²[i÷10:i], 1)
        D2 = f.coeffs[2] / 2
        D3 = (1/γ)*Du - (1/γ)*Statistics.mean(ξ.^2)/(i*Δt) + D1
        println("D₁ = ", D1, " D₂ = ", D2, " D₃ = ", D3)

        # Estimation of the effective diffusion with control variate
        D = (1/γ)*Du - (1/γ)*Statistics.mean(ξ.^2)/tf + Statistics.mean((q - q0).^2)/(2*tf)
    end

end

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

