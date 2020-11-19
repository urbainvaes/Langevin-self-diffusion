#!/usr/bin/env julia

import Plots
import Random
import Statistics
import SparseArrays
import LinearAlgebra
import QuadGK
import Polynomials

sparse = SparseArrays;
linalg = LinearAlgebra;

import PyCall
sym = PyCall.pyimport("sympy");

# PARAMETERS {{{1

# Friction and inverse temperature
# γ, β = .01, 1;
γ, β = .0001, 1;

# Potential and its derivative
V = q -> (1 .- cos.(q))/2;
dV = q -> sin.(q)/2;

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
Du = (1/Zb)*(1/β)*8*π^2*integral;

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

# Sample from the Gibbs distribution
function sample_gibbs(np)
    p = (1/sqrt(β)) * Statistics.randn(np);
    q, naccepts = zeros(Float64, np), 0;
    while naccepts < length(q)
        v = Statistics.rand()
        u = -π + 2π*Statistics.rand();
        if v <= exp(-β*V(u))/exp(-β*V(0))
            naccepts += 1;
            q[naccepts] = u;
        end
    end
    return q, p
end

# Number of particles
np = 1000;

# Time step and final time
Δt = .01;
tf = 1000000;

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q0, p0 = sample_gibbs(np);
q, p, ξ = q0, p0, zeros(np);

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

nsave = 100
qsave = zeros(nsave, np)
qsave[0, :] = q0

# Integrate the evolution
for i = 1:niter
    global p, q
    method = "geometric_langevin"

    if method == "geometric_langevin"
        # Generate Gaussian increments
        gaussian_increments = rt_cov*Random.randn(2, np)
        Δw, gs = gaussian_increments[1, :], gaussian_increments[2, :]

        ξ += ∇p_φ₀.(q, p) .* Δw
        p += - (Δt/2)*dV(q);
        q += Δt*p;
        p += - (Δt/2)*dV(q);
        p = α*p + sqrt(2γ/β)*gs

    elseif method == "euler_maruyama"
        Δw = sqrt(Δt)*Random.randn(np);

        ξ += ∇p_φ₀.(q, p) .* Δw
        q += Δt*p;
        p += - Δt*dV(q) - Δt*γ*p + Δw*sqrt(2*γ/β)
    end

    mean_q²[i] = Statistics.mean((q-q0).^2)

    if i % (niter ÷ nsave) == 0
        qsave[i ÷ (niter ÷ nsave), :] = q
    end

    if i % (niter ÷ 100) == 0
        print("Pogress: ", (100*i) ÷ niter, "%. ")
        f = Polynomials.fit(times[i÷10:i], mean_q²[i÷10:i], 1)
        println("D = ", f.coeffs[2] / 2)
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
