#!/usr/bin/env julia
import Plots
import LinearAlgebra
import Random
import Statistics
import QuadGK
import Polynomials
linalg = LinearAlgebra;
include("src/lib.jl")

# PARAMETERS {{{1

# Friction and inverse temperature
γ, β, α = 0, 1, 1;
γ, β, α = 1, 1, 0;

# Potential and its derivative
V = (q₁, q₂) -> - cos(q₁)/2 - cos(q₂)/2 - α*cos(q₁)*cos(q₂);
d₁V = (q₁, q₂) -> sin(q₁)/2 + α*sin(q₁)*cos(q₂);
d₂V = (q₁, q₂) -> sin(q₂)/2 + α*cos(q₁)*sin(q₂);

V = (q₁, q₂) -> - cos(q₁)/2 - cos(q₂)/2;
d₁V = (q₁, q₂) -> sin(q₁)/2;
d₂V = (q₁, q₂) -> sin(q₂)/2;

# V = (q₁, q₂) -> - cos(q₁)/2 - cos(q₂)/2
# d₁V = (q₁, q₂) -> sin(q₁)/2
# d₂V = (q₁, q₂) -> sin(q₂)/2

# V = (q₁, q₂) -> - cos(q₂)/2;
# d₁V = (q₁, q₂) -> 0;
# d₂V = (q₁, q₂) -> sin(q₂)/2;

# V = (q₁, q₂) -> - cos(q₁)/2;
# d₁V = (q₁, q₂) -> sin(q₁)/2;
# d₂V = (q₁, q₂) -> 0;

function ∇p_φ₀(q, p)
    E = V(q) + p*p/2
    E > E₀ ? sign(p)*p*2π/S(E) : 0
end

# MONTE CARLO METHOD {{{1

# Fix seed
# Random.seed!(0);

# Number of particles
np = 5000;

# Time step and final time
Δt = .01;
tf = 1000;

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q0, p0 = sample_gibbs_2d(V, β, np);
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

# nsave = 100
# qsave = zeros(nsave, np)
# qsave[0, :] = q0

# Integrate the evolution
for i = 1:niter
    global p, q

    # Generate Gaussian increments
    gaussian_increments = rt_cov*Random.randn(2, np)
    Δw₁, gs₁ = gaussian_increments[1, :], gaussian_increments[2, :]
    gaussian_increments = rt_cov*Random.randn(2, np)
    Δw₂, gs₂ = gaussian_increments[1, :], gaussian_increments[2, :]

    # # ξ += ∇p_φ₀.(q, p) .* Δw
    p[:, 1] += - (Δt/2)*d₁V.(q[:, 1], q[:, 2]);
    p[:, 2] += - (Δt/2)*d₂V.(q[:, 1], q[:, 2]);
    q[:, 1] += Δt*p[:, 1];
    q[:, 2] += Δt*p[:, 2];
    p[:, 1] += - (Δt/2)*d₁V.(q[:, 1], q[:, 2]);
    p[:, 2] += - (Δt/2)*d₂V.(q[:, 1], q[:, 2]);
    p[:, 1] = α*p[:, 1] + sqrt(2γ/β)*gs₁
    p[:, 2] = α*p[:, 2] + sqrt(2γ/β)*gs₂

    # # ξ += ∇p_φ₀.(q, p) .* Δw
    # p[:, 1] += - Δt*d₁V.(q[:, 1], q[:, 2]) - Δt*γ*p[:, 1] + Δw₁*sqrt(2*γ/β);
    # p[:, 2] += - Δt*d₂V.(q[:, 1], q[:, 2]) - Δt*γ*p[:, 2] + Δw₂*sqrt(2*γ/β);
    # q[:, 1] += Δt*p[:, 1];
    # q[:, 2] += Δt*p[:, 2];

    # if i % (niter ÷ nsave) == 0
    #     qsave[i ÷ (niter ÷ nsave), :] = q
    # end
    nprints = 100
    if i % (niter ÷ nprints) == 0
        D = (q - q0)' * (q - q0) / np
        print("Pogress: ", (nprints*i) ÷ niter, "‰.")
        # f = Polynomials.fit(times[i÷10:i], mean_q²[i÷10:i], 1)
        # D2 = f.coeffs[2] / 2
        # D3 = (1/γ)*Du - (1/γ)*Statistics.mean(ξ.^2)/(i*Δt) + D1
        println(" D₁₁ = ", D[1, 1] / (2*i*Δt),
                " D₂₂ = ", D[2, 2] / (2*i*Δt),
                " D₁₂ = ", D[1, 2] / (2*i*Δt))
    end
end

# Check energy conservation is Δt²
# E0 = V.(q0[:, 1], q0[:, 2]) + p0[:, 1].^2/2 + p0[:, 2].^2/2;
# E = V.(q[:, 1], q[:, 2]) + p[:, 1].^2/2 + p[:, 2].^2/2;
# Statistics.mean(abs.(E - E0))

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
