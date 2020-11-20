#!/usr/bin/env julia
import Random
import Statistics
import Polynomials
import QuadGK
import DelimitedFiles
import Printf
include("src/lib.jl")

# PARAMETERS {{{1

# Friction and inverse temperature
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : .01;
δ, β = .2, 1;

# Create directory for data
datadir = "data2d/γ=$γ-δ=$δ"
run(`rm -rf "$datadir"`)
run(`mkdir -p "$datadir"`)

# Potential and its derivative
V = (q₁, q₂) -> - cos(q₁)/2 - cos(q₂)/2 - δ*cos(q₁)*cos(q₂);
d₁V = (q₁, q₂) -> sin(q₁)/2 + δ*sin(q₁)*cos(q₂);
d₂V = (q₁, q₂) -> sin(q₂)/2 + δ*cos(q₁)*sin(q₂);

# V = (q₁, q₂) -> - cos(q₁)/2 - cos(q₂)/2;
# d₁V = (q₁, q₂) -> sin(q₁)/2;
# d₂V = (q₁, q₂) -> sin(q₂)/2;

# V = (q₁, q₂) -> - cos(q₁)/2 - cos(q₂)/2
# d₁V = (q₁, q₂) -> sin(q₁)/2
# d₂V = (q₁, q₂) -> sin(q₂)/2

# V = (q₁, q₂) -> - cos(q₂)/2;
# d₁V = (q₁, q₂) -> 0;
# d₂V = (q₁, q₂) -> sin(q₂)/2;

# V = (q₁, q₂) -> - cos(q₁)/2;
# d₁V = (q₁, q₂) -> sin(q₁)/2;
# d₂V = (q₁, q₂) -> 0;

# MONTE CARLO METHOD {{{1

# Fix seed
Random.seed!(0);

# Number of particles
np = 5000;

# Time step and final time
Δt = .01;
tf = ceil(Int, 100/γ);

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q0, p0 = sample_gibbs_2d(V, β, np);
q, p, ξ = copy(q0), copy(p0), zeros(np, 2);

# Covariance between Δw and ∫_{0}^{Δt} e^{-γ(Δt-s)} ds
α = exp(-γ*Δt)
rt_cov = root_cov(γ, Δt)

# Track q2 at each iteration
mean_q² = zeros(niter, 3);
DelimitedFiles.writedlm("$datadir/Δt=$Δt-mean_q2.txt", "");
DelimitedFiles.writedlm("$datadir/Δt=$Δt-q0.txt", q0)
DelimitedFiles.writedlm("$datadir/Δt=$Δt-p0.txt", p0)

times = Δt*(1:niter) |> collect;
nsave = 1000;
nslice = niter ÷ nsave;

# Underdamped limit
Du = diff_underdamped(β);
φ₀ = solution_underdamped();

# Integrate the evolution
for i = 1:niter
    global p, q

    # Generate Gaussian increments
    gaussian_increments = rt_cov*Random.randn(2, np)
    Δw₁, gs₁ = gaussian_increments[1, :], gaussian_increments[2, :]
    gaussian_increments = rt_cov*Random.randn(2, np)
    Δw₂, gs₂ = gaussian_increments[1, :], gaussian_increments[2, :]

    ξ[:, 1] += (∇p_φ₀.(q[:, 1], p[:, 1])/γ) .* (sqrt(2γ/β)*Δw₁)
    ξ[:, 2] += (∇p_φ₀.(q[:, 2], p[:, 2])/γ) .* (sqrt(2γ/β)*Δw₂)
    p[:, 1] += - (Δt/2)*d₁V.(q[:, 1], q[:, 2]);
    p[:, 2] += - (Δt/2)*d₂V.(q[:, 1], q[:, 2]);
    q[:, 1] += Δt*p[:, 1];
    q[:, 2] += Δt*p[:, 2];
    p[:, 1] += - (Δt/2)*d₁V.(q[:, 1], q[:, 2]);
    p[:, 2] += - (Δt/2)*d₂V.(q[:, 1], q[:, 2]);
    p[:, 1] = α*p[:, 1] + sqrt(2γ/β)*gs₁
    p[:, 2] = α*p[:, 2] + sqrt(2γ/β)*gs₂

    mean_q²[i, 1] = Statistics.mean((q[:, 1]-q0[:, 1]).^2)
    mean_q²[i, 2] = Statistics.mean((q[:, 1]-q0[:, 1]).*(q[:, 2]-q0[:, 2]))
    mean_q²[i, 3] = Statistics.mean((q[:, 2]-q0[:, 2]).^2)
    if i % nslice == 0
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-p.txt", p)
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-q.txt", q)
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-ξ.txt", ξ)
        open("$datadir/Δt=$Δt-mean_q2.txt", "a") do io
            DelimitedFiles.writedlm(io, mean_q²[i-nslice+1:i, :])
        end
        term11 = (q[:, 1] - q0[:, 1]).^2 / (2*i*Δt)
        term12 = (q[:, 1] - q0[:, 1]).*(q[:, 2] - q0[:, 2]) / (2*i*Δt)
        term22 = (q[:, 2] - q0[:, 2]).^2 / (2*i*Δt)
        D11 = Statistics.mean(term11)
        D12 = Statistics.mean(term12)
        D22 = Statistics.mean(term22)
        σ11 = Statistics.std(term11)
        σ22 = Statistics.std(term22)
        println("Pogress: ", (1000*i) ÷ niter, "‰. ")
        println(@Printf.sprintf("D₁₁ = %.3E, D₁₂ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                                D11, D12, D22, σ11, σ22))
        control1 = ξ[:, 1] + φ₀.(q0[:, 1], p0[:, 1])/γ - φ₀.(q[:, 1], p[:, 1])/γ;
        control2 = ξ[:, 2] + φ₀.(q0[:, 2], p0[:, 2])/γ - φ₀.(q[:, 2], p[:, 2])/γ;
        term11 = (1/γ)*Du .- control1.^2 / (2*i*Δt) + term11
        term22 = (1/γ)*Du .- control2.^2 / (2*i*Δt) + term22
        D11 = Statistics.mean(term11)
        D12 = Statistics.mean(term12)
        D22 = Statistics.mean(term22)
        σ11 = Statistics.std(term11)
        σ22 = Statistics.std(term22)
        println(@Printf.sprintf("D₁₁ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                                D11, D22, σ11, σ22))
    end
end
