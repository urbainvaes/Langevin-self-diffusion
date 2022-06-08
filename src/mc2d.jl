# !/usr/bin/env julia
import Random
import Statistics
import Polynomials
import DelimitedFiles
import Printf
using ProfileCanvas
include("lib_galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
const γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1.;
const δ = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : .0;
control_type = length(ARGS) > 2 ? ARGS[3] : "galerkin"

# Batch number
batches = length(ARGS) > 3 ? ARGS[4] : "1/1";
ibatch = parse(Int, match(r"^[^/]*", batches).match);
nbatches = parse(Int, match(r"[^/]*$", batches).match);

# Inverse temperature
β = 1;

# Create directory for data
appenddir = (nbatches > 1 ? "/$ibatch" : "")
datadir = "data2d/$control_type-γ=$γ-δ=$δ$appenddir"
run(`rm -rf "$datadir"`)
run(`mkdir -p "$datadir"`)

# Potential and its derivative
V(q₁, q₂) = - cos(q₁)/2 - cos(q₂)/2 - δ*cos(q₁)*cos(q₂);
d₁V(q₁, q₂) = sin(q₁) * (1/2 + δ*cos(q₂));
d₂V(q₁, q₂) = sin(q₂) * (1/2 + δ*cos(q₁));

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
Random.seed!(ibatch);
Random.seed!(floor(Int, 1e6*Random.rand()));

# Number of particles
np_total = 5000;
np = ceil(Int, (np_total / nbatches))

# Time step and final time
Δt = .01;
tf = ceil(Int, 100/γ);

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q0, p0 = Sampling.sample_gibbs_2d(V, β, np);
q, p, ξ = copy(q0), copy(p0), zeros(np, 2);

# Control
if control_type == "galerkin"
    # !!! φ is solution of -Lφ = p (negative sign) !!!
    Dc, ψ, ∂ψ = Spectral.get_controls(γ, δ, true)
elseif control_type == "underdamped"
    Dc = (1/γ)*Underdamped.diff_underdamped(β);
    φ₀ = Underdamped.solution_underdamped();
    ψ(q, p) = Underdamped.φ₀(q, p)/γ
    ∂ψ(q, p) = Underdamped.∂φ₀(q, p)/γ
end
println(@Printf.sprintf("Dc = %.3E", Dc))

# Covariance matrix of (Δw, ∫ e¯... dW)
const rt_cov = Sampling.root_cov(γ, Δt);
α = exp(-γ*Δt);

# Number of saves
nsave = 100;
nslice = niter ÷ nsave;

# Write initial condition to file
DelimitedFiles.writedlm("$datadir/Δt=$Δt-q0.txt", q0)
DelimitedFiles.writedlm("$datadir/Δt=$Δt-p0.txt", p0)

@views function main()

    # Integrate the evolution
    for i = 1:niter
        global p, q, ξ

        # Generate Gaussian increments
        gaussian_increments = rt_cov*Random.randn(2, np)
        Δw₁, gs₁ = gaussian_increments[1, :], gaussian_increments[2, :]
        gaussian_increments = rt_cov*Random.randn(2, np)
        Δw₂, gs₂ = gaussian_increments[1, :], gaussian_increments[2, :]

        # qper .= q - 2π*floor.(Int, (q.+π)/2π);
        ξ[:, 1] .+= ∂ψ.(q[:, 1], p[:, 1]) .* (sqrt(2γ/β)*Δw₁)
        ξ[:, 2] .+= ∂ψ.(q[:, 2], p[:, 2]) .* (sqrt(2γ/β)*Δw₂)
        p[:, 1] .+= - (Δt/2)*d₁V.(q[:, 1], q[:, 2]);
        p[:, 2] .+= - (Δt/2)*d₂V.(q[:, 1], q[:, 2]);
        q[:, 1] .+= Δt*p[:, 1];
        q[:, 2] .+= Δt*p[:, 2];
        p[:, 1] .+= - (Δt/2)*d₁V.(q[:, 1], q[:, 2]);
        p[:, 2] .+= - (Δt/2)*d₂V.(q[:, 1], q[:, 2]);
        p[:, 1] .= α*p[:, 1] + sqrt(2γ/β)*gs₁
        p[:, 2] .= α*p[:, 2] + sqrt(2γ/β)*gs₂

        if i % nslice == 0
            DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-p.txt", p)
            DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-q.txt", q)
            DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-ξ.txt", ξ)
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
            control1 = ξ[:, 1] + ψ.(q0[:, 1], p0[:, 1]) - ψ.(q[:, 1], p[:, 1]);
            control2 = ξ[:, 2] + ψ.(q0[:, 2], p0[:, 2]) - ψ.(q[:, 2], p[:, 2]);
            term11 = Dc .- control1.^2 / (2*i*Δt) + term11
            term22 = Dc .- control2.^2 / (2*i*Δt) + term22
            D11 = Statistics.mean(term11)
            D12 = Statistics.mean(term12)
            D22 = Statistics.mean(term22)
            σ11 = Statistics.std(term11)
            σ22 = Statistics.std(term22)
            println(@Printf.sprintf("D₁₁ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
            D11, D22, σ11, σ22))
        end
    end
end
