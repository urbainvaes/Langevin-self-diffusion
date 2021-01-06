#!/usr/bin/env julia
import Random
import Statistics
import Polynomials
import DelimitedFiles
import Printf
include("lib_galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : .01;
control_type = length(ARGS) > 1 ? ARGS[2] : "galerkin";

# Batch number
batches = length(ARGS) > 2 ? ARGS[3] : "1/1";
ibatch = parse(Int, match(r"^[^/]*", batches).match);
nbatches = parse(Int, match(r"[^/]*$", batches).match);

# Inverse temperature
β = 1;

# Create directory for data
appenddir = (nbatches > 1 ? "/$ibatch" : "")
datadir = "data2d/$control_type-γ=$γ$appenddir"
run(`rm -rf "$datadir"`)
run(`mkdir -p "$datadir"`)

# Potential and its derivative
V(q) = (1 - cos(q))/2;
dV(q) = sin(q)/2;
# V(q) = 0;
# dV(q) = 0;

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
q0, p0 = sample_gibbs(V, β, np);
q, p, ξ = copy(q0), copy(p0), zeros(np);

# Control
if control_type == "galerkin"
    # !!! φ is solution of -Lφ = p (negative sign) !!!
    Dc, ψ, ∂ψ = get_controls(γ, true, false)
elseif control_type == "underdamped"
    Dc = (1/γ)*diff_underdamped(β);
    φ₀ = solution_underdamped();
    ψ(q, p) = φ₀(q, p)/γ
    ∂ψ(q, p) = ∂φ₀(q, p)/γ
end
println(@Printf.sprintf("Dc = %.3E", Dc))

# Covariance matrix of (Δw, ∫ e¯... dW)
rt_cov = root_cov(γ, Δt);

# Number of saves
nsave = 1000;
nslice = niter ÷ nsave;

# Write initial condition to file
DelimitedFiles.writedlm("$datadir/Δt=$Δt-q0.txt", q0)
DelimitedFiles.writedlm("$datadir/Δt=$Δt-p0.txt", p0)

# Integrate the evolution
for i = 1:niter
    global p, q, ξ

    # Generate Gaussian increments
    gaussian_increments = rt_cov*Random.randn(2, np)
    Δw, gs = gaussian_increments[1, :], gaussian_increments[2, :]

    ξ += ∂ψ.(q, p) .* (sqrt(2γ/β)*Δw)
    p += - (Δt/2)*dV.(q);
    q += Δt*p;
    p += - (Δt/2)*dV.(q);
    p = exp(-γ*Δt)*p + sqrt(2γ/β)*gs

    if i % nslice == 0
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-p.txt", p)
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-q.txt", q)
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-ξ.txt", ξ)
        print("Pogress: ", (1000*i) ÷ niter, "‰. ")
        control = ξ + ψ.(q0, p0) - ψ.(q, p);
        D1 = Statistics.mean((q - q0).^2) / (2*i*Δt)
        D2 = Dc + D1 - Statistics.mean(control.^2)/(2*i*Δt);
        σ1 = Statistics.std((q - q0).^2/(2*i*Δt))
        σ2 = Statistics.std(((q - q0).^2 - control.^2)/(2*i*Δt))
        println(@Printf.sprintf("D₁ = %.3E, D₂ = %.3E, σ₁ = %.3E, σ₂ = %.3E",
                                D1, D2, σ1, σ2))
    end
end
