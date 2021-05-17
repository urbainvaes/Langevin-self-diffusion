#!/usr/bin/env julia
# import Plots
import Random
import Statistics
import DelimitedFiles
import Printf
import LinearAlgebra
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : .01;
ν = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 1;
control = length(ARGS) > 2 ? ARGS[3] : "gle";

# Batch number
batches = length(ARGS) > 2 ? ARGS[3] : "1/1";
ibatch = parse(Int, match(r"^[^/]*", batches).match);
nbatches = parse(Int, match(r"[^/]*$", batches).match);

# Inverse temperature
β = 1;

# Create directory for data
appenddir = (nbatches > 1 ? "/$ibatch" : "")
datadir = "data_gle/$control_type-γ=$γ$appenddir"
run(`rm -rf "$datadir"`)
run(`mkdir -p "$datadir"`)

# Potential and its derivative
V(q) = (1 - cos(q))/2;
dV(q) = sin(q)/2;

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
q0, p0 = Sampling.sample_gibbs(V, β, np);
z0 = (1/sqrt(β)) * Statistics.randn(np);
q, p, z, ξ = copy(q0), copy(p0), copy(z0), zeros(np);

# Control
ν = .2

if control == "gle"
    Dc, dz_ψ = Underdamped.get_controls_gle(γ, ν, false)
    _, ψ, _ = Underdamped.get_controls(γ, false)
else
    Dc, ψ, ∂ψ = Underdamped.get_controls(γ, false)
end
println(@Printf.sprintf("Dc = %.3E", Dc))

# Covariance matrix of (Δw, ∫ e¯... dW)
operator_mean, rt_cov = Sampling.gle_params(β, γ, ν, Δt)

# Number of saves
nsave = 1000;
nslice = niter ÷ nsave;

# Write initial condition to file
DelimitedFiles.writedlm("$datadir/Δt=$Δt-q0.txt", q0)
DelimitedFiles.writedlm("$datadir/Δt=$Δt-p0.txt", p0)

# Integrate the evolution
for i = 1:niter
    if i % 1000 == 0
        print(".")
    end
    global p, q, z, ξ

    # Generate Gaussian increments
    gauss_inc = rt_cov*Random.randn(3, np)
    Δw, gp, gz = gauss_inc[1, :], gauss_inc[2, :], gauss_inc[3, :]

    if control == "gle"
        ξ += dz_ψ.(q, p) .* (sqrt(2/(β*γ*ν^2))*Δw)
    else
        ξ += ∂ψ.(q, p) .* (sqrt(2γ/β)*Δw)
    end
    p += - (Δt/2)*dV.(q);
    q += Δt*p;
    p += - (Δt/2)*dV.(q);
    mpz = operator_mean*[p, z]
    p = mpz[1] + gp
    z = mpz[2] + gz

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

# control = ξ + ψ.(q0, p0) - ψ.(q, p);

# dx, xmax = .1, 5
# Plots.histogram(((q - q0).^2 - control.^2)/(2*niter*Δt), bins=0:dx:xmax, normalize=:pdf)
# Plots.histogram((q - q0).^2/(2*niter*Δt), bins=0:dx:xmax, normalize=:pdf, size=(2000, 1500))

