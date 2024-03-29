#!/usr/bin/env julia
# import Plots
import Random
import Statistics
import DelimitedFiles
import Printf
using ProfileCanvas
include("lib_gridap.jl")
include("lib_galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
const γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : .1;
control_type = length(ARGS) > 1 ? ARGS[2] : "underdamped";

if control_type == "underdamped"
    Control = Underdamped
elseif control_type == "galerkin"
    Control = Spectral
elseif control_type == "gridap"
    Control = FemGridap
end

# Batch number
batches = length(ARGS) > 2 ? ARGS[3] : "1/1";
ibatch = parse(Int, match(r"^[^/]*", batches).match);
nbatches = parse(Int, match(r"[^/]*$", batches).match);

# Inverse temperature
β = 1;

# Create directory for data
appenddir = (nbatches > 1 ? "/$ibatch" : "")
datadir = "data/$control_type-γ=$γ$appenddir"
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
Δt = .01 / max(1, γ);

# Δt = .01;
tf = ceil(Int, 100*max(1/γ, γ));

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q0, p0 = Sampling.sample_gibbs(V, β, np);
q, p, ξ = copy(q0), copy(p0), zeros(np);

# Control
recalculate = false
Dc, ψ, ∂ψ = Control.get_controls(γ, 0, recalculate)
println(@Printf.sprintf("Dc = %.3E", Dc))

# Covariance matrix of (Δw, ∫ e¯... dW)
rt_cov = Sampling.root_cov(γ, Δt);

# Number of saves
nsave = 1000;
nslice = niter ÷ nsave;

# Write initial condition to file
DelimitedFiles.writedlm("$datadir/Δt=$Δt-q0.txt", q0)
DelimitedFiles.writedlm("$datadir/Δt=$Δt-p0.txt", p0)

function main()

    # Integrate the evolution
    for i = 1:niter
        if i % 1000 == 0
            print(".")
        end
        print(".")
        global p, q, ξ

        # Generate Gaussian increments
        gaussian_increments = rt_cov*Random.randn(2, np)
        Δw, gs = gaussian_increments[1, :], gaussian_increments[2, :]

        ξ .+= ∂ψ.(q, p) .* (sqrt(2γ/β)*Δw)
        p .+= - (Δt/2)*dV.(q);
        q .+= Δt*p;
        p .+= - (Δt/2)*dV.(q);
        p .= exp(-γ*Δt)*p + sqrt(2γ/β)*gs

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
end
# control = ξ + ψ.(q0, p0) - ψ.(q, p);

# dx, xmax = .1, 5
# Plots.histogram(((q - q0).^2 - control.^2)/(2*niter*Δt), bins=0:dx:xmax, normalize=:pdf)
# Plots.histogram((q - q0).^2/(2*niter*Δt), bins=0:dx:xmax, normalize=:pdf, size=(2000, 1500))
