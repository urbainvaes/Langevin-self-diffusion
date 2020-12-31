#!/usr/bin/env julia
import Random
import Statistics
import Polynomials
import QuadGK
import DelimitedFiles
import Printf
include("galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : .01;
control_type = length(ARGS) > 1 ? ARGS[2] : "galerkin";

# Batch number
batches = length(ARGS) > 2 ? ARGS[3] : "2/1";
ibatch = parse(Int, match(r"^[^/]*", batches).match);
nbatches = parse(Int, match(r"[^/]*$", brtches).match);

# Inverse temperature
β = 1;

# Create directory for data
datadir = "data/$control_type-γ=$γ/$ibatch"
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

# Track q2 at each iteration
mean_q² = zeros(niter);
DelimitedFiles.writedlm("$datadir/Δt=$Δt-mean_q2.txt", "");
DelimitedFiles.writedlm("$datadir/Δt=$Δt-q0.txt", q0)
DelimitedFiles.writedlm("$datadir/Δt=$Δt-p0.txt", p0)

# Control
if control_type == "galerkin"
    # !!! φ is solution of -Lφ = p (negative sign) !!!
    Dc, φ, ∂φ = get_controls(γ, true, false)
elseif control_type == "underdamped"
    Dc = (1/γ)*diff_underdamped(β);
    φ₀ = solution_underdamped();
    φ(q, p) = φ₀(q, p)/γ
    ∂φ(q, p) = ∂φ₀(q, p)/γ
end
println(@Printf.sprintf("Dc = %.3E", Dc))

# Covariance matrix of (Δw, ∫ e¯... dW)
rt_cov = root_cov(γ, Δt);

times = Δt*(1:niter) |> collect;
nsave = 1000;
nslice = niter ÷ nsave;

# Integrate the evolution
for i = 1:niter
    global p, q, ξ
    method = "geometric_langevin"

    gaussian_incs = rt_cov*Random.randn(2, np)
    Δw, gs = gaussian_incs[1, :], gaussian_incs[2, :]

    ξ += ∂φ.(q, p) .* (sqrt(2γ/β)*Δw)
    p += - (Δt/2)*dV.(q);
    q += Δt*p;
    p += - (Δt/2)*dV.(q);
    p = exp(-γ*Δt)*p + sqrt(2γ/β)*gs

    mean_q²[i] = Statistics.mean((q-q0).^2)
    if i % nslice == 0
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-p.txt", p)
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-q.txt", q)
        DelimitedFiles.writedlm("$datadir/Δt=$Δt-i=$i-ξ.txt", ξ)
        open("$datadir/Δt=$Δt-mean_q2.txt", "a") do io
            DelimitedFiles.writedlm(io, mean_q²[i-nslice+1:i])
        end
        print("Pogress: ", (1000*i) ÷ niter, "‰. ")
        control = ξ + φ.(q0, p0) - φ.(q, p);
        D1 = Statistics.mean((q - q0).^2) / (2*i*Δt)
        D2 = Dc + D1 - Statistics.mean(control.^2)/(2*i*Δt);
        σ1 = Statistics.std((q - q0).^2/(2*i*Δt))
        σ2 = Statistics.std(((q - q0).^2 - control.^2)/(2*i*Δt))
        println(@Printf.sprintf("D₁ = %.3E, D₂ = %.3E, σ₁ = %.3E, σ₂ = %.3E",
                                D1, D2, σ1, σ2))
        # f = Polynomials.fit(times[i÷10:i], mean_q²[i÷10:i], 1)
        # D2 = f.coeffs[2] / 2
        # D3 = Dc - Statistics.mean(ξ.^2)/(2*i*Δt) + D1
        # D4 = Dc - Statistics.mean((ξ + φ.(q0, p0) - φ.(q, p)).^2)/(2*i*Δt) + D1
        # println("D₁ = ", D1, " D₄ = ", D4)
        # println("error", q0 - q + ξ - φ.(q, p) + φ.(q0, p0))
        # println("q0 - q + ξ ", q0 - q + ξ, " φ - φ₀ ", φ.(q, p) - φ.(q0, p0))
        # println("q0 - q + ξ ", q0 - q + ξ, " φ - φ₀ ", φ.(q, p) - φ.(q0, p0))
    end
end
