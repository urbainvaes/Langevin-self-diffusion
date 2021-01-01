#!/usr/bin/env julia

import Random
import Statistics
import Polynomials
import QuadGK
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1.;
control_type = length(ARGS) > 1 ? ARGS[2] : "galerkin"

# Inverse temperature
β = 1  # β always 1!

# Control
if control_type == "galerkin"
    Dc, φ, _ = get_controls(γ, true, false)
elseif control_type == "underdamped"
    Dc = (1/γ)*diff_underdamped(β);
    φ₀ = solution_underdamped();
    φ(q, p) = φ₀(q, p)/γ
end

# Parse the files
datadir = "data/$control_type-γ=$γ"
readf, writef = DelimitedFiles.readdlm, DelimitedFiles.writedlm

function datafiles(batchdir)

    # List data files in batch directory
    datafiles = readdir(batchdir);
    index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1]);

    # Files
    q0file = filter(s -> occursin(r"q0.txt", s), datafiles)[1];
    p0file = filter(s -> occursin(r"p0.txt", s), datafiles)[1];
    qfiles = filter(s -> occursin(r"Δt=0.01-i=.*q.txt", s), datafiles);
    pfiles = filter(s -> occursin(r"Δt=0.01-i=.*p.txt", s), datafiles);
    ξfiles = filter(s -> occursin(r"Δt=0.01-i=.*ξ.txt", s), datafiles);
    qfiles = map(s -> "$batchdir/$s", sort(qfiles, by=index));
    pfiles = map(s -> "$batchdir/$s", sort(pfiles, by=index));
    ξfiles = map(s -> "$batchdir/$s", sort(ξfiles, by=index));

    # Iteration indices
    indices = map(index, qfiles);

    # Initial condition
    q0 = readf("$batchdir/$q0file");
    p0 = readf("$batchdir/$p0file");

    return Dict("q0" => q0, "p0" => p0, "qfiles" => qfiles,
                "pfiles" => pfiles, "ξfiles" => ξfiles, "indices" => indices)
end

# Extract data from all batches
listfiles = readdir(datadir);
batches = filter(s -> occursin(r"^[0-9]*$", s), listfiles);
batchdirs = length(batches) > 1 ? map(s -> "$datadir/$s", batches) : [datadir];
data = map(datafiles, batchdirs)

# Extract initial condition
q0 = vcat((d["q0"] for d in data)...)
p0 = vcat((d["p0"] for d in data)...)

# Time step
Δt = parse(Float64, match(r"Δt=([^-]+)", data[1]["qfiles"][1]).captures[1]);

# Calculate diffusion coefficients
for i in 1:minimum(length(d["indices"]) for d in data)

    index = data[1]["indices"][i]
    time = index*Δt

    q = vcat(map(d -> readf(d["qfiles"][i]), data)...);
    p = vcat(map(d -> readf(d["pfiles"][i]), data)...);
    ξ = vcat(map(d -> readf(d["ξfiles"][i]), data)...);

    writef("$datadir/Δt=$Δt-i=$i-q.txt", q)
    writef("$datadir/Δt=$Δt-i=$i-p.txt", p)
    writef("$datadir/Δt=$Δt-i=$i-ξ.txt", ξ)

    print("Iteration: ", index, ". ");
    control = ξ + φ.(q0, p0) - φ.(q, p);
    D1 = Statistics.mean((q - q0).^2) / (2*time);
    D2 = Dc + D1 - Statistics.mean(control.^2)/(2*time);
    σ1 = Statistics.std((q - q0).^2/(2*time))
    σ2 = Statistics.std(((q - q0).^2 - control.^2)/(2*time))
    println(@Printf.sprintf("D₁ = %.3E, D₂ = %.3E, σ₁ = %.3E, σ₂ = %.3E",
                            D1, D2, σ1, σ2))
end

# qend = DelimitedFiles.readdlm(string(datadir, qfiles[end]));
# pend = DelimitedFiles.readdlm(string(datadir, pfiles[end]));
# ξend = DelimitedFiles.readdlm(string(datadir, ξfiles[end]));
# control_end = ξend + φ₀.(q0, p0)/γ - φ₀.(qend, pend)/γ;
# tend = indices[end]*Δt

# import Plots
# Plots.histogram((qend - q0).^2, bins=20)

# f = Polynomials.fit(times[i÷10:i], mean_q²[i÷10:i], 1)
# D2 = f.coeffs[2] / 2
# D3 = (1/γ)*Du - Statistics.mean(ξ.^2)/(2*i*Δt) + D1

# f = Polynomials.fit(times[niter÷10:end], mean_q²[niter÷10:end], 1)
# D = f.coeffs[2] / 2

# Plots.plot(times, mean_q²)
# Plots.plot!(f, times)
# Plots.plot(times, mean_q²./(2*times))

# Squared position (shorthand for this is q.*q)
# q2 = broadcast(*, q - q0, q - q0);

# Plots.histogram(q2, bins=20)

# Plots.histogram(q0, bins=-π:(π/10):π)
# Plots.histogram(p0, bins=20)
