#!/usr/bin/env julia

import Random
import Statistics
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("lib_galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Friction and inverse temperature
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1.0;
δ = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : .04;
control_type = length(ARGS) > 2 ? ARGS[3] : "galerkin";

# Inverse temperature
β = 1  # β always 1!

# Control
if control_type == "galerkin"
    Dc, ψ, _ = get_controls(γ, true, false);
elseif control_type == "underdamped"
    Dc = (1/γ)*diff_underdamped(β);
    φ₀ = solution_underdamped();
    ψ(q, p) = φ₀(q, p)/γ;
end;

# Parse the files
datadir = "data2d/$control_type-γ=$γ-δ=$δ";
readf, writef = DelimitedFiles.readdlm, DelimitedFiles.writedlm;

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
if !("Δt=0.01-q0.txt" in listfiles)
    batchdirs = map(s -> "$datadir/$s", batches)
    writefiles = true
else
    batchdirs = [datadir];
    writefiles = false;
end;
data = map(datafiles, batchdirs)

# Time step
Δt = parse(Float64, match(r"Δt=([^-]+)", data[1]["qfiles"][1]).captures[1]);

# Extract initial condition
q0 = vcat((d["q0"] for d in data)...);
p0 = vcat((d["p0"] for d in data)...);

if writefiles
    writef("$datadir/Δt=$Δt-q0.txt", q0);
    writef("$datadir/Δt=$Δt-p0.txt", p0);
end

# Calculate diffusion coefficients
for i in 1:minimum(length(d["indices"]) for d in data)

    index = data[1]["indices"][i];
    time = index*Δt;

    q = vcat(map(d -> readf(d["qfiles"][i]), data)...);
    p = vcat(map(d -> readf(d["pfiles"][i]), data)...);
    ξ = vcat(map(d -> readf(d["ξfiles"][i]), data)...);

    if writefiles
        writef("$datadir/Δt=$Δt-i=$index-q.txt", q);
        writef("$datadir/Δt=$Δt-i=$index-p.txt", p);
        writef("$datadir/Δt=$Δt-i=$index-ξ.txt", ξ);
    end

    print("Iteration: ", index, ". ");
    term11 = (q[:, 1] - q0[:, 1]).^2 / (2*time)
    term12 = (q[:, 1] - q0[:, 1]).*(q[:, 2] - q0[:, 2]) / (2*time)
    term22 = (q[:, 2] - q0[:, 2]).^2 / (2*time)
    D11 = Statistics.mean(term11)
    D12 = Statistics.mean(term12)
    D22 = Statistics.mean(term22)
    σ11 = Statistics.std(term11)
    σ22 = Statistics.std(term22)
    println(@Printf.sprintf("D₁₁ = %.3E, D₁₂ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                            D11, D12, D22, σ11, σ22))
    control1 = ξ[:, 1] + ψ.(q0[:, 1], p0[:, 1]) - ψ.(q[:, 1], p[:, 1]);
    control2 = ξ[:, 2] + ψ.(q0[:, 2], p0[:, 2]) - ψ.(q[:, 2], p[:, 2]);
    term11 = Dc .- control1.^2 / (2*time) + term11
    term22 = Dc .- control2.^2 / (2*time) + term22
    D11 = Statistics.mean(term11)
    D12 = Statistics.mean(term12)
    D22 = Statistics.mean(term22)
    σ11 = Statistics.std(term11)
    σ22 = Statistics.std(term22)
    println(@Printf.sprintf("D₁₁ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                            D11, D22, σ11, σ22))
end
