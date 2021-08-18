#!/usr/bin/env julia

import Random
import Statistics
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("lib_sampling.jl")
include("lib_underdamped.jl")

# PARAMETERS {{{1

# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1e-5;
ν = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 2.;
control_type = "gle"

# Inverse temperature
β = 1  # β always 1!

# Parse the files
clusterdir = "cluster/"
localdir = "data_gle/gle-γ=$γ/"
is_cluster = occursin("clustern", gethostname())
datadir = (is_cluster ? "" : clusterdir) * localdir
if !isdir(datadir); exit(-1); end
readdlm = DelimitedFiles.readdlm
writedlm = DelimitedFiles.writedlm

# Control
Dc, dz_ψ = Underdamped.get_controls_gle(γ, ν, false);
_, ψ, _ = Underdamped.get_controls(γ, false);

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
    q0 = readdlm("$batchdir/$q0file");
    p0 = readdlm("$batchdir/$p0file");

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
end
data = map(datafiles, batchdirs);

# Time step
Δt = parse(Float64, match(r"Δt=([^-]+)", data[1]["qfiles"][1]).captures[1]);

nsteps = maximum(length(d["indices"]) for d in data)
# nsteps = min(nsteps, 220) # FIXME: change later!
ts = zeros(nsteps);
D = zeros(nsteps);
σ = zeros(nsteps);
D_control = zeros(nsteps);
σ_control = zeros(nsteps);
n_particles = zeros(nsteps);

# Calculate diffusion coefficients
for i in 1:10:nsteps

    index = maximum(map(d -> i <= length(d["qfiles"]) ? d["indices"][i] : 0, data))
    ts[i] = index*Δt;

    # Extract initial condition for still existing particles
    q0 = vcat(map(d -> i <= length(d["qfiles"]) ? d["q0"] : [], data)...);
    p0 = vcat(map(d -> i <= length(d["qfiles"]) ? d["p0"] : [], data)...);

    # Extract data for current iteration
    q = vcat(map(d -> i <= length(d["qfiles"]) ? readdlm(d["qfiles"][i]) : [], data)...);
    p = vcat(map(d -> i <= length(d["qfiles"]) ? readdlm(d["pfiles"][i]) : [], data)...);
    ξ = vcat(map(d -> i <= length(d["qfiles"]) ? readdlm(d["ξfiles"][i]) : [], data)...);

    print("Iteration: ", index, ". ");
    control = ξ + ψ.(q0, p0) - ψ.(q, p);
    to_average_1 = (q - q0).^2 / (2*ts[i])
    to_average_2 = Dc/γ .+ (q - q0).^2 / (2*ts[i]) .- control.^2 / (2*ts[i])

    D[i] = Statistics.mean(to_average_1);
    σ[i] = Statistics.std(to_average_1);
    D_control[i] = Statistics.mean(to_average_2);
    σ_control[i] = Statistics.std(to_average_2);
    n_particles[i] = length(q)
    println(@Printf.sprintf("D₁ = %.3E, D₂ = %.3E, σ₁ = %.3E, σ₂ = %.3E, np = %d",
                            D[i], D_control[i], σ[i], σ_control[i], n_particles[i]));
end

data_array = [ts D σ D_control σ_control n_particles]
run(`mkdir -p "data_gle/time/"`)
open("data_gle/time/gle-gamma=$γ.txt"; write=true) do f
         write(f, "ts\tD\tσ\tD_control\tσ_control\tn_particles\n")
         DelimitedFiles.writedlm(f, data_array)
end
