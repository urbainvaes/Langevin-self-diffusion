#!/usr/bin/env julia

import Statistics
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("lib_sampling.jl")
include("lib_underdamped.jl")

# Parameters
β, γ, ν = 1, .001, 2

# Get directory of data
clusterdir = "cluster/"
localdir = "data_gle/gle-γ=$γ/"
datadir = clusterdir * localdir
if !isdir(datadir)
    exit(-1)
end

# Get the index of a filename
index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1]);
listfiles = readdir(datadir);
if !("Δt=0.01-q0.txt" in listfiles)
    return -1;
end
q0file = filter(s -> occursin(r"q0.txt", s), listfiles)[1];
p0file = filter(s -> occursin(r"p0.txt", s), listfiles)[1];
qfiles = filter(s -> occursin(r"Δt=0.01-i=.*q.txt", s), listfiles);
pfiles = filter(s -> occursin(r"Δt=0.01-i=.*p.txt", s), listfiles);
ξfiles = filter(s -> occursin(r"Δt=0.01-i=.*ξ.txt", s), listfiles);
qfiles = sort(qfiles, by=index);
pfiles = sort(pfiles, by=index);
ξfiles = sort(ξfiles, by=index);

# Get data of last iteration
Δt = parse(Float64, match(r"Δt=([^-]+)", qfiles[1]).captures[1]);
q0 = DelimitedFiles.readdlm(string(datadir, q0file));
p0 = DelimitedFiles.readdlm(string(datadir, p0file));

nsteps = length(qfiles)
ts = zeros(nsteps)
D = zeros(nsteps)
σ = zeros(nsteps)
D_control = zeros(nsteps)
σ_control = zeros(nsteps)

# Control
recalculate = true
Dc, _ = Underdamped.get_controls_gle(γ, ν, recalculate)
_, ψ, _ = Underdamped.get_controls(γ, recalculate)

for i in 1:nsteps
    println("$i/$nsteps")
    ts[i] = Δt*index(qfiles[i])
    qi = DelimitedFiles.readdlm(string(datadir, qfiles[i]));
    pi = DelimitedFiles.readdlm(string(datadir, pfiles[i]));
    ξi = DelimitedFiles.readdlm(string(datadir, ξfiles[i]));

    control = ξi + ψ.(q0, p0) - ψ.(qi, pi);
    to_average_1 = (qi - q0).^2 / (2*ts[i])
    to_average_2 = Dc .+ (qi - q0).^2 / (2*ts[i]) .- control.^2 / (2*ts[i])

    D[i] = Statistics.mean(to_average_1);
    σ[i] = Statistics.std(to_average_1);
    D_control[i] = Statistics.mean(to_average_2);
    σ_control[i] = Statistics.std(to_average_2);
end

data_array = [ts D σ D_control σ_control]
run(`mkdir -p "data-gle/time/"`)
open("data-gle/time/gle-gamma=$γ.txt"; write=true) do f
         write(f, "ts\tD\tσ\tD_control\tσ_control")
         DelimitedFiles.writedlm(f, data_array)
end
