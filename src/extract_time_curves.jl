#!/usr/bin/env julia

import Statistics
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("lib_galerkin.jl")
include("lib_sampling.jl")
include("lib_underdamped.jl")

# Parse arguments
control_type = "underdamped"
if control_type == "underdamped"
    Control = Underdamped
elseif control_type == "galerkin"
    Control = Spectral
end

# Parameters
β, γ, δ = 1, .001, 0

# Get directory of data
clusterdir = "cluster/"
if δ == 0
    localdir = "data/$control_type-γ=$γ/"
else
    localdir = "data2d/$control_type-γ=$γ-δ=$δ/"
end
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
index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1]);
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
D11 = zeros(nsteps)
D22 = zeros(nsteps)
σ11 = zeros(nsteps)
σ22 = zeros(nsteps)
D11_control = zeros(nsteps)
D22_control = zeros(nsteps)
σ11_control = zeros(nsteps)
σ22_control = zeros(nsteps)

# Control
recalculate = true
Dc, ψ, ∂ψ = Control.get_controls(γ, recalculate)
println(@Printf.sprintf("Dc = %.3E", Dc))

for i in 1:nsteps
    println("$i/$nsteps")
    ts[i] = Δt*index(qfiles[i])
    qi = DelimitedFiles.readdlm(string(datadir, qfiles[i]));
    pi = DelimitedFiles.readdlm(string(datadir, pfiles[i]));
    ξi = DelimitedFiles.readdlm(string(datadir, ξfiles[i]));

    if δ == 0
        control = ξi + ψ.(q0, p0) - ψ.(qi, pi);
        to_average_1 = (qi - q0).^2 / (2*ts[i])
        to_average_2 = Dc .+ (qi - q0).^2 / (2*ts[i]) .- control.^2 / (2*ts[i])

        D1 = Statistics.mean(to_average_1);
        D2 = Statistics.mean(to_average_2);
        σ1 = Statistics.std(to_average_1);
        σ2 = Statistics.std(to_average_2);

        D11[i] = D1;
        D22[i] = D1;
        σ11[i] = σ1;
        σ22[i] = σ1;
        D11_control[i] = D2;
        D22_control[i] = D2;
        σ11_control[i] = σ2;
        σ22_control[i] = σ2;
        continue
    end

    # Without control
    term11 = (qi[:, 1] - q0[:, 1]).^2 / (2*ts[i])
    term12 = (qi[:, 1] - q0[:, 1]).*(qi[:, 2] - q0[:, 2]) / (2*ts[i])
    term22 = (qi[:, 2] - q0[:, 2]).^2 / (2*ts[i])
    D11[i] = Statistics.mean(term11)
    D12[i] = Statistics.mean(term12)
    D22[i] = Statistics.mean(term22)
    σ11[i] = Statistics.std(term11)
    σ22[i] = Statistics.std(term22)

    # With control
    control1 = ξi[:, 1] + ψ.(q0[:, 1], p0[:, 1]) - ψ.(qi[:, 1], pi[:, 1]);
    control2 = ξi[:, 2] + ψ.(q0[:, 2], p0[:, 2]) - ψ.(qi[:, 2], pi[:, 2]);
    term11 = Dc .- control1.^2 / (2*ts[i]) + term11
    term22 = Dc .- control2.^2 / (2*ts[i]) + term22
    D11_control[i] = Statistics.mean(term11)
    D22_control[i] = Statistics.mean(term22)
    σ11_control[i] = Statistics.std(term11)
    σ22_control[i] = Statistics.std(term22)
end


data_array = [ts D11 D22 σ11 σ22 D11_control D22_control σ11_control σ22_control]
run(`mkdir -p "data/time/"`)
open("data/time/$control_type-gamma=$γ-delta=$δ.txt"; write=true) do f
         write(f, "ts\tD11\tD22\tσ11\tσ22\tD11_control\tD22_control\tσ11_control\tσ22_control\n")
         DelimitedFiles.writedlm(f, data_array)
end
