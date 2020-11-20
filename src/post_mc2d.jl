#!/usr/bin/env julia

import Random
import Statistics
import Polynomials
import QuadGK
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("lib.jl")

# PARAMETERS {{{1

# Friction and inverse temperature
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 10;
δ = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : .2;
β =  1;

# Parse the files
datadir = "data2d/γ=$γ-δ=$δ/"
listfiles = readdir(datadir);
index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1]);
q0file = filter(s -> occursin(r"q0.txt", s), listfiles)[1];
p0file = filter(s -> occursin(r"p0.txt", s), listfiles)[1];
qfiles = filter(s -> occursin(r"Δt=0.01-i=.*q.txt", s), listfiles);
pfiles = filter(s -> occursin(r"Δt=0.01-i=.*p.txt", s), listfiles);
ξfiles = filter(s -> occursin(r"Δt=0.01-i=.*ξ.txt", s), listfiles);
qfiles = sort(qfiles, by=index);
pfiles = sort(pfiles, by=index);
ξfiles = sort(ξfiles, by=index);

# Iteration indices
indices = map(index, qfiles);

# Time step
Δt = parse(Float64, match(r"Δt=([^-]+)", qfiles[1]).captures[1]);

# Initial condition
q0 = DelimitedFiles.readdlm(string(datadir, q0file));
p0 = DelimitedFiles.readdlm(string(datadir, p0file));

# Underdamped limit
Du = diff_underdamped(β);
φ₀ = solution_underdamped();

# Calculate diffusion coefficients
for i in 1:length(indices)
    time = indices[i] * Δt
    q = DelimitedFiles.readdlm(string(datadir, qfiles[i]));
    p = DelimitedFiles.readdlm(string(datadir, pfiles[i]));
    ξ = DelimitedFiles.readdlm(string(datadir, ξfiles[i]));

    println("Iteration: ", indices[i], ". ");
    term11 = (q[:, 1] - q0[:, 1]).^2 / (2*i*Δt)
    term12 = (q[:, 1] - q0[:, 1]).*(q[:, 2] - q0[:, 2]) / (2*time)
    term22 = (q[:, 2] - q0[:, 2]).^2 / (2*time)
    D11 = Statistics.mean(term11)
    D12 = Statistics.mean(term12)
    D22 = Statistics.mean(term22)
    σ11 = Statistics.std(term11)
    σ22 = Statistics.std(term22)
    println(@Printf.sprintf("D₁₁ = %.3E, D₁₂ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                            D11, D12, D22, σ11, σ22))
    control1 = ξ[:, 1] + φ₀.(q0[:, 1], p0[:, 1])/γ - φ₀.(q[:, 1], p[:, 1])/γ;
    control2 = ξ[:, 2] + φ₀.(q0[:, 2], p0[:, 2])/γ - φ₀.(q[:, 2], p[:, 2])/γ;
    term11 = (1/γ)*Du .- control1.^2 / (2*time) + term11
    term22 = (1/γ)*Du .- control2.^2 / (2*time) + term22
    D11 = Statistics.mean(term11)
    D12 = Statistics.mean(term12)
    D22 = Statistics.mean(term22)
    σ11 = Statistics.std(term11)
    σ22 = Statistics.std(term22)
    println(@Printf.sprintf("D₁₁ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                            D11, D22, σ11, σ22))
end

qend = DelimitedFiles.readdlm(string(datadir, qfiles[end]));
pend = DelimitedFiles.readdlm(string(datadir, pfiles[end]));
ξend = DelimitedFiles.readdlm(string(datadir, ξfiles[end]));
control_end = ξend + φ₀.(q0, p0)/γ - φ₀.(qend, pend)/γ;
tend = indices[end]*Δt
