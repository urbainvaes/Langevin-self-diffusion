#!/usr/bin/env julia

import Random
import Statistics
import Polynomials
import QuadGK
import LinearAlgebra
import DelimitedFiles
import Printf
linalg = LinearAlgebra;
include("lib.jl");

γs = [.001, .00215, .00464, .01, .0215, .0464, .1, .215, .464, 1.0];
δs = [.04, .08, .16, .32, .64];
β = 1;

# Underdamped limit
Du = diff_underdamped(β);
φ₀ = solution_underdamped();

function get_diffusion(γ, δ)
    datadir = "data2d/γ=$γ-δ=$δ/"

    if !isdir(datadir)
        return -1
    end

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

    # Get data of last iteration
    Δt = parse(Float64, match(r"Δt=([^-]+)", qfiles[1]).captures[1]);
    q0 = DelimitedFiles.readdlm(string(datadir, q0file));
    p0 = DelimitedFiles.readdlm(string(datadir, p0file));
    qend = DelimitedFiles.readdlm(string(datadir, qfiles[end]));
    pend = DelimitedFiles.readdlm(string(datadir, pfiles[end]));
    ξend = DelimitedFiles.readdlm(string(datadir, ξfiles[end]));

    # Iteration indices
    indices = map(index, qfiles);
    tend = indices[end]*Δt

    # Without control
    term11 = (qend[:, 1] - q0[:, 1]).^2 / (2*tend)
    term12 = (qend[:, 1] - q0[:, 1]).*(qend[:, 2] - q0[:, 2]) / (2*tend)
    term22 = (qend[:, 2] - q0[:, 2]).^2 / (2*tend)
    D11 = Statistics.mean(term11)
    D12 = Statistics.mean(term12)
    D22 = Statistics.mean(term22)
    σ11 = Statistics.std(term11)
    σ22 = Statistics.std(term22)
    wout_control = Dict("D11" => D11, "D22" => D22, "σ11" => σ11, "σ22" => σ22)

    # With control
    control1 = ξend[:, 1] + φ₀.(q0[:, 1], p0[:, 1])/γ - φ₀.(qend[:, 1], pend[:, 1])/γ;
    control2 = ξend[:, 2] + φ₀.(q0[:, 2], p0[:, 2])/γ - φ₀.(qend[:, 2], pend[:, 2])/γ;
    term11 = (1/γ)*Du .- control1.^2 / (2*tend) + term11
    term22 = (1/γ)*Du .- control2.^2 / (2*tend) + term22
    D11 = Statistics.mean(term11)
    D22 = Statistics.mean(term22)
    σ11 = Statistics.std(term11)
    σ22 = Statistics.std(term22)
    with_control = Dict("D11" => D11, "D22" => D22, "σ11" => σ11, "σ22" => σ22)

    return (wout_control, with_control)
end

D11_wo = zeros(length(γs), length(δs));
σ11_wo, D11_wi, σ11_wi = copy(D11_wo), copy(D11_wo), copy(D11_wo);

for iγ in 1:length(γs)
    for iδ in 1:length(δs)
        γ, δ = γs[iγ], δs[iδ];
        data = get_diffusion(γ, δ);
        if data == -1
            println("No data for parameters (γ=$γ, δ=$δ)");
            continue
        end
        data1, data2 = data;
        println("Parameters: γ=$γ, δ=$δ, with and without control variate:");
        println(@Printf.sprintf("-> D₁₁ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                                data1["D11"], data1["D22"], data1["σ11"], data1["σ22"]));
        println(@Printf.sprintf("-> D₁₁ = %.3E, D₂₂ = %.3E, σ₁₁ = %.3E, σ₂₂ = %.3E",
                                data2["D11"], data2["D22"], data2["σ11"], data2["σ22"]));
        D11_wo[iγ, iδ] = data1["D11"];
        D11_wi[iγ, iδ] = data2["σ11"];
        σ11_wo[iγ, iδ] = data1["D11"];
        σ11_wi[iγ, iδ] = data2["σ11"];
    end
end

DelimitedFiles.writedlm("data_D11_wo.txt", D11_wo);
DelimitedFiles.writedlm("data_σ11_wo.txt", σ11_wo);
DelimitedFiles.writedlm("data_D11_wi.txt", σ11_wi);
DelimitedFiles.writedlm("data_σ11_wi.txt", σ11_wi);
