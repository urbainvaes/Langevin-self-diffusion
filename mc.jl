#!/usr/bin/env julia
# import Pkg
# Pkg.add("QuadGK")
# Pkg.add("SpecialFunctions")
# Pkg.add("Elliptic")

import FFTW
import Plots
import Random
import Statistics
import SparseArrays
import LinearAlgebra
import QuadGK

sparse = SparseArrays
linalg = LinearAlgebra

import PyCall
sym = PyCall.pyimport("sympy")

# PARAMETERS {{{1

# Friction and inverse temperature
# γ, β = .01, 1;
γ, β = .005, 1;

# Potential and its derivative
V = q -> (1 .- cos.(q))/2;
dV = q -> sin.(q)/2;

# Normalization constant
Zν = QuadGK.quadgk(q -> exp(-β*V(q)), -π, π)[1]

# Numerical parameters
p = 200

# ωmax is the highest frequency of trigonometric functions in q and
# dmax is the highest degree of Hermite polynomials in p
ωmax, dmax = p, p*2;

# UNDERDAMPED LIMIT {{{1
import SpecialFunctions
import Elliptic

inf = 100
Zb = (2π)^(3/2) / β^(1/2) * exp(-β/2) * SpecialFunctions.besselj(0, β/2)
S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z)
integral = QuadGK.quadgk(z -> exp(-β*z) / S(z), 1, inf)[1]
diffusion_underdamped = (1/Zb)*(1/β)*8*π^2*integral

# FOURIER TOOLS {{{1
function flat_fourier(func)
    ngrid = 1 + 2*ωmax;
    qgrid = (2π/ngrid)*collect(0:ngrid-1);
    result = (1/ngrid) * FFTW.fft(func(qgrid))
    result = [result[ωmax+2:end]; result[1]; result[2:ωmax+1]]
    for i in 1:ngrid
        if abs(real(result[i])) < 1e-16
            result[i] = im*imag(result[i])
        end
        if abs(imag(result[i])) < 1e-16
            result[i] = real(result[i])
        end
    end
    return result
end

function prod_operator(s)
    ωmax = (length(s) - 1) ÷ 2;
    result = zeros(Complex, length(s), length(s))
    for ωi in -ωmax:ωmax
        i = ωmax + 1 + ωi
        for ωj in -ωmax:ωmax
            j = ωmax + 1 + ωj
            if abs(ωi - ωj) <= ωmax
                result[i, j] = s[ωmax + 1 + ωi - ωj]
            end
        end
    end
    return result
end

# In flat L² sin and cos, i.e. without the weight
function diff_operator()
    len = 2*ωmax + 1
    result = zeros(Complex, len, len)
    for i in 1:len
        ω = i - (1 + ωmax)
        result[i, i] = (ω*im)
    end
    return result
end

function to_sin_cos()
    len = 2*ωmax + 1
    result = zeros(Complex, len, len)
    result[1,ωmax+1] = sqrt(2π)
    # Constant, sin(x), cos(x), sin(2x), cos(2x)…
    for ω in 1:ωmax
        result[1+2*ω, ωmax+1-ω] += sqrt(π)
        result[1+2*ω-1, ωmax+1-ω] += -im*sqrt(π)
        result[1+2*ω, ωmax+1+ω] += 1*sqrt(π)
        result[1+2*ω-1, ωmax+1+ω] += im*sqrt(π)
    end
    return result
end

# Change of basis (normalization constants cancel out)
T = to_sin_cos();
T¯¹ = inv(T);

# Fourier series of dV
dVf = flat_fourier(dV);

# Differentiation operator
Q = real(T*(prod_operator(β/2*dVf) + diff_operator())*T¯¹);

# Identity matrix

# HERMITE TOOLS {{{1
P = zeros(dmax + 1, dmax + 1)
N = zeros(dmax + 1, dmax + 1)
for d in 1:dmax
    i = d + 1
    P[i-1, i] = sqrt(β*d)
    N[i, i] = d
end

# TENSORIZATION {{{1

# This determines the map for going from multidimensional to linear indices an
# back. This will need to be optimized at some point if we want to reduce the
# stiffness matrix bandwidth.
Nq, Np = 1 + 2*ωmax, 1 + dmax
multi_indices = zeros(Int, Nq*Np, 2);
lin_indices = zeros(Int, Nq, Np);

lin_index = 1;
for k in 1:Nq
    for l in 1:Np
        multi_indices[lin_index,:] = [k l];
        lin_indices[k, l] = lin_index;
        lin_index += 1;
    end
end

function toCOO(matCSC)
    size_vecs = length(matCSC.nzval);
    R, C, V = zeros(Int, size_vecs), zeros(Int, size_vecs), zeros(size_vecs)
    for (c, i) in enumerate(matCSC.colptr[1:end-1])
        centries = matCSC.colptr[c + 1] - matCSC.colptr[c]
        for z in 0:(centries-1)
            C[i+z] = c;
            R[i+z] = matCSC.rowval[i+z];
            V[i+z] = matCSC.nzval[i+z];
        end
    end
    return (R, C, V, matCSC.m, matCSC.n)
end

function tensorize(qmat, pmat)
    if (size(qmat)[1], size(pmat)[1]) != size(lin_indices)
        println("Invalid dimensions!")
    end
    qmat = sparse.sparse(qmat);
    pmat = sparse.sparse(pmat);
    (Rq, Cq, Vq, np, _) = toCOO(qmat);
    (Rp, Cp, Vp, nq, _) = toCOO(pmat);
    R = zeros(Int, length(Vp)*length(Vq));
    C = zeros(Int, length(Vp)*length(Vq));
    V = zeros(length(Vp)*length(Vq));
    counter = 1;
    for i in 1:length(Rq)
        for j in 1:length(Rp)
            R[counter] = lin_indices[Rq[i], Rp[j]];
            C[counter] = lin_indices[Cq[i], Cp[j]];
            V[counter] = Vq[i]*Vp[j];
            counter += 1
        end
    end
    return sparse.sparse(R, C, V, np*nq, np*nq);
end

function tensorize_vecs(qvec, pvec)
    result = zeros(Nq*Np)
    for i in 1:(Nq*Np)
        iq, ip = multi_indices[i, 1], multi_indices[i, 2]
        result[i] = qvec[iq]*pvec[ip]
    end
    return result
end

# Assemble the generator
I = sparse.sparse(1.0*linalg.I(2*ωmax + 1));
L = (1/β)*(tensorize(Q, P') - tensorize(Q', P)) + γ*tensorize(I, N);

# Right-hand side
one_q = real(T*flat_fourier(q -> exp.(-β*V(q)/2)/sqrt(Zν)));
rhs_p = zeros(Np); rhs_p[2] = 1/sqrt(β);
one_p = zeros(Np); one_p[1] = 1;
rhs = tensorize_vecs(one_q, rhs_p);
u = tensorize_vecs(one_q, one_p);

# Matrix
A = [[L u]; [u' 0]];
b = [rhs; 0];

# Effective diffusion
# D = (Array(L)\rhs)'rhs
solution = (A\b)[1:end-1];
D = solution'rhs

# MONTE CARLO METHOD {{{1

# Fix seed
# Random.seed!(0);

# Sample from the Gibbs distribution
function sample_gibbs(np)
    p = (1/sqrt(β)) * Statistics.randn(np);
    q, naccepts = zeros(Float64, np), 0;
    maxρ = exp(-β*V(pi));
    while naccepts < length(q)
        proposal = - pi + 2*pi*Statistics.rand();
        u = Statistics.rand();
        if v <= exp(-β*V(proposal))/exp(-β*V(0))
            naccepts += 1;
            q[naccepts] = u;
        end
    end
    return q, p
end

# Number of particles
np = 1000;

# Time step and final time
Δt = .01;
tf = 1000;

# Number of iterations
niter = ceil(Int, tf/Δt);
tf = niter*Δt;

# Position and momentum
q, p = sample_gibbs(np);

# Integrate the evolution
for i = 0:niter
    global p, q
    method = "geometric_langevin"
    if method == "geometric_langevin"
        p += - (Δt/2)*Vp(q);
        q += Δt*p;
        p += - (Δt/2)*Vp(q);
        α, gs = exp(-γ*Δt), Random.randn(np);
        p = α*p + sqrt((1 - α*α)/β)*gs
    elseif method == "euler_maruyama"
        Δw = sqrt(Δt)*Random.randn(np);
        q = q + Δt*p;
        p += - Δt*Vp(q) - Δt*γ*p + Δw*sqrt(2*γ/β)
    end
end

# Squared position (shorthand for this is q.*q)
q2 = broadcast(*, q, q);

# Estimation of the effective diffusion
D = Statistics.mean(q2) / (2*tf)

# Plots.plot(q)
Plots.histogram(q)
