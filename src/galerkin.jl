import FFTW
import SparseArrays
import LinearAlgebra
import QuadGK

sparse = SparseArrays;
linalg = LinearAlgebra;

# PARAMETERS {{{1

# Friction and inverse temperature
γ, β = .001, 1;
# γ, β = .001, 1;

# Potential and its derivative
V = q -> (1 .- cos.(q))/2;
dV = q -> sin.(q)/2;

# Normalization constant
Zν = QuadGK.quadgk(q -> exp(-β*V(q)), -π, π)[1];

# Numerical parameters
p = 300;

# ωmax is the highest frequency of trigonometric functions in q and
# dmax is the highest degree of Hermite polynomials in p
ωmax, dmax = p÷4, p*2;

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
P = zeros(dmax + 1, dmax + 1);
N = zeros(dmax + 1, dmax + 1);
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
