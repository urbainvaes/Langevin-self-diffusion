module Spectral

import Arpack
import DelimitedFiles
import FFTW
import GaussQuadrature
import Cubature
import LinearAlgebra
import QuadGK
import SparseArrays
import Statistics
include("lib_sampling.jl")
sparse = SparseArrays;
la = LinearAlgebra;

export get_controls

struct Quadrature
    nodes::Array{BigFloat}
    weights::Array{BigFloat}
end

struct Series
    sigma::BigFloat
    coeffs::Array{BigFloat}
end

function get_controls(γ, δ, recalculate)
    datadir = "precom_data/galerkin/γ=$γ";
    run(`mkdir -p "$datadir"`);
    if !recalculate && isfile("$datadir/galerkin_q.txt")
        println("Using existing Galerkin solution!")
        qgrid = DelimitedFiles.readdlm("$datadir/galerkin_q.txt");
        pgrid = DelimitedFiles.readdlm("$datadir/galerkin_p.txt");
        solution_values = DelimitedFiles.readdlm("$datadir/galerkin_phi.txt");
        dp_solution_values = DelimitedFiles.readdlm("$datadir/galerkin_dp_phi.txt");
        dq, dp = qgrid[2] - qgrid[1], pgrid[2] - pgrid[1]
        Lp = pgrid[end]
    else
        _, solution_fun, dp_solution_fun = galerkin_solve(γ)
        # nq, np, Lp = 300, 500, 9;
        nq, np, Lp = 100, 100, 9; # Old parameters!
        dq, dp = 2π/nq, Lp/np;
        qgrid = -π .+ dq*collect(0:nq);
        pgrid = dp*collect(-np:np);

        solution_values = solution_fun(qgrid, pgrid);
        dp_solution_values = dp_solution_fun(qgrid, pgrid);

        DelimitedFiles.writedlm("$datadir/galerkin_q.txt", qgrid);
        DelimitedFiles.writedlm("$datadir/galerkin_p.txt", pgrid);
        DelimitedFiles.writedlm("$datadir/galerkin_phi.txt", solution_values)
        DelimitedFiles.writedlm("$datadir/galerkin_dp_phi.txt", dp_solution_values)
    end

    function bilinear_interpolant(values, q, p)
        q = q - 2π*floor(Int, (q+π)/2π);
        if abs(p) >= Lp
            println("p is out of interpolation grid!")
        end
        iq, ip = 1 + floor(Int, (q+π)/dq), 1 + floor(Int, (p+Lp)/dp);
        x, y = (q - qgrid[iq])/dq, (p - pgrid[ip])/dp
        a11 = values[iq, ip];
        a21 = values[iq+1, ip] - a11;
        a12 = values[iq, ip+1] - a11;
        a22 = values[iq+1, ip+1] + a11 - values[iq+1, ip] - values[iq, ip+1];
        return a11 + a21*x + a12*y + a22*x*y
    end

    φ(q, p) = bilinear_interpolant(solution_values, q, p)
    ∂φ(q, p) = bilinear_interpolant(dp_solution_values, q, p)

    # Improve on φ
    β = 1
    if !recalculate && isfile("$datadir/galerkin_D.txt")
        println("Using existing approximate diffusion coefficient!")
        D = DelimitedFiles.readdlm("$datadir/galerkin_D.txt")[1];
    else
        # nsamples = 10^7
        # q, p = Sampling.sample_gibbs(q -> (1 - cos(q))/2, β, nsamples)
        # D = γ*Statistics.mean(∂φ.(q, p).^2)
        # println("Effective diffusion (Interpolant): $D")
        # DelimitedFiles.writedlm("$datadir/galerkin_D.txt", D);

        Vδ(q₁, q₂) = - cos(q₁)/2 - cos(q₂)/2 - δ*cos(q₁)*cos(q₂);
        Zδ, _ = Cubature.hcubature(q -> exp(-β*Vδ(q[1], q[2])), [-π, -π], [π, π])
        Zp = sqrt(2π/β)

        nevals = 0
        function integrand(x)
            global nevals += 1
            if nevals % 100000 == 0
                println(nevals)
            end
            q₁, q₂, p₁ = x
            μ(q₁, q₂, p₁) = exp(-β*(Vδ(q₁, q₂) + p₁^2/2)) / (Zδ * Zp)
            return φ(q₁, p₁)*p₁ * μ(q₁, q₂, p₁)
        end
        D = Cubature.hcubature(integrand, [-π, -π, -Lp], [π, π, Lp], reltol=1e-5)
        println("Effective diffusion (Interpolant): $D")
        DelimitedFiles.writedlm("$datadir/galerkin_D.txt", D);
    end

    return (D, φ, ∂φ)
end

function galerkin_solve(γ)
    # PARAMETERS {{{1

    # Inverse temperature
    β, σ = 1, .1

    # Potential and its derivative
    V(q) = (1 - cos(q))/2;
    dV(q) = sin(q)/2;

    # Normalization constant
    Zν = QuadGK.quadgk(q -> exp(-β*V(q)), -π, π)[1];

    # Numerical parameters
    p = 300;

    # ωmax is the highest frequency of trigonometric functions in q and
    # dmax is the highest degree of Hermite polynomials in p
    ωmax, dmax = p, p*2;

    # The following was used to generate data :
    σ, p = 1, 400;
    ωmax, dmax = p ÷ 4, p*2;

    # Test overdamped
    # σ, ωmax, dmax = 1, 20, 40;

    # FOURIER TOOLS {{{1
    function flat_fourier(func)
        ngrid = 1 + 2*ωmax;
        qgrid = (2π/ngrid)*collect(0:ngrid-1);
        result = (1/ngrid) * FFTW.fft(func.(qgrid))
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
            result[1+2*ω, ωmax+1+ω] += sqrt(π)
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

    # HERMITE TOOLS {{{1
    Dp = zeros(dmax + 2, dmax + 2);
    for d in 1:dmax + 1
        i = d + 1
        Dp[i-1, i] = sqrt(d)/σ
    end
    Dp = Dp + (β*σ^2 - 1) * (Dp + Dp')/2;
    N = (Dp'*Dp)[1:end-1, 1:end-1];
    Dp = Dp[1:end-1, 1:end-1];

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
        R = zeros(Int, size_vecs)
        C = zeros(Int, size_vecs)
        local V = zeros(size_vecs)
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

    function tensorize_with_indices(qmat, pmat, indices)
        if (size(qmat)[1], size(pmat)[1]) != size(indices)
            println("Invalid dimensions!")
        end
        qmat = sparse.sparse(qmat);
        pmat = sparse.sparse(pmat);
        (Rq, Cq, Vq, nq, _) = toCOO(qmat);
        (Rp, Cp, Vp, np, _) = toCOO(pmat);
        R = zeros(Int, length(Vp)*length(Vq));
        C = zeros(Int, length(Vp)*length(Vq));
        local V = zeros(length(Vp)*length(Vq));
        counter = 1;
        for i in 1:length(Rq)
            for j in 1:length(Rp)
                R[counter] = indices[Rq[i], Rp[j]];
                C[counter] = indices[Cq[i], Cp[j]];
                V[counter] = Vq[i]*Vp[j];
                counter += 1
            end
        end
        return sparse.sparse(R, C, V, np*nq, np*nq);
    end

    function tensorize_vecs_with_indices(qvec, pvec, indices)
        Nq, Np = length(qvec), length(pvec)
        result = zeros(Nq*Np)
        for iq in 1:Nq
            for ip in 1:Np
                index = indices[iq, ip]
                result[index] = qvec[iq]*pvec[ip]
            end
        end
        return result
    end

    function hermite_eval(dmax, x)
        X = zeros(BigFloat, dmax + 1, length(x))
        X[1, :] .+= 1
        if dmax > 0
            X[2, :] += x
        end
        for d in 2:dmax
            X[d+1, :] = x .* X[d, :] - (d-1) * X[d-1, :]
        end
        normalizations = 1 ./ sqrt.(factorial.(big.(0:dmax)))
        return normalizations .* X
    end

    function gauss_hermite(N, σ)
        gq = GaussQuadrature
        nodes, weights = [output for output in gq.hermite(BigFloat, N)]
        return Quadrature(σ*sqrt(big(2))*nodes, weights/sqrt(big(π)))
    end

    function decompose(f, d, σ)
        σ = big(σ)
        quad = gauss_hermite(2*d + 1, σ)
        hermite_evals = hermite_eval(d, quad.nodes/σ)
        factors = (β*σ^2)^(1/4) * exp.(- (β - 1/σ^2)*quad.nodes.^2/4)
        coeffs = hermite_evals * (f.(quad.nodes) .* factors .* quad.weights)
        return Series(σ, coeffs)
    end

    # Assemble the generator
    I = sparse.sparse(1.0*la.I(2*ωmax + 1));
    tensorize(qmat, pmat) = tensorize_with_indices(qmat, pmat, lin_indices);
    minusL = (1/β)*(tensorize(Q', Dp) - tensorize(Q, Dp')) + γ*tensorize(I, N);
    diffp = tensorize(I, Dp);

    # Right-hand side
    one_q = real(T*flat_fourier(q -> exp(-β*V(q)/2)/sqrt(Zν)));
    one_p = decompose(p -> 1, dmax, σ).coeffs;
    rhs_p = decompose(p -> p, dmax, σ).coeffs;
    tensorize_vecs(qvec, pvec) = tensorize_vecs_with_indices(qvec, pvec, lin_indices)
    rhs = tensorize_vecs(one_q, rhs_p);
    u = tensorize_vecs(one_q, one_p);

    # Matrix
    A = [[minusL u]; [u' 0]];
    b = [rhs; 0];

    function fourier_eval(ωmax, qs)
        result = zeros(2*ωmax + 1)
        result[1] = 1/sqrt(2π)
        z, r = 1, exp(qs*im)
        for ω in 1:ωmax
            z *= r
            result[2ω] = imag(z)/sqrt(π)
            result[2ω + 1] = real(z)/sqrt(π)
        end
        return result * sqrt(Zν*exp(β*V(qs)))
    end

    function gen_hermite_eval(dmax, ps, σ)
        np = length(ps)
        hermite_evals = hermite_eval(dmax, ps/σ)
        factors = 1/(β*σ^2)^(1/4) * exp.((β - 1/σ^2)*ps.^2/4)
        return Float64.(factors .* hermite_evals')
    end

    function eval_series(series)
        function result(qs, ps)
            fevals = hcat(fourier_eval.(ωmax, qs)...)'
            hevals = gen_hermite_eval(dmax, ps, σ)
            values = zeros(length(qs), length(ps))
            step = max(length(series) ÷ 100, 1)
            for i in 1:length(series)
                if i ÷ step > (i - 1) ÷ step
                    print(".")
                end
                iq, ip = multi_indices[i, :]
                values += series[i]*fevals[:, iq]*hevals[:, ip]'
            end
            return values
        end
        return result
    end

    # datadir = "precom_data/galerkin/γ=$γ";
    # if isfile("$datadir/eigenvals.txt")
    #     println("Using existing eigenvalue decomposition!")
    #     eigenvals = DelimitedFiles.readdlm("$datadir/eigenvals.txt");
    #     eigenvecs = DelimitedFiles.readdlm("$datadir/eigenvecs.txt");
    # else
    #     eigenvals, eigenvecs = Arpack.eigs(minusL, which=:SM, nev=100)
    #     DelimitedFiles.writedlm("$datadir/eigenvals.txt", real.(eigenvals));
    #     DelimitedFiles.writedlm("$datadir/eigenvecs.txt", real.(eigenvecs));
    # end

    # For 2d
    # sin_mul = real(T*prod_operator(flat_fourier(q -> sin(q)))*T¯¹);
    # cos_mul = real(T*prod_operator(flat_fourier(q -> cos(q)))*T¯¹);
    # Ip = sparse.sparse(1.0*la.I(dmax + 1));
    # M1 = tensorize(sin_mul, Dp);
    # M2 = tensorize(cos_mul, Ip);

    # Projection of the rhs
    # inner_products = eigenvecs'eigenvecs
    # decomp_rhs = inner_products\(eigenvecs'rhs)
    # D_approx = decomp_rhs'inner_products*(decomp_rhs./eigenvals)

    # new_minusL = inner_products .* eigenvals
    # new_inner_products_rhs = eigenvecs'rhs
    # new_sol = new_minusL\new_inner_products_rhs
    # D_approx = new_inner_products_rhs'new_sol
    # new_M1 = eigenvecs'M1*eigenvecs
    # new_M2 = eigenvecs'M2*eigenvecs
    # new_one = eigenvecs'u
    # new_I = sparse.sparse(1.0*la.I(nev));

    # nev = size(eigenvecs)[2]
    # twod_multi_indices = zeros(Int, nev*nev, 2);
    # twod_lin_indices = zeros(Int, nev, nev);
    # lin_index = 1;
    # for k in 1:nev
    #     for l in 1:nev
    #         twod_multi_indices[lin_index,:] = [k l];
    #         twod_lin_indices[k, l] = lin_index;
    #         lin_index += 1;
    #     end
    # end
    # twod_tensorize(op1, op2) = tensorize_with_indices(op1, op2, twod_lin_indices)
    # twod_tensorize_vecs(vec1, vec2) = tensorize_vecs_with_indices(vec1, vec2, twod_lin_indices)

    # δ = .1
    # twod_minusL = (twod_tensorize(new_minusL, new_I) + twod_tensorize(new_I, new_minusL)
    #                + δ*twod_tensorize(new_M1, new_M2) + δ*twod_tensorize(new_M2, new_M1))
    # twod_rhs = twod_tensorize_vecs(new_inner_products_rhs, new_one)
    # sol = twod_minusL \ twod_rhs

    # function new_tensorize(op1, op2)
    #     op1 = sparse.sparse(op1);
    #     op2 = sparse.sparse(op2);
    #     (R1, C1, V1, n1, _) = toCOO(op1);
    #     (R2, C2, V2, n2, _) = toCOO(op2);
    #     R = zeros(Int, length(V2)*length(V1));
    #     C = zeros(Int, length(V2)*length(V1));
    #     local V = zeros(length(V2)*length(V1));
    #     counter = 1;
    #     for i in 1:length(R1)
    #         for j in 1:length(R2)
    #             R[counter] = new_lin_indices[R1[i], R2[j]];
    #             C[counter] = new_lin_indices[C1[i], C2[j]];
    #             V[counter] = V1[i]*V2[j];
    #             counter += 1
    #         end
    #     end
    #     return sparse.sparse(R, C, V, n1*n2, n1*n2);
    # end
    # function new_tensorize_vecs(vec1, vec2)
    #     result = zeros(nev*nev)
    #     for i in 1:(nev*nev)
    #         i1, i2 = new_multi_indices[i, :]
    #         result[i] = vec1[iq]*vec2[ip]
    #     end
    #     return result
    # end
    # twod = rhs

    # Effective diffusion
    solution = (A\b)[1:end-1];
    dp_solution = diffp*solution;
    D = solution'rhs
    println("Effective diffusion (Galerkin): $D")

    # Turn them into functions
    solution_fun = eval_series(solution);
    dp_solution_fun = eval_series(dp_solution);
    return (D, solution_fun, dp_solution_fun)
end

function series_eval(series, ps)
    d = length(series.coeffs) - 1
    hermite_evals = hermite_eval(d, ps/series.sigma)
    factors = 1/(β*series.sigma^2)^(1/4) * exp.((β - 1/series.sigma^2)*ps.^2/4)
    return (hermite_evals .* factors')' * series.coeffs
end

end
