module Underdamped

import DelimitedFiles
import DifferentialEquations
import Elliptic
import QuadGK
import SpecialFunctions
import Statistics
include("lib_sampling.jl")

export get_controls

V(q) = (1 - cos(q))/2;

function get_controls(γ, _required_by_api)

    # Inverse temperature
    β = 1

    # This is only for the case of the cosine potential!
    # V(q) = (1 - cos(q))/2
    function diff_underdamped(β)
        E₀, Estep, Einf = 1, .001, 20
        Es = E₀:Estep:Einf
        inf = 100;
        Zb = (2π)^(3/2) / β^(1/2) * exp(-β/2) * SpecialFunctions.besseli(0, β/2);
        S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z);
        integral = QuadGK.quadgk(z -> exp(-β*z) / S(z), 1, inf)[1];
        Du = (1/Zb)*(1/β)*8*π^2*integral;
    end

    E₀ = 1
    V_1d(q) = (1 - cos(q))/2;
    S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z);
    function ∂φ₀(q, p)
        E = V_1d(q) + p*p/2
        E > E₀ ? sign(p)*p*2π/S(E) : 0
    end


    # This takes vectors!
    function solution_underdamped()
        E₀ = 1
        S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z);
        diff(u, p, t) = t < E₀ ? 0 : 2π/S(t)
        prob = DifferentialEquations.ODEProblem(diff, 0, (0, 100))
        sol = DifferentialEquations.solve(prob, reltol=1e-14, abstol=1e-14)
        V(q) = (1 - cos(q))/2;

        # This takes vectors
        φ0(q, p) = p > 0 ? sol(V(q) + p*p/2) : - sol(V(q) + p*p/2)
    end

    φ₀ = solution_underdamped();
    Dc = (1/γ)*diff_underdamped(β);
    ψ(q, p) = φ₀(q, p)/γ
    ∂ψ(q, p) = ∂φ₀(q, p)/γ
    return Dc, ψ, ∂ψ
end

function gle_solve(ν)
    qgrid = LinRange(-π, π, 300)
    egrid = LinRange(1.000001, 30, 2000)
    # egrid = LinRange(1.000001, 25, 10)

    function Sν(E)
        println(E)

        diffeq = DifferentialEquations

        # Momentum function
        P(q, E) = sqrt(2*(E-V(q)))

        # Right-hand side of BVP
        function rhs!(du, u, p, t)
            du[1] = u[1]/(ν^2*P(t, E)) - 1
            # du[2] = u[1]
        end

        # Boundary conditions of BVP
        function bc!(residual, u, p, t)
            residual[1] = u[end][1] - u[1][1]
            # residual[2] = u[1][2]
        end

        bvp = diffeq.BVProblem(rhs!, bc!, [1], (-float(π),float(π)))
        sol = diffeq.solve(bvp, diffeq.GeneralMIRK4(), dt=.05)

        s_nu = hcat(sol.(qgrid)...)[1, :]
        # S_nu = sol.(qgrid[end])[2]
        S_nu = (qgrid[2] - qgrid[1])*sum(s_nu[1:end-1])

        return s_nu, S_nu
    end

    result = Sν.(egrid)
    s_nu = hcat(first.(result)...)
    S_nu = last.(result)
    return qgrid, egrid, s_nu, S_nu

    # β = 1
    # inf = 100;
    # Zb = (2π)^(3/2) / β^(1/2) * exp(-β/2) * SpecialFunctions.besseli(0, β/2);
    # integral = QuadGK.quadgk(z -> exp(-β*z) / gle(z), 1 + 1e-10, inf)[1];
    # D = 8π^2*ν^2/(β*Zb) * integral
end

function get_controls_gle(γ, ν, recalculate)

    # Shorthand name
    dlm = DelimitedFiles

    # Directory for data
    datadir = "data_gle/precomp_nu=$ν";
    if !recalculate && isfile("$datadir/q.txt")
        println("Using existing solution!")
        qgrid = dlm.readdlm("$datadir/q.txt");
        egrid = dlm.readdlm("$datadir/e.txt");
        s_nu = dlm.readdlm("$datadir/s_nu.txt");
        S_nu = dlm.readdlm("$datadir/S_nu.txt");
    else
        # Finite element solution
        qgrid, egrid, s_nu, S_nu = gle_solve(ν);

        # Save files
        run(`mkdir -p "$datadir"`);
        dlm.writedlm("$datadir/q.txt", qgrid);
        dlm.writedlm("$datadir/e.txt", egrid);
        dlm.writedlm("$datadir/s_nu.txt", s_nu);
        dlm.writedlm("$datadir/S_nu.txt", S_nu);
    end

    dq = qgrid[2] - qgrid[1];
    de = egrid[2] - egrid[1];
    Le = egrid[end]

    # Interpolant of solution (without factor 1/√γ)
    function dz_φ(q, p)

        # Bring position in (-π, π)
        q = q - 2π*floor(Int, (q+π)/2π);

        # Symmetry of the solution
        if p < 0
            q = -q
        end

        e = sqrt(V(q) + p^2/2)

        # Avoid error when energy is too large
        if e >= Le
            println("Energy too large!")
            e = Le - 1e-12
        end

        e0 = egrid[1]
        if e <= egrid[1]
            return 0
        end

        iq, ie = 1 + floor(Int, (q+π)/dq), 1 + floor(Int, (e - e0)/de);
        x, y = (q - qgrid[iq])/dq, (e - egrid[ie])/de
        a11 = s_nu[iq, ie];
        a21 = s_nu[iq+1, ie] - a11;
        a12 = s_nu[iq, ie+1] - a11;
        a22 = s_nu[iq+1, ie+1] + a11 - s_nu[iq+1, ie] - s_nu[iq, ie+1];
        my_s_nu = a11 + a21*x + a12*y + a22*x*y
        my_S_nu = (1-y)*S_nu[ie] + y*S_nu[ie+1]
        return 2π*ν*my_s_nu/my_S_nu
    end

    # Calculate effective diffusion corresponding to discrete gradient
    if !recalculate && isfile("$datadir/D_nu=$ν.txt")
        println("Using existing approximate diffusion coefficient!")
        D = dlm.readdlm("$datadir/D_nu=$ν.txt")[1];
    else
        β, nsamples = 1, 10^8
        qsamples, psamples = Sampling.sample_gibbs(q -> (1 - cos(q))/2, β, nsamples)
        D = 1/(β*γ*ν^2)*Statistics.mean(dz_φ.(qsamples, psamples).^2)
        DelimitedFiles.writedlm("$datadir/D_nu=$ν.txt", D);
    end

    return (D, dz_φ)
end

end
