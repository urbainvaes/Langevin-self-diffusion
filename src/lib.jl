# SAMPLE FROM THE GIBBS DISTRIBUTION {{{1
function sample_gibbs(V, β, np)
    p = (1/sqrt(β)) * Statistics.randn(np);
    q, naccepts = zeros(Float64, np), 0;
    while naccepts < length(q)
        v = Statistics.rand()
        u = -π + 2π*Statistics.rand();
        if v <= exp(-β*V(u))/exp(-β*V(0))
            naccepts += 1;
            q[naccepts] = u;
        end
    end
    return q, p
end

function sample_gibbs_2d(V, β, np)
    p₁ = (1/sqrt(β)) * Statistics.randn(np);
    p₂ = (1/sqrt(β)) * Statistics.randn(np);
    q₁, q₂, naccepts = zeros(Float64, np), zeros(Float64, np), 0;
    while naccepts < np
        v = Statistics.rand()
        u₁ = -π + 2π*Statistics.rand();
        u₂ = -π + 2π*Statistics.rand();
        if v <= exp(-β*V(u₁, u₂))/exp(-β*V(0, 0))
            naccepts += 1;
            q₁[naccepts] = u₁;
            q₂[naccepts] = u₂;
        end
    end
    samples = [q₁ q₂], [p₁ p₂]
end

# UNDERDAMPED LIMIT {{{1
import SpecialFunctions
import Elliptic

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

# CONTROL VARIATE {{{1
function ∇p_φ₀(q, p)
    E₀ = 1
    E = V(q) + p*p/2
    S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z);
    E > E₀ ? sign(p)*p*2π/S(E) : 0
end

# This takes vectors!
import DifferentialEquations
import Elliptic

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


# COVARIANCE MATRIX BETWEEN Δw AND OU GAUSSIAN INCREMENTS {{{1
import LinearAlgebra
linalg = LinearAlgebra;

function root_cov(γ, Δt)
    α = exp(-γ*Δt)
    if γ > 0
        cov = [Δt (1-α)/γ; (1-α)/γ (1-α*α)/(2γ)];
        rt_cov = (linalg.cholesky(cov).L);
    elseif γ == 0
        rt_cov = sqrt(Δt)*[1 0; 1 0];
    end
    return rt_cov
end
