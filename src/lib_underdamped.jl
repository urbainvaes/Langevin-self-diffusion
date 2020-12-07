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
E₀ = 1
V_1d(q) = (1 - cos(q))/2;
S = z -> 2^(5/2) * sqrt(z) * Elliptic.E(1/z);
function ∂φ₀(q, p)
    E = V_1d(q) + p*p/2
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

φ₀ = solution_underdamped();
