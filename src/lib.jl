# SAMPLE FROM THE GIBBS DISTRIBUTION {{{1
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
