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

# COVARIANCE MATRIX BETWEEN Δw AND OU GAUSSIAN INCREMENTS {{{1
import LinearAlgebra
linalg = LinearAlgebra;

function root_cov(γ, Δt)
    α = exp(-γ*Δt)
    cov = [Δt (1-α)/γ; (1-α)/γ (1-α*α)/(2γ)];
    if linalg.isposdef(cov)
        rt_cov = (linalg.cholesky(cov).L);
    else
        rt_cov = sqrt(Δt)*[1 0; 1 0];
    end
    return rt_cov
end
