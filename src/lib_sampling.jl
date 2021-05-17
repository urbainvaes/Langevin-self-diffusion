module Sampling

import QuadGK
import Statistics
import LinearAlgebra

linalg = LinearAlgebra

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


# Covariance matrix for GLE
function gle_params(β, γ, ν, Δt)
    drift = [0 sqrt(γ)/ν; -sqrt(γ)/ν -1/ν^2]
    diffusion = [0 0; 0 sqrt(2/β/ν^2)]
    Q = diffusion*diffusion'
    mean = exp(drift*Δt)
    self_var = QuadGK.quadgk(t -> exp(drift*t)*Q*exp(drift'*t), 0, Δt)[1]
    cov_white = QuadGK.quadgk(t -> exp(drift*t)*diffusion, 0, Δt)[1]
    var_white = [Δt 0; 0 Δt];
    cov = [var_white cov_white'; cov_white self_var]
    cov = cov[2:end, 2:end]
    evals, evecs = linalg.eigen(cov)
    sqrt_cov = evecs*linalg.diagm(sqrt.(abs.(evals)))*evecs'
    return mean, sqrt_cov
end

end
