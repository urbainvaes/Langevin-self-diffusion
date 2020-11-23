function logspace(emin, emax, n)
    10 .^(emin .+ (emax-emin)*collect(0:n-1)/(n-1))
end

γs = round.(logspace(-3, 0, 10), sigdigits=3)

γs = [0.001, 0.00215, 0.00464, 0.01, 0.0215, 0.0464, 0.1, 0.215, 0.464, 1.0]
δ = [-.04, -.08, -.16, -.32, -.64]
for γ in γs
    for δ in δs
        run(`tmux new-window -n "γ=$γ,δ=$δ " julia mc2d.jl $γ $δ`)
    end
end
