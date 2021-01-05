# γs = [.001, .00215, .00464, .01, .0215, .0464, .1, .215, .464, 1.0]
γs = [.01, .0215, .0464, .1, .215, .464, 1.0]
# δs = [.04, .08, .16, .32, .64]
δs = [.16, .32, .64]

nbatches = 4
for γ in γs
    for δ in δs
        for b in 1:nbatches
            run(`tmux new-window -n "γ=$γ,δ=$δ,b=$b" julia mc2d.jl $γ $δ galerkin $b/$nbatches`)
        end
    end
end
