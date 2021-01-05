γs = [0.001, 0.00215, 0.00464, 0.01, 0.0215, 0.0464, 0.1, 0.215, 0.464, 1.0]
δs = [.04, .08, .16, .32, .64]
for γ in γs
    for δ in δs
        run(`tmux new-window -n "γ=$γ,δ=$δ" julia mc2d-post.jl $γ $δ underdamped`)
        run(`tmux new-window -n "γ=$γ,δ=$δ" julia mc2d-post.jl $γ $δ galerkin`)
    end
end
