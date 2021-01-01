function logspace(emin, emax, n)
    10 .^(emin .+ (emax-emin)*collect(0:n-1)/(n-1))
end

γs = round.(logspace(-3, 0, 10), sigdigits=3)

γs = [0.00001, 0.0000215, 0.0000464, 0.0001, 0.000215, 0.000464]
for γ in γs
    run(`tmux new-window -n "γ=$γ" julia mc-post.jl $γ underdamped`)
    run(`tmux new-window -n "γ=$γ" julia mc-post.jl $γ galerkin`)
end

# γs = [0.001, 0.00215, 0.00464, 0.01, 0.0215, 0.0464, 0.1, 0.215, 0.464, 1.0]
# δs = [-.04, -.08, -.16, -.32, -.64]
# for γ in γs
#     for δ in δs
#         run(`tmux new-window -n "γ=$γ,δ=$δ " julia mc2d.jl $γ $δ`)
#     end
# end
