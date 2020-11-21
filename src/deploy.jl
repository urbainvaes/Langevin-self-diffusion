function logspace(emin, emax, n)
    10 .^(emin .+ (emax-emin)*collect(0:n-1)/(n-1))
end

γs = round.(logspace(-3, 0, 10), sigdigits=3)


δ = .04
for γ in γs
    run(`tmux new-window -n "γ=$γ,δ=$δ " julia mc.jl $γ $δ`)
end
