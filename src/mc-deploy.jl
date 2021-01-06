function logspace(emin, emax, n)
    10 .^(emin .+ (emax-emin)*collect(0:n-1)/(n-1))
end

γs = round.(logspace(-3, 0, 10), sigdigits=3)
# γs = [0.00001, 0.0000215, 0.0000464, 0.0001, 0.000215, 0.000464]
for γ in γs
    run(`tmux new-window -n "γ=$γ " julia mc.jl $γ galerkin 1`)
end
