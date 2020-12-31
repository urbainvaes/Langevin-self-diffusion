nbatches = 2
γ = .01

for ibatch in 1:nbatches
    run(`tmux new-window -n "γ=$γ-b=$ibatch" julia mc.jl $γ underdamped "$ibatch/$nbatches"`)
end
