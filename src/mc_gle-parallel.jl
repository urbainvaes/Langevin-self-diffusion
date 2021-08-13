# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1.0;
ν, nbatches = 2, 28

for ibatch in 1:nbatches
    run(`tmux new-window -n "γ=$γ-b=$ibatch" julia mc_gle.jl $γ $ν "$ibatch/$nbatches"`)
end

