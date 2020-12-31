# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : .01;
control_type = length(ARGS) > 1 ? ARGS[2] : "galerkin";
nbatches = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 25;

for ibatch in 1:nbatches
    run(`tmux new-window -n "γ=$γ-b=$ibatch" julia mc.jl $γ $control_type "$ibatch/$nbatches"`)
end
