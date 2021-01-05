# Parse arguments
γ = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1.0;
δ = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : .64;
control_type = length(ARGS) > 2 ? ARGS[3] : "galerkin";
nbatches = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 25;

for ibatch in 1:nbatches
    run(`tmux new-window -n "γ=$γ-δ=$δ-b=$ibatch" julia mc2d.jl $γ $δ $control_type "$ibatch/$nbatches"`)
end
