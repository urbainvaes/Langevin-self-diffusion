using Gridap
using GridapGmsh
import Plots
import LinearAlgebra
la = LinearAlgebra
# include("lib_underdamped.jl")
# include("lib_galerkin.jl")

# Potential and its derivative
V(q) = (1 - cos(q))/2;
γ, β = .00001, 1

# W(x) = 1 + x[2]^2

domain = (-π,π,-10,10)
partition = (400,400)
# model = CartesianDiscreteModel(domain, partition, isperiodic=(true,false))
# model = GmshDiscreteModel("periodic_square.msh")
model = GmshDiscreteModel("mymesh.msh")
# model = GmshDiscreteModel("halfmesh.msh")

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V0 = TestFESpace(model,reffe, dirichlet_tags=["top","ptop","bottom","pbottom"])
# V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["top", "ptop", "bottom", "pbottom"])
# V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags="boundary")
U = TrialFESpace(V0, x -> 0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)


∂q(a::CellField) = Operation(_get_x)(∇(a))
∂p(a::CellField) = Operation(_get_y)(∇(a))
_get_x(a::VectorValue) = a[1]
_get_y(a::VectorValue) = a[2]

p(x) = (1/γ) * x[2];
∂qV(x) = (1/γ) * sin(x[1])/2;
W(x) = β/2 - β^2/4 * x[2]^2
drift(x) = [-dV(x[1]), x[2]]

a(u,v) = ∫( ∂p(u)⋅∂p(v) - W*u*v - p*∂q(u)*v + ∂qV*∂p(u)*v )*dΩ
# a(u,v) = ∫( ∇(u)⋅∇(v) )*dΩ
rhs(x) = (1/γ) * x[2] * exp(- (β/2) * (V(x[1]) + x[2]^2/2))
b(v) = ∫( v*rhs )*dΩ
op = AffineFEOperator(a,b,U,V0)
uh = solve(op)

# nnodes = length(Ω.node_coords)
# coord_nodes(i) = [n[i] for n in Ω.node_coords]
# xnodes, ynodes = coord_nodes.((1, 2))
# znodes = zeros(nnodes)
# for (ielem, dof_values) in enumerate(uh.cell_dof_values)
#     for (idof, dof_value) in enumerate(dof_values)
#         inode = Ω.cell_node_ids[ielem][idof]
#         znodes[inode] = dof_value
#     end
# end
# znodes = reshape(znodes, size(xnodes))
# underdamped = φ₀.(xnodes, ynodes) .* exp.(- (β/2) * (V.(xnodes) + ynodes.^2/2))
# error = γ*znodes - underdamped
# Plots.contour(xnodes[:, 1], ynodes[1, :], error', 
#               fill=(true, :viridis), levels=20, size=(1600,1000))
# Plots.ylims!((-4, 4))

# sol_galerkin = galerkin_solve(.001)[2]
# sol_galerkin = sol_galerkin.(xnodes, ynodes)
# sol_galerkin .*= exp.(- (β/2) * (V.(xnodes) + ynodes.^2/2))

writevtk(Ω, "solution", order=order, cellfields=["uh"=>uh])
