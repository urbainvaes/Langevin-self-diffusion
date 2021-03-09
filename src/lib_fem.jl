using Gridap
import LinearAlgebra
la = LinearAlgebra

# Potential and its derivative
V(q) = (1 - cos(q))/2;
γ, β = 1, 1

# W(x) = 1 + x[2]^2

domain = (-π,π,-10,10)
partition = (100,100)
model = CartesianDiscreteModel(domain, partition, isperiodic=(true,false))

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags="boundary")
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
rhs(x) = (1/γ) * x[2] * exp(- (β/2) * (V(x[1]) + x[2]^2/2))
b(v) = ∫( v*rhs )*dΩ
op = AffineFEOperator(a,b,U,V0)
uh = solve(op)

writevtk(Ω, "solution", order=order, cellfields=["uh"=>uh])
