module FemGridap

using Gridap
import QuadGK
import Statistics
import DelimitedFiles
export get_controls

include("lib_sampling.jl")
function get_controls(γ, recalculate)

    # Directory for data
    datadir = "data/fem_gridap/γ=$γ";

    if !recalculate && isfile("$datadir/q.txt")
        println("Using existing Galerkin solution!")
        qgrid = DelimitedFiles.readdlm("$datadir/q.txt");
        pgrid = DelimitedFiles.readdlm("$datadir/p.txt");
        solution_values = DelimitedFiles.readdlm("$datadir/phi.txt");
        dq, dp = qgrid[2] - qgrid[1], pgrid[2] - pgrid[1]
        Lp = pgrid[end]
    else
        # Finite element solution
        qgrid, pgrid, solution_values = fem_solve(γ)
        Lp = pgrid[end]

        # Parameters for the calculation of the interpolant
        dq = qgrid[2] - qgrid[1]
        dp = pgrid[2] - pgrid[1]

        # Save files
        run(`mkdir -p "$datadir"`);
        DelimitedFiles.writedlm("$datadir/q.txt", qgrid);
        DelimitedFiles.writedlm("$datadir/p.txt", pgrid);
        DelimitedFiles.writedlm("$datadir/phi.txt", solution_values)
    end

    # Interpolant of solution
    function φ(q, p)
        q = q - 2π*floor(Int, (q+π)/2π);
        if abs(p) >= Lp
            p = sign(p)*(Lp - 1e-12)
        end
        iq, ip = 1 + floor(Int, (q+π)/dq), 1 + floor(Int, (p+Lp)/dp);
        x, y = (q - qgrid[iq])/dq, (p - pgrid[ip])/dp
        a11 = solution_values[iq, ip];
        a21 = solution_values[iq+1, ip] - a11;
        a12 = solution_values[iq, ip+1] - a11;
        a22 = solution_values[iq+1, ip+1] + a11 - solution_values[iq+1, ip] - solution_values[iq, ip+1];
        return a11 + a21*x + a12*y + a22*x*y
    end

    # Interpolant of gradient
    function ∂φ(q, p)
        q = q - 2π*floor(Int, (q+π)/2π);
        if abs(p) >= Lp
            p = sign(p)*(Lp - 1e-12)
        end
        iq, ip = 1 + floor(Int, (q+π)/dq), 1 + floor(Int, (p+Lp)/dp);
        x, y = (q - qgrid[iq])/dq, (p - pgrid[ip])/dp
        a11 = solution_values[iq, ip];
        a12 = solution_values[iq, ip+1] - a11;
        a22 = solution_values[iq+1, ip+1] + a11 - solution_values[iq+1, ip] - solution_values[iq, ip+1];
        return (a12 + a22*x)/dp
    end

    # Calculate effective diffusion corresponding to discrete gradient
    if !recalculate && isfile("$datadir/D.txt")
        println("Using existing approximate diffusion coefficient!")
        D = DelimitedFiles.readdlm("$datadir/D.txt")[1];
    else
        β, nsamples = 1, 10^7
        qsamples, psamples = sample_gibbs(q -> (1 - cos(q))/2, β, nsamples)
        D = γ*Statistics.mean(∂φ.(qsamples, psamples).^2)
        DelimitedFiles.writedlm("$datadir/D.txt", D);
    end

    return (D, φ, ∂φ)
end

function fem_solve(γ)
    # Inverse temperature
    β = 1

    # Potential and its derivative
    V(q) = (1 - cos(q))/2;

    # Inverse temperature
    β = 1

    # Domain
    Lp = 10
    domain = (-π,π,-Lp,Lp)
    partition = (300,500)
    # partition = (30, 50)
    model = CartesianDiscreteModel(domain, partition, isperiodic=(true,false))

    # Finite element space
    order = 1
    reffe = ReferenceFE(lagrangian, Float64, order)
    V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags="boundary")

    # Bonudary condition
    U = TrialFESpace(V0, x -> 0)

    # Parameters for Gridap
    degree = 2
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Operators for partial derivatives
    ∂q(a::CellField) = Operation(_get_x)(∇(a))
    ∂p(a::CellField) = Operation(_get_y)(∇(a))
    _get_x(a::VectorValue) = a[1]
    _get_y(a::VectorValue) = a[2]

    # Solution of the Poisson equation
    pfun(x) = (1/γ) * x[2];
    ∂qV(x) = (1/γ) * sin(x[1])/2;
    W(x) = β/2 - β^2/4 * x[2]^2
    drift(x) = [-dV(x[1]), x[2]]
    avar(u,v) = ∫( ∂p(u)⋅∂p(v) - W*u*v - pfun*∂q(u)*v + ∂qV*∂p(u)*v )*dΩ
    rhsfun(x) = (1/γ) * x[2] * exp(- (β/2) * (V(x[1]) + x[2]^2/2))
    bvar(v) = ∫( v*rhsfun )*dΩ
    op = AffineFEOperator(avar,bvar,U,V0)
    uh = solve(op)

    # (For tests) Normalization
    Zβ = sqrt(2π/β) * QuadGK.quadgk(q -> exp(-β*V(q)), -π, π)[1];

    # (For tests) Effective diffusion
    D = sum(∫(uh*rhsfun)*dΩ) * (γ/Zβ)

    # Evaluation of the solution at the nodes
    nnodes = length(Ω.node_coords)
    coord_nodes(i) = [n[i] for n in Ω.node_coords]
    qnodes, pnodes = coord_nodes.((1, 2))
    znodes = zeros(nnodes)
    for (ielem, dof_values) in enumerate(uh.cell_dof_values)
        for (idof, dof_value) in enumerate(dof_values)
            inode = Ω.cell_node_ids[ielem][idof]
            qnode, pnode = qnodes[inode], pnodes[inode]
            znodes[inode] = dof_value * exp((β/2) * (V(qnode) + pnode^2/2))
        end
    end
    znodes = reshape(znodes, size(qnodes))
    qgrid, pgrid = qnodes[:, 1], pnodes[1, :]
    return (qgrid, pgrid, znodes)
end

end
