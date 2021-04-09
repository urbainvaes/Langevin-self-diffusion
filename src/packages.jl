using Pkg

dependencies = [
    "Arpack",
    "DelimitedFiles",
    "DifferentialEquations",
    "Elliptic",
    "FastGaussQuadrature",
    "FFTW",
    "Gridap",
    "GridapGmsh",
    "LinearAlgebra",
    "Plotly",
    "Plots",
    "Polynomials",
    "QuadGK",
    "Random",
    "SparseArrays",
    "SpecialFunctions",
    "Statistics",
]

Pkg.add(dependencies)
