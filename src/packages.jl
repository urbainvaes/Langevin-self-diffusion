using Pkg

dependencies = [
    "Arpack",
    "DelimitedFiles",
    "DifferentialEquations",
    "Elliptic",
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
