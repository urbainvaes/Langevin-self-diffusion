using Pkg

dependencies = [
    "Random",
    "Statistics",
    "QuadGK",
    "Plots",
    "FFTW",
    "SparseArrays",
    "LinearAlgebra",
    "Polynomials",
    "SpecialFunctions",
    "Elliptic",
    "Plotly",
    "DelimitedFiles",
    "DifferentialEquations"
]

Pkg.add(dependencies)
