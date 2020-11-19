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
]

Pkg.add(dependencies)
