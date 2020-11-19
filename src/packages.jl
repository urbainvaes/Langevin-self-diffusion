using Pkg

;export https_proxy="http://proxy.enpc.fr:3128/"
;export http_proxy="http://proxy.enpc.fr:3128/"
;export ftp_proxy="http://proxy.enpc.fr:3128/"
;export HTTPS_PROXY="http://proxy.enpc.fr:3128/"
;export HTTP_PROXY="http://proxy.enpc.fr:3128/"
;export FTP_PROXY="http://proxy.enpc.fr:3128/"

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
]

Pkg.add(dependencies)
