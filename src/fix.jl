#!/usr/bin/env julia
import DelimitedFiles

# γs = [.464, .01, .0215, .0464, .1, .215, .464, 1.0]
γs = [.464]
δs = [.04, .08, .16, .32, .64]

control_type = "galerkin";

readf, writef = DelimitedFiles.readdlm, DelimitedFiles.writedlm;
for γ in γs
    for δ in δs
        datadir = "data2d/$control_type-γ=$γ-δ=$δ";
        if !isdir(datadir)
            continue
        end
        datafiles = readdir(datadir);
        index(filename) = parse(Int, match(r"i=(\d+)", filename).captures[1]);
        ξfiles = filter(s -> occursin(r"Δt=0.01-i=.*ξ.txt", s), datafiles);
        ξfiles = map(s -> "$datadir/$s", sort(ξfiles, by=index));

        for f in ξfiles
            ξ = readf(f)
            ξ = ξ*γ
            writef(f, ξ)
        end
    end
end
