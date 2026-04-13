include("testimports.jl")

for f in readdir(joinpath(@__DIR__, "tests/"); join=true)
    include(f)
end