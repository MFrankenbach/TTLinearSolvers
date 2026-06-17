module LinearSolvers
    using LinearAlgebra    
    using Random
    using ITensors
    using ITensorMPS
    using FastMPOContractions

    const DEFAULT_CUTOFF = 1.e-20

    function ls_add end
    function ls_truncate! end

    include("utils.jl")
    include("testing_utilities.jl")

    # overloads for gmres and conjugate_gradient
    ls_add(t1::Array{T1,N}, t2::Array{T2,N}; kwargs...) where {T1,T2,N} = t1 + t2
    ls_truncate!(t::Array{T,N}; kwargs...) where {T,N} = t

    include("myVec.jl")
    include("gmres.jl")
    include("conjugate_gradient.jl")
    include("lanczos.jl")

    # DMRG-like solvers
    include("environment.jl")
    include("localproblem.jl")
    include("amen.jl")
    include("sweep.jl")
    include("dmrg.jl")
end