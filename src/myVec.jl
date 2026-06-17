"""
This class is a wrapper used to test the generality of linear solvers.
These solvers should only use the methods defined in this file.
"""
mutable struct myVec{T}
    data::Vector{T}
end

ls_add(u::myVec, v::myVec; kwargs...)::myVec = myVec(u.data .+ v.data)
ls_truncate!(v::myVec; kwargs...)  = nothing # used for compressed representations of vectors such as tensor trains
Base.:*(a::Number, v::myVec)::myVec = myVec(a * v.data)
Base.:-(v::myVec)::myVec = myVec(-v.data)

LinearAlgebra.norm(v::myVec) = norm(v.data)
LinearAlgebra.dot(u::myVec, v::myVec) = dot(u.data, v.data)
