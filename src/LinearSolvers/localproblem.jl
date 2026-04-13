"""
For 2-site (i.e., nsite=2), gives left hand side of:
```
L----b--b----R  
L    |  |    R  =  --b_loc--
L---       --R        | |
```
"""
function local_RHS_dense(
    env_L::ITensor,
    b::MPS,
    env_R::ITensor,
    firstsite::Integer,
    nsite::Integer=_default_nsite()
    )::ITensor
    local_b = env_L
    for site in _locsites(b, firstsite, nsite)
        local_b *= b[site]
    end
    local_b *= env_R
    return local_b
end

"""
Starting from (and including) `firstsite`, multiply site tensors.
"""
function local_x_dense(
    x::MPS,
    firstsite::Integer,
    nsite::Integer=_default_nsite()
    )::ITensor
    local_x = ITensor(one(eltype(x[1])))
    for site in _locsites(x, firstsite, nsite)
        local_x *= x[site]
    end
    return local_x
end

"""
Split a tensor `local_x` into tensor blocks, each with one index in `siteinds`.
The tensor blocks are in the same order as `siteinds`.
`leftlink` is the leftmost link index and must be present in `local_x` if not `nothing`.
```
leftlink-xxx- --> -x-x-x-
         |||       | | |
         siteinds
```
"""
function split_x_local(
    local_x::ITensor,
    siteinds::Vector{<:Index},
    leftlink::Union{Index,Nothing};
    maxdim::Integer=typemax(Int),
    cutoff=nothing
    )::Vector{ITensor}
    hasinds(local_x, siteinds) ||
        error("local_x with indices $(inds(local_x)) does not have the specified site indices $(siteinds). Left link is $leftlink.")
    if length(siteinds)==1
        return [local_x]
    else
        ret = ITensor[]
        remainder = local_x
        link_act = leftlink
        for i in 1:length(siteinds)-1
            # TODO does the cutoff refer to the square of the summed singular values?
            U, S, V = svd(remainder, filter(x -> !isnothing(x), (siteinds[i], link_act))...; cutoff=cutoff, maxdim=maxdim)
            push!(ret, U)
            remainder = S*V
            link_act = only(commoninds(U, remainder))
        end
        push!(ret, remainder)
    end
    return ret
end

"""
Perform the following computation:
```
L---xxxx---R  
L   |  |   R
L-- A--A---R  = --Ax_loc--
L   |  |   R      |   |
L--      --R
```
"""
function apply_Aloc_dense(
    env_L::ITensor,
    A::MPO,
    env_R::ITensor,
    x_loc::ITensor,
    firstsite::Integer,
    nsite::Integer=_default_nsite()
    )::ITensor
    local_Ax = env_L * x_loc
    for site in _locsites(A, firstsite, nsite)
        local_Ax *= A[site]
    end
    local_Ax *= env_R
    return noprime(local_Ax)
end

"""
Compute local operator as a dense tensor:
```
L---    ---R  
L   |  |   R      |   |
L-- A--A---R  = --Ax_loc--
L   |  |   R      |   |
L--      --R
```
"""
function compute_Aloc_dense(
    env_L::ITensor,
    A::MPO,
    env_R::ITensor,
    firstsite::Integer,
    nsite::Integer=_default_nsite()
    )::ITensor
    local_A = env_L
    for site in _locsites(A, firstsite, nsite)
        local_A *= A[site]
    end
    local_A *= env_R
    return local_A
end

"""
Build and solve local problem of Ax=b:
```
L---xxxx---R
L   |  |   R
L-- A--A---R =  L_RHS---bbbb---R_RHS
L   |  |   R      |     |  |     |
L--      --R    L_RHS--      --R_RHS
```
The MPS `x` is updated in-place.
"""
function localupdate!(
    env_L::ITensor,
    env_R::ITensor,
    env_L_RHS::ITensor,
    env_R_RHS::ITensor,
    A::MPO,
    x::MPS,
    b::MPS,
    firstsite::Integer,
    nsite::Integer=_default_nsite();
    solver=:gmres,
    normchange::Union{Nothing,Base.RefValue{Float64}}=nothing,
    howverbose=0,
    maxdim::Integer=typemax(Int),
    cutoff=nothing,
    kwargs...
    )::ITensor

    local_b = local_RHS_dense(env_L_RHS, b, env_R_RHS, firstsite, nsite)
    local_x = local_x_dense(x, firstsite, nsite)
    x_inds = inds(local_x)
    local_b_arr = array(local_b, x_inds...)
    local_x_arr = array(local_x, x_inds...)
    function _apply_A(x_loc_arr)
        x_loc = ITensor(x_loc_arr, x_inds...)
        return array(apply_Aloc_dense(env_L, A, env_R, x_loc, firstsite, nsite), x_inds...)
    end

    # Krylov solve
    if solver==:gmres
        howverbose>1 && printstyled("   GMRES error before: $(norm(_apply_A(local_x)-local_b))\n"; color=:red, bold=true)
        x_loc_new_arr, conv, relres = gmres(_apply_A, local_b_arr, local_x_arr; ValueType=eltype(local_b_arr), kwargs...)
        x_loc_new = ITensor(x_loc_new_arr, x_inds...)
        howverbose>1 && printstyled("   GMRES error after: $(norm(_apply_A(x_loc_new)-local_b))\n"; color=:red, bold=true)
    elseif solver==:exact
        howverbose>1 && printstyled("   Solving local problem exactly...\n"; color=:red, bold=true)
        Aloc = compute_Aloc_dense(env_L, A, env_R, firstsite, nsite)
        Aloc_mat = reshape(array(Aloc, prime.(x_inds)..., x_inds...), prod(dim.(x_inds)), prod(dim.(x_inds)))
        x_loc_new_mat = Aloc_mat \ reshape(local_b_arr, prod(dim.(x_inds)), 1)
        x_loc_new = ITensor( reshape(x_loc_new_mat, dim.(x_inds)...), x_inds... )
    elseif solver==:conjugate_gradient
        howverbose>1 && printstyled("   CG: Energy before: $(energy(A,x,b))\n"; color=:red, bold=true)
        x_loc_new_arr = conjugate_gradient(_apply_A, local_b_arr, local_x_arr; kwargs...)
        x_loc_new = ITensor(x_loc_new_arr, x_inds...)
        howverbose>1 && printstyled("   CG: Energy after: $(energy(A,x,b))\n"; color=:red, bold=true)
    end

    # update solution vector
    siteinds_x = siteinds(x)
    locsites = _locsites(x, firstsite, nsite)
    x.data[locsites] .= split_x_local(
        x_loc_new,
        siteinds_x[locsites],
        _leftlink(x, firstsite);
        maxdim=maxdim,
        cutoff=cutoff
    )
    x.llim=locsites[end]-1
    x.rlim=locsites[end]+1

    if !isnothing(normchange)
        normchange[] = norm(prod(x.data[locsites])) - norm(local_x)
    end

    if solver==:gmres && !conv
        @warn "GMRES in local update from sites $firstsite to $(_lastsite(b, firstsite+nsite-1)) did not converge. Relative residual: $relres."
    end

    return x_loc_new
end

_firstsite(X::AbstractMPS, leftsweep::Bool)=leftsweep ? length(X) : 1
_finalsite_sweep(X::AbstractMPS, nsite::Integer, leftsweep::Bool)=leftsweep ? nsite+1 : length(X)-nsite
_lastsite(X::AbstractMPS, site::Integer)=min(site,length(X))
_locsites(X::AbstractMPS, firstsite::Integer, nsite::Integer)=firstsite:_lastsite(X,(firstsite+nsite-1))
_leftlink(x::MPS, firstsite::Integer)=firstsite>1 ? commonind(x[firstsite-1], x[firstsite]) : nothing