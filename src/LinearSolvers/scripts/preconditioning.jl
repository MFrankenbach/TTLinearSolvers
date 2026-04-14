#=
Explore some preconditioning ideas.
=#

using LinearAlgebra
using ITensors, ITensorMPS
using Random
using Mochi.LinearSolvers: AdagA, make_test_MPO, to_matrix
import FastMPOContractions as FMPOC

ITensors.disable_warn_order()

function to_matrix(A::MPO)
    A_dense = prod(A.data)
    inds_nop = filter(i->plev(i)==0, inds(A_dense))
    inds_p = filter(i->plev(i)>0, inds(A_dense))
    @assert length(inds_nop) == length(inds_p) "MPO does not have balanced primed and unprimed indices."
    @assert length(inds_nop) == length(A) "Number of sites in MPO does not match number of unprimed indices."
    A_mat = to_matrix(A, collect(inds_nop), collect(inds_p))
    return A_mat
end

function LinearAlgebra.cond(A::MPO)
    return LinearAlgebra.cond(to_matrix(A))
end

"""
Approximate inverse as inverse of rank-1 approximation of `A`.
"""
function rank1_inverse!(A::MPO, incoming_sites::Vector{<:Index})
    length(A)==length(incoming_sites) ||
        error("Length of incoming_sites $(length(incoming_sites)) must match length of MPO $(length(A)).")
    a = ITensorMPS.truncate!(deepcopy(A); maxdim=1)
    # svd each block along physical dimensions
    Us = Vector{ITensor}(undef, length(A))
    Vs = Vector{ITensor}(undef, length(A))
    Ss = Vector{ITensor}(undef, length(A))
    links = linkinds(a)
    for ia in eachindex(a)
        # remove 1D link indices
        block_mat = a[ia]
        for il in max(ia-1,1):min(ia,length(links))
            block_mat *= onehot(links[il] => 1)
        end
        # println()
        # display(array(block_mat, inds(block_mat)...))
        # @show inds(block_mat), incoming_sites[ia]
        # @show svd(array(block_mat, inds(block_mat)...)).S
        # println()
        u, s, v = svd(block_mat, incoming_sites[ia])
        s_mat = array(s, inds(s)...)
        s_inv_sqrt = ITensor( diagm(1 ./ sqrt.(diag(s_mat))), inds(s)... )
        toprime = only(uniqueinds(s_inv_sqrt, v))
        s_inv_sqrt2 = replaceind(s_inv_sqrt, toprime, prime(toprime))
        A[ia] *= u*s_inv_sqrt
        A[ia] *= v*s_inv_sqrt2
        Us[ia] = u
        Ss[ia] = s
        Vs[ia] = v
    end
    return A, Us, Ss, Vs
end

function make_random_MPO(; rng=MersenneTwister(1234), len::Int=6, sitedim::Int=2)
    sites = [Index(sitedim, "Site $(i)") for i in 1:len]
    t = ITensor(randn(rng, Float64, dim.(sites)..., dim.(sites)...), sites..., prime.(sites)...)
    mpo = MPO(t, sites)
    return mpo, sites
end

"""
Check condition number before and after preconditioning with `method`.
`method` should be a function that takes `A` as an argument and returns the preconditioned `A`.
"""
function _preconditioning_check(A, method)
    println("Condition number before preconditioning: $(cond(A))")
    A_prec = method(deepcopy(A))
    @show A
    @show A_prec
    display(to_matrix(A))
    display(to_matrix(A_prec))
    println("Condition number after preconditioning: $(cond(A_prec))\n")
end

function check_normalize_precondition()
    R = 6
    rng = MersenneTwister(42)
    A, sites = make_random_MPO(;rng=rng, len=R, sitedim=2)
    _preconditioning_check(A, a -> rank1_inverse!(a, sites)[1])
end