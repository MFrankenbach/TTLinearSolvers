function _lengthcheck(x::AbstractMPS, y::AbstractMPS)
    length(x) == length(y) || error("MPS and MPO must have the same length")
end

function AdagA(A::MPO)
    Adag = replaceprime(conj(A), 0=>2)
    AdagA = FastMPOContractions.contract_mpo_mpo(Adag, A)
    AdagA = replaceprime(AdagA, 2=>1)
    return AdagA
end

"""
Create an MPO that is the identity of given dimension `localdims[i]` on site `i`.
Optionally provide `2R` sites, so that `sites[2i]` and `sites[2i-1]` sit on block `i`.
"""
function identity_MPO(::Type{T}, R::Int; sites=nothing, localdims=fill(2,R)) where T<:Number
    isnothing(sites) || length(sites)==2R || error("Wrong number of sites given ($(length(sites)) vs $R)")
    length(localdims)==R || error("Wrong number of local dimensions given ($(length(localdims)) vs $R)")
    isnothing(sites) || (all(dim.(sites[1:2:end]).==localdims) && all(dim.(sites[2:2:end]).==localdims)) || 
        error("Site dimensions do not match local dimensions $(localdims)")

    linkindices = [Index(1,"link,l=$(i)") for i in 1:R-1]
    sites = isnothing(sites) ? [Index(localdims[div(r-1,2)+1], "site$(r)") for r in 1:2R] : sites
    blocks = Vector{ITensor}(undef, R)
    for r in eachindex(blocks)
        ld = localdims[r]
        if r>1 && r<R
            blocks[r] = ITensor(reshape(diagm(ones(T,ld)), ld,ld,1,1), sites[2r-1],sites[2r],linkindices[r-1],linkindices[r])
        elseif r==1
            blocks[r] = ITensor(reshape(diagm(ones(T,ld)), ld,ld,1), sites[2r-1],sites[2r],linkindices[r])
        elseif r==R
            blocks[r] = ITensor(reshape(diagm(ones(T,ld)), ld,ld,1), sites[2r-1],sites[2r],linkindices[r-1])
        end
    end
    return MPO(blocks)
end

"""
Compute a0⋅I+a1⋅A, where I is the identity MPO.
"""
function add_identity(A::MPO, a0::Number, a1::Number)
    all(dim(s[1])==dim(s[2]) for s in siteinds(A)) ||
        error("Cannot add identity to MPO incoming and outgoing physical indices of different dimension.")
    id = identity_MPO(eltype(A[1]), length(A); sites=vcat(siteinds(A)...), localdims=[dim(s[1]) for s in siteinds(A)])
    return ITensorMPS.add(a1*A,a0*id; alg="directsum")
end

function unity(inds::Vararg{<:Index})
    unity = reshape(diagm(ones(Float64, prod(dim.(inds)))), dim.(inds)..., dim.(inds)...)
    unity_t = ITensor(unity, inds..., prime.(inds)...)
    return unity_t
end

unity(inds) = unity(inds...)

function safe_norm(x::AbstractMPS; abstol=1.e-14, reltol=1.e-14)
    n_sq = inner(x, x)
    if abs(n_sq) < abstol
        return abs(n_sq)^0.5
    elseif real(n_sq)>0 && abs(imag(n_sq))/abs(n_sq) < reltol
        return sqrt(real(n_sq))
    else
        throw(DomainError(n_sq, "Cannot compute norm of MPS with negative inner product with itself."))
    end
end

function relative_error(x::MPS, y::MPS; reltol=1.e-12)
    d = ITensorMPS.add(x, -y; alg="directsum")
    nd_sq_rel = inner(d, d) / min(safe_norm(x), safe_norm(y))
    return sqrt(abs(real(nd_sq_rel)))
    if imag(nd_sq_rel)>reltol || real(nd_sq_rel)<-reltol
        @show inner(d,d), safe_norm(x), safe_norm(y)
        throw(DomainError(nd_sq_rel, "Relative error between MPSs is not positive."))
    else
        return sqrt(real(nd_sq_rel))
    end
end