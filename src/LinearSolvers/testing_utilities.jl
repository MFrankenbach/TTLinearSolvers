"""
Make random MPO of length `len`. `linkdims` is either an integer or a vector
of `2len-1` integers specifying the bond dimensions of the MPS that the output MPO is made from. 
"""
function make_test_MPO(len::Int, sitedim::Int=2; suffix="", linkdims=3, rng=MersenneTwister(101))
    sites = [Index(sitedim, "Site $(i)" * suffix) for i in 1:len]
    sitesp = prime.(sites)
    allsites = vcat([ [sites[i], sitesp[i]] for i in 1:len ]...)
    mps = random_mps(rng, Float64, allsites; linkdims=linkdims)
    mpo = MPO([mps[2(i-1)+1]*mps[2i] for i in 1:len])
    return mpo
end

"""
Create Hamiltonian MPO for Heisenberg spin-1/2 chain of length `len` with open boundary conditions.
Shift by `pos_shift` to make positive definite.
"""
function make_Heisenberg(;len=7, pos_shift=2.0)
    sites = siteinds("S=1/2",len)
    os = OpSum()
    for j=1:len-1
    os += 0.5,"S+",j,"S-",j+1
    os += 0.5,"S-",j,"S+",j+1
    os += "Sz",j,"Sz",j+1
    end
    A = MPO(os,sites)
    # make positive definite
    A = A + identity_MPO(eltype(A[1]), len; sites=vcat(siteinds(A)...), localdims=fill(2,len))*pos_shift
    return A, sites
end

function to_matrix(A::MPO, incoming_sites::Vector{<:Index}, outgoing_sites::Vector{<:Index})
    A_arr = array(prod(A.data), outgoing_sites..., incoming_sites...)
    return reshape(A_arr, prod(dim.(outgoing_sites)), prod(dim.(incoming_sites)))
end

function to_matrix(A::MPO, incoming_sites::Vector{<:Index})
    return to_matrix(A, incoming_sites, prime.(incoming_sites))
end
