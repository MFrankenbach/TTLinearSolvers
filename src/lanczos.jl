function energy_converged(hist, tol)
    if length(hist) < 2
        return false
    end
    return abs(hist[end] - hist[end-1]) < tol
end

"""
Finds the lowest eigenvalue and corresponding eigenvector of the Hermitian operator `H`.
Convergence is reached when the change in energy between successive iterations is less than `tol`.
# Arguments
- `H`: Function that applies the Hermitian operator to a vector.
- `v0`: Initial guess vector.
- `ValueType`: In which field scalar products live.
- `add_kwargs`: Keyword arguments passed to the `add` function used for vector operations.
"""
function lanczos(
    H, v0;
    ValueType=ComplexF64,
    maxiter=100,
    tol=1e-8,
    howverbose=0,
    add_kwargs=Dict()
)
    n = maxiter

    howverbose > 0 && printstyled("Starting Lanczos with maxiter = $maxiter, tol = $tol\n"; color=:blue, bold=true)

    V = Vector{typeof(v0)}(undef, n)  # orthonormal Krylov basis
    Heff = zeros(ValueType, n, n)  # effective Hamiltonian

    V[1] = (1/norm(v0)) * v0 
    v1p = H(V[1])
    a0 = dot(V[1], v1p)
    v1p = ls_add(v1p, -a0 * V[1]; add_kwargs...)
    b1 = norm(v1p)
    V[2] = (1 / b1) * v1p

    Heff[1,1] = a0
    Heff[1,2] = b1
    Heff[2,1] = b1

    E_hist = Float64[]

    Ψ = nothing
    for k in 3:n
        vp = H(V[k-1])
        a = dot(V[k-1], vp)
        vp = ls_add(vp, -a * V[k-1]; add_kwargs...)
        vp = ls_add(vp, -Heff[k-2,k-1] * V[k-2]; add_kwargs...)
        b = norm(vp)
        V[k] = (1 / b) * vp
        Heff[k-1,k-1] = a
        Heff[k-1,k] = b
        Heff[k,k-1] = b

        eigval, eigvec = eigen(Hermitian(Heff[1:k-1, 1:k-1]))
        E = eigval[1]
        Ψ = eigvec[:,1]
        push!(E_hist, E)
        if energy_converged(E_hist, tol)
            howverbose > 0 && println("Lanczos converged after $(k-1) iterations with energy $E")
            # reconstruct eigenvector
            v_eigen = linear_combination(Ψ, V[1:k-1]; add_kwargs=add_kwargs)
            return E, v_eigen
        end
    end

    @warn "Lanczos did not converge within $maxiter iterations. Returning last estimate."
    return E_hist[end], linear_combination(Ψ, V; add_kwargs=add_kwargs)
end

function linear_combination(Ψ::Vector{<:Number}, V; add_kwargs=Dict())
    v_eigen = Ψ[1]*V[1]
    for j in 2:length(Ψ)
        v_eigen = ls_add(v_eigen, Ψ[j] * V[j]; add_kwargs...)
    end
    return v_eigen
end