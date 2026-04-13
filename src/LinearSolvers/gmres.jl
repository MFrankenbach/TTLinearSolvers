"""
    gmres(A, b; x0=zero(b), maxiter=100, tol=1e-6)

Solves Ax = b using the GMRES method. Works with custom vector types.
Requires only `+`, `*`, `dot`, `norm`, and `zero`.
The first version of this function was written by ChatGPT.

# Arguments:
- A: function A(v) that returns A applied to v (same type as v)
- b: right-hand side vector
- x0: initial guess (no default)
- maxiter: maximum number of iterations
- tol: relative residual tolerance
- add_kwargs: keyword arguments for the `add` function (e.g., a cutoff for tensor train addition)
- gmres_truncate_cutoff: cutoff for truncation of Krylov vectors and solution vector
- operator_kwargs: keyword arguments for the operator application
- relaxed: whether to relax the truncation cutoff based on the current residual

# Returns:
- x: approximate solution
- flag: true if converged, false if not
- relres: final relative residual norm
"""
function gmres(
    A, b, x0;
    ValueType=ComplexF64,
    maxiter=100,
    tol=1e-6,
    howverbose=0,
    add_kwargs=Dict(),
    gmres_truncate_cutoff=DEFAULT_CUTOFF,
    orthocheck=true,
    residual_check=false,
    relaxed=false,
    operator_kwargs=Dict()
    )
    n = maxiter  # dimension of Krylov space
    operator_kwargs_ = deepcopy(operator_kwargs)
    howverbose > 0 && printstyled("Starting GMRES with maxiter = $maxiter, tol = $tol\n"; color=:blue, bold=true)

    relaxed && @warn "Relaxed GMRES is experimental."

    # Initialize storage
    V = Vector{typeof(b)}(undef, n+1)  # Orthonormal basis
    H = zeros(ValueType, n+1, n)         # Hessenberg matrix

    r0 = ls_add(b, -A(x0; operator_kwargs_...); add_kwargs...)
    beta = norm(r0)
    nb = norm(b)
    relres = beta / nb

    if relres < tol
        return x0, true, relres
    end

    V[1] = (1 / beta) * r0  # v₁ = r₀ / ||r₀||
    g = zeros(n+1)
    g[1] = beta

    xk = nothing
    y = nothing

    residual_check && @info "Residual check enabled: This incurs one more application of the operator in each iteration!"

    for k in 1:n

        # relaxed TT-GMRES
        relax_fac = relaxed ? 1 / sqrt(relres*nb/beta) : 1.0
        trunc_cut_relaxed = relaxed ? gmres_truncate_cutoff*relax_fac : gmres_truncate_cutoff 
        if relaxed && haskey(operator_kwargs_, _cutoff_key())
            operator_kwargs_[_cutoff_key()] = operator_kwargs[_cutoff_key()] * relax_fac
            printstyled("    GMRES: Using relaxed cutoff = $trunc_cut_relaxed / $(operator_kwargs_[_cutoff_key()])\n"; color=:yellow)
        end

        @showtime w = A(V[k]; operator_kwargs_...)

        howverbose > 2 && (println("    GMRES: A(Vₖ)"); display(w))

        # Arnoldi: orthogonalize
        t = @elapsed for j in 1:k
            H[j,k] = dot(V[j], w)
            w = ls_add(w, -H[j,k] * V[j]; add_kwargs...)
            ls_truncate!(w; cutoff=trunc_cut_relaxed) # TODO: should we scale the cutoff with k?
        end
        howverbose>0 && println("           Orthogonalization time in iteration $k: $t seconds")
        H[k+1,k] = norm(w)

        if H[k+1,k] ≈ 0
            error("Krylov vector approximately zero: this is not yet implemented")
            break
        end

        V[k+1] = (1 / H[k+1,k]) * w

        ls_truncate!(V[k+1]; cutoff=trunc_cut_relaxed)
        if howverbose > 2
            println("    GMRES: $(k+1)-th Krylov vector:")
            display(V[k+1])
            println()
        end

        # Solve least squares min ||g - H*y|| with QR or backsubstitution
        y = H[1:k+1, 1:k] \ g[1:k+1]

        # printstyled("        GMRES: Step $(k) solution:\n", color=:blue, bold=true)
        # printstyled("        GMRES: Step $(k) rank of Vₖ₊₁: $(BubbleTeaCI.rank(V[k+1])):\n", color=:blue, bold=false)
        # display(y)

        if orthocheck
        # Check orthogonality
        howverbose>1 && printstyled("    GMRES Metric, iteration $(k):\n"; bold=true, color=:red)
        metric = zeros(ValueType,k+1,k+1)
        for i in 1:k+1
            for j in 1:k+1
                metric[i,j] = dot(V[i], V[j])
            end
        end
        all(abs.(metric .- diagm(ones(ValueType,size(metric,1)))) .>= 1.e-10) && @warn "Loss of orthogonality in GMRES!"
        howverbose>1 && display(round.(metric; digits=12))
        end

        if residual_check
            # Check true residual
            @showtime xk = build_solution_vector(y, V, x0; add_kwargs=add_kwargs, cutoff=gmres_truncate_cutoff)
            howverbose>0 && println("    GMRES: True residual in iteration $(k): $(norm(b - A(xk; operator_kwargs_...))/nb)")
        end

        # Check convergence
        relres = norm(g[1:k+1] - H[1:k+1, 1:k] * y) / nb  # Check residual norm
        howverbose > 0 && println("    GMRES Iteration $k: relative residual = $relres")
        if relres < tol
            # Build solution approximation x_k = x₀ + V_k * y
            howverbose > 0 && printstyled("Converged in $k iterations with relative residual $relres. Building solution vector...\n"; color=:blue, bold=true)
            @showtime xk = build_solution_vector(y, V, x0; add_kwargs=add_kwargs, cutoff=gmres_truncate_cutoff)
            howverbose > 0 && printstyled("... done\n"; color=:blue, bold=true)
            return xk, true, relres
        end

    end

    xk = build_solution_vector(y, V, x0; add_kwargs=add_kwargs, cutoff=trunc_cut_relaxed)
    @warn "GMRES did not converge in $maxiter iterations. Final relative residual: $relres."
    return xk, false, relres  # Did not converge
end

function build_solution_vector(y::Vector{<:Number}, V, x0; add_kwargs=Dict(), cutoff)
    xk = x0
    for j in eachindex(y)
        xk = ls_add(xk, y[j]*V[j]; add_kwargs...)
        ls_truncate!(xk; cutoff=cutoff) # TODO: should we scale the cutoff with k?
    end
    return xk
end

_cutoff_key() = :cutoff

#=
# Extremely unstable numerically
function gmres_lazysum(A, b, x0; ValueType=ComplexF64, maxiter=100, tol=1e-6, howverbose=0, add_kwargs=Dict())
    n = maxiter  # dimension of Krylov space

    howverbose > 0 && printstyled("Starting GMRES with maxiter = $maxiter, tol = $tol\n"; color=:blue, bold=true)

    # Initialize storage
    Ars = Vector{typeof(b)}(undef, n+1)  # powers of A applied to initial residue
    Vc = zeros(ValueType, n+1, n+1) # coefficients of orthonormal basis: each vector corresponds to a column
    H = zeros(ValueType, n+1, n)         # Hessenberg matrix

    nb = norm(b)
    r0 = BubbleTeaCI.add(b, -A(x0); add_kwargs...)
    beta = norm(r0)
    relres = beta / nb

    if relres < tol
        return x0, true, relres
    end

    # First Krylov vector: v₁ = r₀ / ||r₀||
    Vc[1,1] = (1 / beta)
    Ars[1] = r0  
    g = zeros(n+1)
    g[1] = beta

    xk = x0
    metric = zeros(ValueType, n+1, n+1)  # metric for orthonormalization
    metric[1,1] = beta^2 # ⟨r₀,r₀⟩

    for k in 1:n
        # w = Aᵏr₀
        Ars[k+1] = A(Ars[k])
        Vc[k+1,k+1] = 1 # before orthonormalization

        # extend metric
        m = [dot(Ars[i],Ars[k+1]) for i in 1:k+1]
        metric[1:k+1,k+1] .= m
        metric[k+1,1:k] .= conj.(m[1:end-1])

        howverbose > 2 && (println("    GMRES: Aᵏr₀"); display(w))

        # orthogonalize
        for j in 1:k
            # compute ⟨vⱼ,w⟩ for
            H[j,k] = dot(Vc[1:k+1,j], metric[1:k+1,1:k+1], Vc[1:k+1,k+1])
            # w = w - ⟨vⱼ,w⟩⋅vⱼ 
            Vc[1:k+1,k+1] .-= H[j,k] * Vc[1:k+1,j]
        end
        # re-orthogonalize
        for j in 1:k
            # compute ⟨vⱼ,w⟩ for
            hjk = dot(Vc[1:k+1,j], metric[1:k+1,1:k+1], Vc[1:k+1,k+1])
            # w = w - ⟨vⱼ,w⟩⋅vⱼ 
            Vc[1:k+1,k+1] .-= hjk * Vc[1:k+1,j]
        end
        # symmetrize metric
        metric[1:k+1,1:k+1] .= 0.5 * ( metric[1:k+1,1:k+1] + metric[1:k+1,1:k+1]' )
        hsq = dot(Vc[1:k+1,k+1], metric[1:k+1,1:k+1], Vc[1:k+1,k+1])
        H[k+1,k] = sqrt(hsq)

        if H[k+1,k] ≈ 0
            y = H[1:k+1, 1:k] \ g[1:k+1]
            xk = x0
            @time for j in k:-1:1
                xk = BubbleTeaCI.add(xk, sum(y .* Vc[j,1:k]) * Ars[j]; add_kwargs...)
            end
            howverbose > 0 && printstyled("... done\n"; color=:blue, bold=true)
            return xk, true, relres
        end

        # normalize
        Vc[1:k+1,k+1] ./= H[k+1,k]

        printstyled("    GMRES: Step $(k)\n", color=:blue, bold=true)
        display(round.(metric[1:k+1,1:k+1]; digits=8))
        display(round.(Vc[1:k+1,1:k+1]; digits=8))
        display((round.(Vc' * metric * Vc; digits=8))[1:k+1,1:k+1])
        println()

        # Solve least squares min ||g - H*y|| with QR or backsubstitution
        y = H[1:k+1, 1:k] \ g[1:k+1]

        # Check convergence
        relres = norm(g[1:k+1] - H[1:k+1, 1:k] * y) / nb  # Check residual norm
        howverbose > 1 && println("    GMRES Iteration $k: relative residual = $relres")
        if relres < tol
            howverbose > 0 && printstyled("Converged in $k iterations with relative residual $relres. Building solution vector...\n"; color=:blue, bold=true)
            # Build solution approximation x_k = x₀ + ∑ₖ yₖ⋅vₖ
            xk = x0
            @time for j in k:-1:1
                xk = BubbleTeaCI.add(xk, sum(y .* Vc[j,1:k]) * Ars[j]; add_kwargs...)
            end
            howverbose > 0 && printstyled("... done\n"; color=:blue, bold=true)

            return xk, true, relres
        end

    end

    return xk, false, relres  # Did not converge
end

function gmres_restart(A, b; x0=zero(b), m=30, maxiter=1000, tol=1e-6, howverbose=1)
    x = x0
    for iter in 1:div(maxiter, m)
        r = b - A(x)
        if norm(r) / norm(b) < tol
            return x, true, norm(r) / norm(b)
        end
        # call gmres with residual r
        dx, flag, relres = gmres(A, r; x0=zero(r), maxiter=m, tol=1.e-2, howverbose=howverbose)
        x += dx
    end
    return x, false, norm(b - A(x)) / norm(b)
end

# this does not converge very well
function test_gmres_restart()
    for N in 10:50:1000
        println("Testing restarted GMRES with N = $N")
        Amat = randn(N,N)
        sol = randn(N)
        b = myVec(Amat*sol)

        A = (v::myVec) -> myVec(Amat * v.data)  # Define A as a function

        x0 = myVec(zeros(N))  # Initial guess as myVec
        sol_test, conv, _ = gmres_restart(A, b; x0=x0, m=30, maxiter=1000, tol=1e-6, howverbose=1)
        @assert conv
        @assert norm(sol_test.data - sol) / norm(b) < 1.e-6 # Check the solution
    end
end

function test_gmres_lazysum()
    rng = MersenneTwister(134)
    for N in 1000:1000
        Amat = I(N) .- 0.01*randn(rng,N,N)
        sol = randn(rng,N)
        b = myVec(Amat*sol)

        A = (v::myVec) -> myVec(Amat * v.data)  # Define A as a function

        x0 = myVec(zeros(N))  # Initial guess as myVec
        sol_test, conv, _ = gmres_lazysum(A, b, x0; ValueType=Float64, maxiter=N, tol=1e-6, howverbose=2)
        @show conv
        @show norm(sol_test.data - sol) / norm(b)
    end
end
=#