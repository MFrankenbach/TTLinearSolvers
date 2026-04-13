#=
Copied from English Wikipedia article on Conjugate Gradient Method
=#
"""
Return the solution to `A(x) = b` using the conjugate gradient method.
`A` must be a positive linear operator `x↦A(x)`.
`x0` is the initial guess for the solution (default is the zero vector).
Convergence is reached when `||r||/||b|| < tol`, where `r = b - A(x)` is the residual.
# Furhter arguments
- `add_kwargs`: Keyword arguments passed to the `add` function used for vector addition.
- `truncate_cutoff`: Cutoff value used in the `truncate!` function to truncate the search direction vector `p`.
- `maxiter`: Maximum number of iterations to perform.

Returns the approximate solution vector `x`.
"""
function conjugate_gradient(
    A, b, x0;
    howverbose=0,
    tol=1.e-6,
    add_kwargs=Dict(),
    maxiter=100,
    truncate_cutoff=get(add_kwargs, :cutoff, DEFAULT_CUTOFF),
    # TODO do this properly instead of just ignoring keywords that are not used in this function
    kwargs...
)
    x = x0                        # initialize the solution
    r = ls_add(b,-A(x0); add_kwargs...)    # initial residual
    p = r                         # initial search direction
    r²old = dot(r,r)                     # squared norm of residual
    bsq = dot(b,b)

    k = 0
    while (abs(r²old/bsq) > tol^2) && k<maxiter   # iterate until convergence
        Ap = A(p)                      # search direction
        α = r²old / dot(p, Ap)           # step size
        x = ls_add(x, α*p; add_kwargs...)                   # update solution
        # Update residual:
        if (k + 1) % 16 == 0            # every 16 iterations, recompute residual from scratch 
            r = ls_add(b, -A(x); add_kwargs...)             # to avoid accumulation of numerical errors
        else
            r = ls_add(r, -α*Ap; add_kwargs...)              # use the updating formula that saves one matrix-vector product
        end
        r²new = dot(r,r)
        p = ls_add(r, (r²new / r²old) * p; add_kwargs...)  # update search direction
        ls_truncate!(p; cutoff=truncate_cutoff)
        r²old = r²new                   # update squared residual norm
        k += 1
    end
    if abs(r²old/bsq) > tol^2
        @warn "Conjugate gradient did not converge within the maximum number of iterations ($maxiter). Final residual norm: $(sqrt(r²old)) (||b||=$(sqrt(bsq)))."
    end

    howverbose>0 && println("    Conjugate gradient converged in $k iterations with residual norm $(sqrt(r²old)) (||b||=$(sqrt(bsq))).")

    return x
end