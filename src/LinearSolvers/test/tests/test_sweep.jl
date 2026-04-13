import BubbleTeaCI.LinearSolvers: halfsweep!, make_Heisenberg, relative_error

@testset "sweeping for linear solve" begin

    function test_sweep(;
        len=8,
        linkdims=3,
        nsite=2,
        nsweeps=2,
        ValueType=ComplexF64,
        gmres_tol=1.e-6,
        _check_normchange=false,
        )

        rng = MersenneTwister(110)
        A, sites = make_Heisenberg(;len=len)
        cutoff = 1.e-12
        apply_kwargs = (; alg="fit", cutoff=cutoff, nsweeps=5)

        # make x and b
        x = random_mps(rng, ValueType, sites; linkdims=linkdims)
            # A⋅sol = b
        sol = random_mps(rng, ValueType, sites; linkdims=linkdims)
        b = apply(A, sol; apply_kwargs...)

        for _ in 1:nsweeps
            halfsweep!(A, x, b, nsite, false; tol=gmres_tol, maxiter=300, cutoff=cutoff, _check_normchange=_check_normchange, _do_check_orthogonality=true)
            halfsweep!(A, x, b, nsite, true; tol=gmres_tol, maxiter=300, cutoff=cutoff, _check_normchange=_check_normchange, _do_check_orthogonality=true)
        end
        error = relative_error(ITensorMPS.apply(A, x; apply_kwargs...), b)

        @test error <  5*gmres_tol
        @test relative_error(x, sol) < 5*gmres_tol
    end

    test_sweep(; gmres_tol=1.e-8, _check_normchange=true)
    test_sweep(; gmres_tol=1.e-8, ValueType=Float64)
    test_sweep(; len=9, nsite=3, _check_normchange=true)
    test_sweep(; nsite=3, ValueType=Float64)
end