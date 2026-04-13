import BubbleTeaCI.LinearSolvers: lanczos, to_matrix, DMRG

@testset "Lanczos" begin
    function test_lanczos()
        rng = MersenneTwister(46)
        for N in 20:30:200
            A = randn(rng, ComplexF64, N,N)
            A = (A + A') / 2  # make Hermitian
            H(x) = A * x
            v0 = randn(rng, ComplexF64, N)
            eigval, eigvec = eigen(Hermitian(A))
            E, v = lanczos(H, v0; maxiter=N, tol=1e-8, howverbose=0)
            u = eigvec[:,1]
            @test abs(E - eigval[1]) < 1.e-8
            @test sqrt(2-2*abs(dot(u,v))) < 5.e-5
        end
    end
    
    test_lanczos()
end

@testset "DMRG@Heisenberg" begin
    function test_dmrg_Heisenberg(; nsite=2, kwargs...)
        rng = MersenneTwister(42)
        ITensors.disable_warn_order()
        for len in 5:10
            H, sites = make_Heisenberg(; len=len, pos_shift=0.0)
            Hmat = Hermitian(to_matrix(H, sites))
            E_exact = eigmin(Hmat)
            x0 = random_mps(rng, ComplexF64, sites; linkdims=2)
            nsweeps=3
            E,_ = DMRG(H, x0; nsite=nsite, nsweeps=nsweeps, howverbose=0, _do_check_orthogonality=true, kwargs...)
            @test abs(E - E_exact) < 1e-11
        end
        ITensors.set_warn_order(14)
    end

    test_dmrg_Heisenberg(; nsite=2, maxiter=100, tol=1.e-8)
    test_dmrg_Heisenberg(; nsite=3, maxiter=100, tol=1.e-8)
    test_dmrg_Heisenberg(; nsite=4, maxiter=100, tol=1.e-8)
    # with AMEn
    test_dmrg_Heisenberg(;
        nsite=1, maxiter=100, tol=1.e-8,
        do_amen_update=true,
        amen_kwargs=(; apply_kwargs=Dict(:alg=>"fit", :cutoff=>1.e-10, :nsweeps=>2, :maxdim=>40))
        )
end