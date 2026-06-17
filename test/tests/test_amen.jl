using TTLinearSolvers:
    expand_bond!,
    kept_space_projector,
    discarded_space_projector,
    unity,
    amen!,
    relative_error,
    make_Heisenberg

@testset "AMEn building blocks" begin

    function test_expand_bond()
        rng = MersenneTwister(123)
        sites = [Index(2), Index(3), Index(2), Index(4), Index(4)]
        ld_x = [2,5,5,4]
        ld_r = [2,2,3,2]
        x_ = random_mps(rng, ComplexF64, sites; linkdims=ld_x)
        r_ = random_mps(rng, ComplexF64, sites; linkdims=ld_r)

        # for rightsweep
        for left in (true,false)
            for target_site in (left ? 1 : 2):(left ? length(sites)-1 : length(sites))
                x = deepcopy(x_)
                r = deepcopy(r_)
                nextsite = target_site+(left ? 1 : -1)
                orthogonalize!(x, nextsite)
                orthogonalize!(r, nextsite)
                expand_bond!(x, r, target_site, left)
                # check orthogonality
                @test abs(norm(x) - norm(x[nextsite])) < 1.e-14
                x_dense = prod(x.data)
                @test abs(norm(x_dense) - norm(x[nextsite])) < 1.e-14
                # index integrity
                @test length(inds(x_dense))==length(x)
            end
        end
    end

    function test_kept_discarded_space_projector()
        rng = MersenneTwister(123)
        sites = [Index(2), Index(3), Index(2), Index(4)]
        x = random_mps(rng, ComplexF64, sites; linkdims=4)
        orthogonalize!(x, 3)
        # left
        p_disc = discarded_space_projector(x, 2, true)
        p_kept, outer_sites = kept_space_projector(x, 2, true)
        for p in (p_disc, p_kept)
            @test hasinds(p, siteind(x,2), prime(siteind(x,2)))
            @test hasinds(p, linkind(x,1), prime(linkind(x,1)))
        end
        @test abs(scalar(p_disc*dag(p_kept))) < 1.e-14
        @test norm(p_disc + p_kept - unity(outer_sites)) < 1.e-14
        # right
        orthogonalize!(x, 1)
        p_disc = discarded_space_projector(x, 2, false)
        p_kept, outer_sites = kept_space_projector(x, 2, false)
        for p in (p_disc, p_kept)
            @test hasinds(p, siteind(x,2), prime(siteind(x,2)))
            @test hasinds(p, linkind(x,2), prime(linkind(x,2)))
        end
        @test abs(scalar(p_disc*dag(p_kept))) < 1.e-14
        @test norm(p_disc + p_kept - unity(outer_sites)) < 1.e-14
    end


    test_expand_bond()
    test_kept_discarded_space_projector()
end

@testset "AMEn sweep" begin
    function test_sweep_amen(;
        len=8,
        linkdims=3,
        nsweeps=2,
        ValueType=ComplexF64,
        gmres_tol=1.e-6,
        _check_normchange=true,
        truncate_cutoff=nothing,
        solver=:gmres,
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

        amen!(
            A, x, b;
            nsweeps=nsweeps,
            _check_normchange=_check_normchange,
            _do_check_orthogonality=true,
            amen_kwargs=Dict(:truncate_cutoff=>truncate_cutoff, :apply_kwargs=>apply_kwargs),
            # arguments for local update
            solver=solver,
            tol=gmres_tol,
            maxiter=300,
            cutoff=cutoff,
        )
        error = relative_error(ITensorMPS.apply(A, x; apply_kwargs...), b)

        @test error <  2*gmres_tol
        # no error here can be amplified depending on condition number of A
        @test relative_error(x, sol) < 5*gmres_tol
    end

    test_sweep_amen(; len=8, gmres_tol=1.e-5, ValueType=Float64)
    test_sweep_amen(; len=8, gmres_tol=1.e-6)
    test_sweep_amen(; len=8, gmres_tol=1.e-6, solver=:conjugate_gradient)
    test_sweep_amen(; len=8, gmres_tol=1.e-6, truncate_cutoff=1.e-20) # with SVD
    test_sweep_amen(; len=8, gmres_tol=1.e-7)
    test_sweep_amen(; len=6, linkdims=[2,4,8,4,2], gmres_tol=1.e-7)
    test_sweep_amen(; len=6, linkdims=[2,4,8,4,2], gmres_tol=1.e-7, truncate_cutoff=1.e-20) # with SVD
end
