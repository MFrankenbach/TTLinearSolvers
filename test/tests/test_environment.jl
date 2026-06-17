import LinearSolvers: make_environments, make_environment_at

@testset "Environment bulding" begin
    function test_make_environments(;ValueType=ComplexF64, len=4, sitedim=2, linkdims=3)
        rng = MersenneTwister(101)
        sites = [Index(sitedim, "Site $(i)") for i in 1:len]
        sitesp = prime.(sites)
        allsites = vcat([[sites[i], sitesp[i]] for i in 1:len]...)
        # make MPO
        mps = random_mps(rng, ValueType, allsites; linkdims=linkdims)
        A = MPO([mps[2(i-1)+1]*mps[2i] for i in 1:len])

        # make x and b
        x = random_mps(rng, ValueType, sites; linkdims=linkdims)
        b = random_mps(rng, ValueType, sites; linkdims=linkdims)

        b_envs_L = make_environments(x, b, len, false)
        b_envs_R = make_environments(x, b, 1, true)
        @test length(b_envs_L) == len+1
        @test length(b_envs_R) == len+1
        bx = inner(x, b)
        for i in 1:len+1
            @test abs(bx - scalar(b_envs_L[i]*b_envs_R[i])) / abs(bx) < 1.e-14
        end

        A_envs_L = make_environments(x, A, len, false)
        A_envs_R = make_environments(x, A, 1, true)
        @test length(A_envs_L) == len+1
        @test length(A_envs_R) == len+1
        xAx = inner(prime(x), A, x)
        for i in 1:len+1
            @test abs(xAx - scalar(A_envs_L[i]*A_envs_R[i])) / abs(xAx) < 1.e-14
        end
    end

    function test_make_environment_at(;ValueType=ComplexF64, len=4, sitedim=2, linkdims=3)
        rng = MersenneTwister(101)
        sites = [Index(sitedim, "Site $(i)") for i in 1:len]
        x = random_mps(rng, ValueType, sites; linkdims=linkdims)
        b = random_mps(rng, ValueType, sites; linkdims=linkdims)

        all_envs_R = make_environments(x, b, len, false)
        all_envs_L = make_environments(x, b, 1, true)
        for i in eachindex(all_envs_R)
            @assert norm(make_environment_at(x, b, i-1, false) - all_envs_R[i]) < 1.e-14
            @assert norm(make_environment_at(x, b, i, true) - all_envs_L[i]) < 1.e-14
        end
    end


    test_make_environments()
    test_make_environments(;len=2, ValueType=Float64)
    test_make_environments(;len=7, ValueType=Float64)
    test_make_environment_at()
end