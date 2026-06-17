import LinearSolvers: AdagA, identity_MPO, unity, add_identity, truncate_upto!

@testset "utilities" begin

    function test_AdagA()
        rng = MersenneTwister(101)
        sites = [Index(2, "Site $(i)") for i in 1:5]
        sitesp = prime.(sites)
        allsites = vcat([[sites[i], sitesp[i]] for i in 1:5]...)
        mps = random_mps(rng, Float64, allsites; linkdims=3)
        A = MPO([mps[2(i-1)+1]*mps[2i] for i in 1:5])

        A_mat = prod(A.data)
        Ada_mat_ref = replaceprime(conj(replaceprime(A_mat, 0=>2)) * A_mat, 2=>1)
        AdA = AdagA(A)
        Ada_mat = prod(AdA.data)

        @test norm(Ada_mat - Ada_mat_ref) < 1e-12
    end

    function test_identity_MPO()
        rng = MersenneTwister(63)
        R = 4
        localdims = [2,2,3,2]
        sites = [Index(localdims[i], "s$(i)") for i in 1:R]
        sites_mpo = vcat([[sites[i], prime(sites[i])] for i in 1:R]...)
        mpo = identity_MPO(Float64, R; localdims=localdims, sites=sites_mpo)
        mps = random_mps(rng, Float64, sites; linkdims=3)
        res = apply(mpo, mps; cutoff=1.e-20)
        @test norm(res - mps) < 1.e-10
    end

    function test_add_identity()
        rng = MersenneTwister(63)
        R = 4
        localdims = [2,2,3,2]
        sites = [Index(localdims[i], "s$(i)") for i in 1:R]
        sites_mpo = vcat([[sites[i], prime(sites[i])] for i in 1:R]...)
        id = identity_MPO(Float64, R; localdims=localdims, sites=sites_mpo)
        mps = random_mps(rng, Float64, sites_mpo; linkdims=3)
        A = MPO([mps[2(i-1)+1]*mps[2i] for i in 1:R])

        A23 = add_identity(A, 2.0, 3.0)
        @test norm(A23 - (3.0*A + 2.0*id)) < 1.e-12
    end

    function test_unity()
        i = Index(3)
        j = Index(2)
        u = unity(i, j)
        u_arr = array(u,i,j,prime(i),prime(j))
        @test norm(u_arr)^2 ≈ dim(i)*dim(j)
        @test norm(u_arr.*u_arr)^2 ≈ dim(i)*dim(j)
    end

    function test_truncate_upto!()
        R = 6
        mps = random_mps(MersenneTwister(1234), Float64, [Index(2, "s$i") for i in 1:R]; linkdims=10)
        orthogonalize!(mps, 2)
        mps_trunc = truncate_upto!(deepcopy(mps), 1, 3; cutoff=1.e-20)
        truncate_upto!(mps_trunc, 6, 3; cutoff=1.e-20)
        @test norm(mps_trunc - mps) < 1.e-14
        @test abs(norm(mps_trunc[3]) - norm(mps_trunc)) < 1.e-14
    end

    test_AdagA()
    test_identity_MPO()
    test_unity()
    test_add_identity()
    test_truncate_upto!()
end