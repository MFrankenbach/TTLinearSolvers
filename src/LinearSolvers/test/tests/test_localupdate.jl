import BubbleTeaCI.LinearSolvers: split_x_local

@testset "Local update" begin

    function test_split_x_local()
        rng = MersenneTwister(24)
        the_inds = [Index(3,"i1"), Index(4,"i2"), Index(5,"i3")]
        t = random_itensor(rng, Float64, the_inds)
        t_single = random_itensor(rng, Float64, the_inds[1])

        t_split = split_x_local(t, the_inds, nothing)
        t_split_link = split_x_local(t, the_inds[2:end], the_inds[1])

        @test only(split_x_local(t_single, [the_inds[1]], nothing)) == t_single
        @test length(t_split)==3
        @test length(t_split_link)==2

        @test norm(t - prod(t_split)) < 5e-14
        @test norm(t - prod(t_split_link)) < 5e-14
    end

    test_split_x_local()
end