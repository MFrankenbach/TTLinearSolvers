import TTLinearSolvers: make_Heisenberg, DMRG, variance
using Random, ITensors, ITensorMPS
using Profile, StatProfilerHTML
using Plots

function Heisenberg_dmrg()
    # spin-1/2 Heisenberg chain, J₁=1
    len = 60
    do_pbc = true
    sites = siteinds("S=1/2",len)
    os = OpSum()
    for j=1:len-1
        os += 0.5,"S+",j,"S-",j+1
        os += 0.5,"S-",j,"S+",j+1
        os += "Sz",j,"Sz",j+1
    end
    # PBC
    if do_pbc
        os += 0.5,"S+",len,"S-",1
        os += 0.5,"S-",len,"S+",1
        os += "Sz",len,"Sz",1
    end
    AJ1 = MPO(os,sites)
    # next-nearest-neighbor coupling
    J2 = 0.25
    os2 = OpSum()
    for j in 1:len-2
        os2 += 0.5,"S+",j,"S-",j+2
        os2 += 0.5,"S-",j,"S+",j+2
        os2 += "Sz",j,"Sz",j+2
    end
    # PBC
    if do_pbc
        for j in 1:2
            os2 += 0.5,"S+",len+j-2,"S-",j
            os2 += 0.5,"S-",len+j-2,"S+",j
            os2 += "Sz",len+j-2,"Sz",j
        end
    end
    AJ2 = MPO(os2,sites)
    A = ITensorMPS.add(AJ1,J2*AJ2; alg="directsum")
    ITensorMPS.truncate!(A; cutoff=1.e-20)


    rng = MersenneTwister(42)
    x = random_mps(rng, ComplexF64, sites; linkdims=2)
    nsweeps = 3
    E_hist_dmrg = Float64[]
    _, x_dmrg = DMRG(
        A, x;
        howverbose=1,
        E_history=E_hist_dmrg,
        nsite=2,
        cutoff=1.e-10,
        nsweeps=nsweeps,
        tol=1.e-8,
        maxiter=300
    )
    E_hist_amen = Float64[]
    _, x_amen = DMRG(
        A, x;
        howverbose=1,
        E_history=E_hist_amen, 
        nsite=1,
        cutoff=1.e-10,
        nsweeps=nsweeps,
        do_amen_update=true,
        amen_kwargs=(; apply_kwargs=Dict(:alg=>"fit", :cutoff=>1.e-10, :nsweeps=>1, :maxdim=>20))
    )

    sig_dmrg = variance(A, x_dmrg; cutoff=1.e-7, alg="fit", nsweeps=2)
    printstyled("  DMRG variance (DMRG): $sig_dmrg\n", color=:green)
    sig_amen = variance(A, x_amen; cutoff=1.e-7, alg="fit", nsweeps=2)
    printstyled("  amen variance (AMEN): $sig_amen\n", color=:green)

    plot(eachindex(E_hist_dmrg), E_hist_dmrg, label="DMRG", xlabel="Half-sweeps", ylabel="Energy", legend=:topleft)
    plot!(eachindex(E_hist_amen), E_hist_amen, label="amen", xlabel="Half-sweeps", ylabel="Energy", legend=:topleft)
end
