import BubbleTeaCI.LinearSolvers: halfsweep!, make_Heisenberg, relative_error, AdagA
using Random, ITensors, ITensorMPS
using Profile, StatProfilerHTML

function test_sweep_vs_ITensorMPS(;len=7, linkdims=3, nsite=2, ValueType=ComplexF64)

    rng = MersenneTwister(110)
    A, sites = make_Heisenberg(;len=len)

    # make x and b
    x = random_mps(rng, ValueType, sites; linkdims=linkdims)
    x0 = deepcopy(x)
    b = random_mps(rng, ValueType, sites; linkdims=linkdims)

    cutoff = 1.e-12
    apply_kwargs = (; alg="fit", cutoff=cutoff, nsweeps=5)
    gmres_tol = 1.e-6
    nsweeps = 2
    norm_before = relative_error(ITensorMPS.apply(A, x; apply_kwargs...), b)
    printstyled("Error before solve: $norm_before\n"; color=:blue)
    @time for _ in 1:nsweeps
        halfsweep!(A, x, b, nsite, false; tol=gmres_tol, maxiter=300, cutoff=cutoff)
        halfsweep!(A, x, b, nsite, true; tol=gmres_tol, maxiter=300, cutoff=cutoff)
    end
    norm_after = relative_error(ITensorMPS.apply(A, x; apply_kwargs...), b)
    printstyled("Error after my solve: $norm_after\n"; color=:blue)

    @time x_itensor = linsolve(
        A, b, x0;
        cutoff=get(apply_kwargs, :cutoff, cutoff),
        updater_kwargs=(; ishermitian=true, tol=gmres_tol, maxiter=100, krylovdim=300),
        nsweeps=nsweeps,
        outputlevel=1
        )
    norm_after_itensor = relative_error(ITensorMPS.apply(A, x_itensor; apply_kwargs...), b)
    printstyled("Norm after ITensorMPS solve: $norm_after_itensor\n"; color=:blue)
end


function profile_halfsweep(;len=7, linkdims=3, nsite=2, ValueType=ComplexF64)

    rng = MersenneTwister(110)
    A, sites = make_Heisenberg(;len=len)

    # make x and b
    x = random_mps(rng, ValueType, sites; linkdims=linkdims)
    b = random_mps(rng, ValueType, sites; linkdims=linkdims)

    cutoff = 1.e-12
    gmres_tol = 1.e-6
    Profile.clear()
    @profile halfsweep!(A, x, b, nsite, false; tol=gmres_tol, maxiter=300, cutoff=cutoff)
    statprofilehtml()
end


function profile_ITensor_linsolve(;len=7, linkdims=3, nsite=2, ValueType=ComplexF64)

    rng = MersenneTwister(110)
    A, sites = make_Heisenberg(;len=len)

    # make x and b
    x = random_mps(rng, ValueType, sites; linkdims=linkdims)
    b = random_mps(rng, ValueType, sites; linkdims=linkdims)

    cutoff = 1.e-12
    gmres_tol = 1.e-6
    Profile.clear()
    @profile begin
    _ = linsolve(
        A, b, x0;
        cutoff=cutoff,
        updater_kwargs=(; ishermitian=true, tol=gmres_tol, maxiter=100, krylovdim=300),
        nsweeps=1,
        outputlevel=1
        )
    end
    statprofilehtml()
end

"""
Try out MALS/AMEn for BSE.
"""
function test_sweepsolve_gmres_BSE(;
    method=:mals,
    combinesites=nothing,
    make_hermitian=false,
    subtract_Fsub=false,
    local_solver_amen=:conjugate_gradient,
    )
    grid = BubbleTeaCI.make_test_grid_5D()
    _U = 0.1
    f(qx,qy,om,nu,nup) = 0.5/(1.0 + (nu-om)^2 + (om+nu+nup)^2 + nup^2) * 1/(2qx^2 + 4qy^2+0.3) + im*0.3/(1.0 + (om+nu+2nup)^2 + nup^2) * 1/(qx^2 + 4qy^2+0.3) + _U
    chi(qx,qy,om,nu) = 0.5/(3.0 + (nu+om)^2 + (om+2nu)^2 + (nu-0.5om)^2) * 1/(2qx^4 + (qy+2qx)^2 + 1) + im*0.2/(2.0 + (nu-om)^2 + (om+3nu)^2) * 1/(qx^2 + (qy-qx)^2 + 1)
    U(qx,qy,om,nu,nup) = _U

    grid_chi = BubbleTeaCI.project_grid(grid, grid.variablenames[[5]])
    F_loc = BubbleTeaCI.compress_to_TTFunction(ComplexF64, grid, f; Dnew=4, tolerance=1.e-6)
    Fsub = BubbleTeaCI.compress_to_TTFunction(ComplexF64, grid, U; Dnew=4, tolerance=1.e-6)
    chi_nonloc = BubbleTeaCI.compress_to_TTFunction(ComplexF64, grid_chi, chi; Dnew=4, tolerance=1.e-6)
    bcs = BubbleTeaCI._make_BasicContractOrders_BSE(
        F_loc, chi_nonloc, F_loc;
        contract_legsL=[3,4],
        contract_legsLchi=[2,1],
        contract_legsRchi=[4,3],
        contract_legsR=[1,2],
        Nq=3,
        Nk1=1
    )

    solver_gmres = BubbleTeaCI.BSESolver(F_loc, chi_nonloc, bcs; F_sub=Fsub, convergence_threshold=1.e-4)

    solver_mals = deepcopy(solver_gmres)
    contract_kwargs = Dict(:alg=>"fit", :cutoff=>1.e-8, :times_dV=>true)
    BubbleTeaCI.gmres!(solver_gmres; contract_kwargs=contract_kwargs, howverbose=0, tol=1.e-6)
    error_before = norm(solver_mals.F - (solver_mals.prefacs[1]*F_loc + solver_mals.prefacs[2]*BubbleTeaCI.BSE(solver_mals.F, chi_nonloc, F_loc, bcs...; contract_kwargs...))) / norm(solver_mals.F)
    BubbleTeaCI.sweepsolve!(solver_mals;
        method=method,
        contract_kwargs=contract_kwargs,
        times_dV=true,
        howverbose=3,
        mals_tol=1.e-12,
        gmres_tol=1.e-6,
        check_residual=false,
        make_hermitian=make_hermitian,
        subtract_Fsub=subtract_Fsub,
        combinesites=combinesites,
        updater_kwargs=(; ishermitian=false, tol=1.e-10, maxiter=10, krylovdim=50),
        gmres_maxiter=300,
        local_solver_amen=local_solver_amen,
        nsweeps=2,
        amen_kwargs=(;
            truncate_cutoff=1.e-20,
            cutoff=1.e-6,
            maxdim=10,
            alg="fit",
            nsweeps=2
        )
    )

    F_gmres = solver_gmres.F
    F_mals = solver_mals.F
    println("Error before MALS/AMEn solve: $error_before")
    # more accurate
    contract_kwargs = Dict(:alg=>"fit", :cutoff=>1.e-25, :times_dV=>true)
    @show norm(F_gmres - F_mals)/norm(F_mals)
    @show norm(F_gmres - (solver_gmres.prefacs[1]*F_loc + solver_gmres.prefacs[2]*BubbleTeaCI.BSE(F_gmres, chi_nonloc, F_loc, bcs...; contract_kwargs...))) / norm(F_gmres)
    @show norm(F_mals - (solver_mals.prefacs[1]*F_loc + solver_mals.prefacs[2]*BubbleTeaCI.BSE(F_mals, chi_nonloc, F_loc, bcs...; contract_kwargs...))) / norm(F_mals)
end

test_sweepsolve_gmres_BSE(
    ;method=:amen,
    make_hermitian=false,
    local_solver_amen=:gmres
    )