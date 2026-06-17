#=
Ground-state search with DMRG / AMEn idea.
=#

function variance(A::MPO, x::MPS; apply_kwargs...)
    Ax = apply(A, x; apply_kwargs...)
    ex = inner(x, Ax)
    ex2 = inner(Ax, Ax)
    var = ex2 - ex^2
    return var
end

"""
Solve local ground-state problem `(A - E I) x = 0` and modify MPS `x` in-place.
"""
function localupdate!(
    env_L::ITensor,
    env_R::ITensor,
    A::MPO,
    x::MPS,
    firstsite::Integer,
    nsite::Integer=_default_nsite();
    howverbose=0,
    maxdim::Integer=typemax(Int),
    cutoff=nothing,
    kwargs...
    )

    local_x = local_x_dense(x, firstsite, nsite)
    x_inds = inds(local_x)
    local_x_arr = array(local_x, x_inds...)
    function _apply_A(x_loc_arr)
        x_loc = ITensor(x_loc_arr, x_inds...)
        return array(apply_Aloc_dense(env_L, A, env_R, x_loc, firstsite, nsite), x_inds...)
    end

    # Krylov ground-state search
    E, x_loc_new_arr = lanczos(
        _apply_A,
        local_x_arr;
        howverbose=howverbose,
        kwargs...
    )
    x_loc_new = ITensor(x_loc_new_arr, x_inds...)

    # update solution vector
    siteinds_x = siteinds(x)
    locsites = _locsites(x, firstsite, nsite)
    x.data[locsites] .= split_x_local(
        x_loc_new,
        siteinds_x[locsites],
        _leftlink(x, firstsite);
        maxdim=maxdim,
        cutoff=cutoff
    )
    x.llim=locsites[end]-1
    x.rlim=locsites[end]+1

    return x_loc_new, E
end

"""
Compute `r = (Ax - E x)`, the residual of the groundstate problem.
Can be truncated to `maxdim`.
"""
function energy_residual(A::MPO, x::MPS, E::Real; maxdim=typemax(Int), apply_kwargs...)
    Ax = apply(A, x; apply_kwargs...)
    r = ITensorMPS.add(Ax, -E*x; alg="directsum")
    ITensorMPS.truncate!(r; maxdim=maxdim)
    return r
end

"""
Perform left- or rightsweep of DMRG ground-state optimization on MPS `x` with MPO `A`.
Return the final energy.
# Arguments
- `nsite`: number of sites to optimize simultaneously
- `leftsweep`: sweep direction
- `_do_check_orthogonality`: whether to check orthogonality of MPS during the sweep
"""
function dmrg_halfsweep!(
    A::MPO,
    x::MPS,
    nsite::Integer=_default_nsite(),
    leftsweep::Bool=false;
    _do_check_orthogonality=false,
    do_amen_update=false,
    amen_kwargs=_default_amen_kwargs(),
    howverbose=0,
    localupdate_kwargs...
)
    if do_amen_update && nsite != 1
        error("AMEn update is only implemented for nsite=1.")
    end
    # bring MPS to correct orthogonality centre
    orthogonalize!(x, _firstsite(x, leftsweep))

    pre_envs = make_environments(x, A, _finalsite_sweep(x, nsite, !leftsweep), !leftsweep)
    dyn_env = init_environment(x)

    E = typemax(Float64)

    # sweep over sites
    sign = leftsweep ? -1 : 1
    for firstsite_act in _firstsite(x, leftsweep):sign:_finalsite_sweep(x, nsite, leftsweep)+sign

        howverbose>1 && println("Sweeping site $firstsite_act")
        pre_shift = leftsweep ? nsite-1 : 0
        env_L, env_R= select_environments(
            pre_envs,
            dyn_env,
            firstsite_act - pre_shift,
            leftsweep
        )

        # update x
        _do_check_orthogonality && _check_orthogonality(x, (firstsite_act-sign, firstsite_act+sign*nsite))

        _, E = localupdate!(
            env_L, env_R,
            A, x,
            firstsite_act-pre_shift,
            nsite;
            localupdate_kwargs...
        )

        howverbose>1 && println("  Completed local update at sites $(firstsite_act)$(ifelse(leftsweep, "<-","->"))$(firstsite_act+sign*(nsite-1)) / $(length(A)).")
        howverbose>1 && println("  Current energy: $E")

        # update environments
        if firstsite_act != _finalsite_sweep(x, nsite, leftsweep)+sign
            if do_amen_update
                howverbose>2 && println("Bond dimensions before AMEn update $(ITensorMPS.linkdims(x)).")
                amen_update_1site_dmrg!(
                    A, x, E,
                    firstsite_act,
                    !leftsweep;
                    amen_kwargs...
                )
                howverbose>2 && println("Bond dimensions after AMEn update $(ITensorMPS.linkdims(x)).")
            end
            orthogonalize!(x, firstsite_act+sign)
            dyn_env = update_environment(dyn_env, x, A, firstsite_act)
        end
    end

    return E
end

"""
Perform DMRG ground-state optimization on MPS `x0` with MPO `A`.
Can pass an optional vector `E_history` to store energy at each half-sweep. It is modified **in-place**.
"""
function DMRG(
    A::MPO,
    x0::MPS;
    E_history::Union{Nothing,Vector{Float64}}=nothing,
    nsite::Integer=_default_nsite(),
    nsweeps::Integer=3,
    leftsweep::Bool=false,
    howverbose=0,
    kwargs...
)
    x = deepcopy(x0)
    E = typemax(Float64)
    for sweep in 1:2nsweeps
        howverbose>0 && println("Starting DMRG half-sweep $sweep / $(2nsweeps)")
        E = dmrg_halfsweep!(
            A, x,
            nsite,
            leftsweep;
            howverbose=howverbose,
            kwargs...
        )
        !isnothing(E_history) && push!(E_history, E)
        howverbose>0 && println("Completed half-sweep. Energy: $E")
        leftsweep = !leftsweep
    end
    return E, x
end

function test_dmrg_Heisenberg(; nsite=2, kwargs...)
    rng = MersenneTwister(42)
    ITensors.disable_warn_order()
    for len in 5:10
        H, sites = make_Heisenberg(; len=len, pos_shift=0.0)
        Hmat = Hermitian(to_matrix(H, sites))
        E_exact = eigmin(Hmat)
        x0 = random_mps(rng, ComplexF64, sites; linkdims=2)
        nsweeps=3
        E = dmrg(H, x0; nsite=nsite, nsweeps=nsweeps, howverbose=0, _do_check_orthogonality=true, kwargs...)
        @assert abs(E - E_exact) < 1e-11
    end
    ITensors.set_warn_order(14)
end