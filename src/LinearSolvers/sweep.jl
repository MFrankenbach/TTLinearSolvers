"""
Precompute those environments for a half-sweep that will not change during the sweep.
'RHS' environments are those for ⟨x,b⟩, 'LHS' those for ⟨x,Ax⟩.
"""
function prepare_environments(
    A::MPO,
    x::MPS,
    b::MPS,
    nsite::Integer,
    leftsweep::Bool
)

    # precompute environments in opposing sweep direction
    pre_envs_LHS = make_environments(x, A, _finalsite_sweep(x, nsite, !leftsweep), !leftsweep)
    pre_envs_RHS = make_environments(x, b, _finalsite_sweep(x, nsite, !leftsweep), !leftsweep)

    # environments to be updated during the sweep
    dyn_env_LHS = init_environment(x)
    dyn_env_RHS = init_environment(x)

    return pre_envs_LHS, pre_envs_RHS, dyn_env_LHS, dyn_env_RHS
end

function select_environments(
    pre_envs::Vector{ITensor},
    dyn_env::ITensor,
    site::Integer,
    leftsweep::Bool
)
    pre_env_act = pre_envs[site]
    env_R = leftsweep ? dyn_env : pre_env_act
    env_L = leftsweep ? pre_env_act : dyn_env
    return env_L, env_R
end

"""
Select left and right environments based on sweep direction.
# Arguments
- pre_envs_LHS/RHS: precomputed environments for ⟨x,Ax⟩/⟨x,b⟩
- dyn_env_LHS/RHS: environments updated during the sweep for ⟨x,Ax⟩/⟨x,b⟩
"""
function select_environments(
    pre_envs_LHS::Vector{ITensor},
    pre_envs_RHS::Vector{ITensor},
    dyn_env_LHS::ITensor,
    dyn_env_RHS::ITensor,
    site::Integer,
    leftsweep::Bool
)

    env_L_LHS, env_R_LHS = select_environments(
        pre_envs_LHS,
        dyn_env_LHS,
        site,
        leftsweep
    )
    env_L_RHS, env_R_RHS = select_environments(
        pre_envs_RHS,
        dyn_env_RHS,
        site,
        leftsweep
    )

    return env_L_LHS, env_R_LHS, env_L_RHS, env_R_RHS
end

"""
Move orthogonality centre from `site` to `site±1` (+ for rightsweep, - for leftsweep)
and get updated environments.
"""
function moveon_sweep!(
    A::MPO,
    x::MPS,
    b::MPS,
    site::Integer,
    dyn_env_LHS::ITensor,
    dyn_env_RHS::ITensor,
)
    # shift orthogonality centre
    dyn_env_LHS = update_environment(dyn_env_LHS, x, A, site)
    dyn_env_RHS = update_environment(dyn_env_RHS, x, b, site)
    return dyn_env_LHS, dyn_env_RHS
end

"""
Perform a half-sweep to improve the solution `x` of the linear system Ax=b.
# Arguments
- `nsite`: number of sites to update simultaneously
- `leftsweep`: direction of the sweep
- `_check_normchange`: if true, check that the norm change reported by localupdate! matches the global norm change
- `_do_check_orthogonality`: if true, check MPS orthogonality after each local update
- `do_amen_update`: if true, perform AMEn update after each local update (only for nsite=1)
- `localupdate_kwargs`: additional keyword arguments passed to `localupdate!`. These are parameters for the local linear solver (CG, GMRES...).
"""
function halfsweep!(
    A::MPO,
    x::MPS,
    b::MPS,
    nsite::Integer=_default_nsite(),
    leftsweep::Bool=false;
    _check_normchange=false,
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

    pre_envs_LHS, pre_envs_RHS, dyn_env_LHS, dyn_env_RHS = prepare_environments(A, x, b, nsite, leftsweep)

    # sweep over sites
    sign = leftsweep ? -1 : 1
    for firstsite_act in _firstsite(x, leftsweep):sign:_finalsite_sweep(x, nsite, leftsweep)+sign

        howverbose>0 && println("Sweeping site $firstsite_act")
        pre_shift = leftsweep ? nsite-1 : 0
        env_L_LHS, env_R_LHS, env_L_RHS, env_R_RHS = select_environments(
            pre_envs_LHS,
            pre_envs_RHS,
            dyn_env_LHS,
            dyn_env_RHS,
            firstsite_act - pre_shift,
            leftsweep
        )

        # update x
        _n = Ref(0.0)
        if _check_normchange
            _norm_before = norm(x)
        end
    _do_check_orthogonality && _check_orthogonality(x, (firstsite_act-sign, firstsite_act+sign*nsite))
        localupdate!(
            env_L_LHS, env_R_LHS,
            env_L_RHS, env_R_RHS,
            A, x, b,
            firstsite_act-pre_shift,
            nsite;
            normchange=_check_normchange ? _n : nothing,
            localupdate_kwargs...
        )

        if _check_normchange
            # check orthogonalization: norm changes should be equal
            _n_global = norm(x) - _norm_before
            @assert isapprox(_n[], _n_global; atol=1e-14, rtol=1e-14) "Norm change in local update $firstsite_act ($(_n[])) does not match global norm change ($(_n_global))."
        end
        howverbose>0 && println(" Completed local update at sites $(firstsite_act)$(ifelse(leftsweep, "<-","->"))$(firstsite_act+sign*(nsite-1)) / $(length(A)).")

        # update environments
        if firstsite_act != _finalsite_sweep(x, nsite, leftsweep)+sign
            if do_amen_update
                howverbose>2 && println("Bond dimensions before AMEn update $(ITensorMPS.linkdims(x)).")
                amen_update_1site!(
                    A, x, b,
                    firstsite_act,
                    !leftsweep;
                    amen_kwargs...
                )
                howverbose>2 && println("Bond dimensions after AMEn update $(ITensorMPS.linkdims(x)).")
            end
            orthogonalize!(x, firstsite_act+sign)
            dyn_env_LHS, dyn_env_RHS = moveon_sweep!(
                A, x, b,
                firstsite_act,
                dyn_env_LHS,
                dyn_env_RHS,
            )
        end
    end
end

function _check_orthogonality(x::MPS, limits::Tuple{Int,Int})
    ll, lr = extrema(limits)
    xlp = prime(linkinds, x)
    envs_L = make_environments(x, xlp, ll, false)
    envs_R = make_environments(x, xlp, lr, true)
    for i in 2:length(envs_L)
        n_ = norm(envs_L[i] - unity(linkind(x, i-1)))
        n_ < 1.e-13 ||
            error("MPS not orthogonal on site $(i-1) (error: $(n_)).")
    end
    for i in 1:length(envs_R)-1
        il = length(x)-length(envs_R)+i
        norm(envs_R[i] - unity(linkind(x, il))) < 1.e-14 ||
            error("MPS not orthogonal on site $(i-1).")
    end
    return true
end

_default_amen_kwargs() = Dict(
    :truncate_cutoff => DEFAULT_CUTOFF,
    :truncate_maxdim => typemax(Int),
    :apply_kwargs => _default_amen_residual_apply_kwargs()
)