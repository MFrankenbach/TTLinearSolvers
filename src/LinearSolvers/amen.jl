"""
`r = Ax - b`
Can be truncated to `maxdim`.
"""
function residual(A::MPO, x::MPS, b::MPS; maxdim=typemax(Int), apply_kwargs...)::MPS
    Ax = apply(A, x; apply_kwargs...)
    r = ITensorMPS.add(Ax, -b; alg="directsum")
    ITensorMPS.truncate!(r; maxdim=maxdim)
    return r
end

"""
`e = ⟨x,Ax⟩ - 2Re⟨x,b⟩`
"""
function energy(A::MPO, x::MPS, b::MPS; apply_kwargs...)
    return inner(x, apply(A, x; apply_kwargs...)) - 2*real(inner(x, b))
end

"""
Update a ground state MPS `x` according to 'eigenvalue residual' `r=(Ax - E x)` at `site`.
"""
function amen_update_1site_dmrg!(
    A::MPO,
    x::MPS,
    E::Real,
    site::Integer,
    left::Bool;
    truncate_cutoff=DEFAULT_CUTOFF,
    truncate_maxdim=typemax(Int),
    apply_kwargs=_default_amen_residual_apply_kwargs()
)
    r = energy_residual(A, x, E; maxdim=truncate_maxdim, apply_kwargs...)
    amen_expand_bond!(
        x,
        r,
        site,
        left;
        truncate_cutoff=truncate_cutoff,
        truncate_maxdim=truncate_maxdim
    )
end

function amen_update_1site!(
    A::MPO,
    x::MPS,
    b::MPS,
    site::Integer,
    left::Bool;
    truncate_cutoff=DEFAULT_CUTOFF,
    truncate_maxdim=typemax(Int),
    apply_kwargs=_default_amen_residual_apply_kwargs()
)
    # println("AMEn update at site $site, left=$left, apply_kwargs:$(apply_kwargs), truncate_maxdim=$(truncate_maxdim)"); flush(stdout)
    r = residual(A, x, b; apply_kwargs...)
    # println("  Residual norm: $(norm(r))"); flush(stdout)
    # println("AMEn update done"); flush(stdout)
    amen_expand_bond!(
        x,
        r,
        site,
        left;
        truncate_cutoff=truncate_cutoff,
        truncate_maxdim=truncate_maxdim
    )
end

"""
Expand bond of `x` after (depending on sweep direction) `site` based on residual `r`.
"""
function amen_expand_bond!(
    x::MPS,
    r::MPS,
    site::Integer,
    left::Bool;
    truncate_cutoff=DEFAULT_CUTOFF,
    truncate_maxdim=typemax(Int),
)
    nextsite = site + (left ? 1 : -1)
    orthogonalize!(r, nextsite)
    truncate_bond!(x, site, !left; cutoff=truncate_cutoff, maxdim=truncate_maxdim)
    orthogonalize!(x, nextsite)
    expand_bond!(x, r, site, left; cutoff=truncate_cutoff)
end

"""
Compute the projector onto the discarded space at `site` for an MPS with orthogonality center at `site`.
For `left=true`:
```
  s
  |
 /|
l--□-∖
     ∣
l'-□-/   
 ∖|
  |
  s'
```
"""
function discarded_space_projector(
    x::MPS,
    site::Integer,
    left::Bool
)
    QQp, outer_inds = kept_space_projector(x, site, left)
    unity_t = unity(outer_inds...)
    return unity_t - QQp
end

"""
Compute the projector onto the kept space at `site` for an MPS with orthogonality center at `site`. 
For `left=true`:
```
  s
  |
 /|
l--□-∖
     ∣
l'-□-/ 
 ∖|
  |
  s'
```
"""
function kept_space_projector(
    x::MPS,
    site::Integer,
    left::Bool
)
    _orthocenter_check(x, site, left)
    nextsite = site+(left ? 1 : -1)
    inner_idx = only(commoninds(x[site],x[nextsite]))
    Q = x[site]
    Qp = prime(Q)
    replaceind!(Qp, prime(inner_idx), inner_idx)
    return dag(Q)*Qp, uniqueinds(Q,Qp)
end

function project_to_discarded(
    x::MPS,
    to_project::MPS,
    site::Integer,
    left::Bool
)
    _orthocenter_check(x, site, left)
    _orthocenter_check(to_project, site, left)
    p_d = discarded_space_projector(x, site, left)
    env = make_environment_at(x, to_project, left ? site-1 : site+1, !left)
    env *= to_project[site]
    env *= p_d
    if site == (left ? length(x) : 1)
        return env/norm(env)
    else
        q_inds = setdiff(inds(env), uniqueinds(to_project[site], inds(p_d)))
        Q, _ = qr(env, q_inds...)
        return noprime(Q)
    end
end

"""
Truncate x starting from `site_start` up to `site_end`:
```
-□--□--□--□- -> -∖--∖--∖--∘-
 |  |  |  |      |  |  |  |
 start    end
```
"""
function truncate_upto!(x::AbstractMPS, site_start::Integer, site_end::Integer; cutoff=nothing)
    sign = site_end >= site_start ? 1 : -1
    for site in site_start:sign:site_end-sign
        nextsite = site + sign
        U,S,V = svd(x[site]*x[nextsite], uniqueinds(x[site], x[nextsite])...; cutoff=cutoff)
        x[site] = U
        x[nextsite] = S*V
    end
    return x
end

function truncate_left!(x::AbstractMPS, site_end::Integer; cutoff=nothing)
    truncate_upto!(x, 1, site_end; cutoff=cutoff)
    x.llim = site_end-1
    return x
end

function truncate_right!(x::AbstractMPS, site_end::Integer; cutoff=nothing)
    truncate_upto!(x, length(x), site_end; cutoff=cutoff)
    x.rlim = site_end+1
    return x
end

"""
If `x` is site-canonical on bond `site`, truncate the neighboring bond in the direction specified by `left`.
Orthogonality centre remains the same.
"""
function truncate_bond!(
    x::AbstractMPS,
    site::Integer,
    left::Bool;
    svd_kwargs...
    )
    (xl,xr) = x.llim, x.rlim
    (x.llim==site-1) && (x.rlim==site+1) ||
        error("MPS must site-canonical on bond $site to truncate neighboring bonds (but llim=$(x.llim), rlim=$(x.rlim)).")
    (site==1 && left) && error("Cannot truncate left bond at first site.")
    (site==length(x) && !left) && error("Cannot truncate right bond at last site.")

    sign = left ? -1 : 1
    nextsite = site + sign
    x2 = x[site]*x[nextsite]
    U,S,V = svd(x2, uniqueinds(x[nextsite], x[site])...; svd_kwargs...)
    x.data[nextsite] = U
    x.data[site] = S*V
    x.llim = xl
    x.rlim = xr
    return x
end

"""
Expand bond after (depending on sweep direction) `site` based on residual.
`x` should already be orthogonalized on the next site.
"""
function expand_bond!(
    x::MPS,
    r::MPS,
    site::Integer,
    # whether discarded space has incoming link index on the left or right
    # if left=true, the link index on the right is expanded
    left::Bool;
    cutoff=nothing
)

    nextsite = site+(left ? 1 : -1)
    _orthocenter_check(x, site, left)

    expander = project_to_discarded(x, r, site, left)
    o_r, o_x = uniqueinds(expander, x[site]), uniqueinds(x[site], expander)
    # this will be the expanded tensor at site `site`
    iso_exp, s = ITensors.directsum(x[site] => o_x, expander => o_r)
    Q,R = if isnothing(cutoff)
            qr(iso_exp, setdiff(inds(iso_exp), s))
        else
            U,S,V = svd(iso_exp, setdiff(inds(iso_exp), s)...; cutoff=cutoff)
            U,S*V
        end
    x_next = (x[site] * dag(iso_exp)) * x[nextsite]

    # update MPS; can set x.data here because orthogonality at nextsite is ensured by QR decomposition
    x.data[site] = Q
    x.data[nextsite] = R*x_next
    return x
end

"""
Check whether `x` is orthogonal up to `site`.
"""
function _orthocenter_check(
    x::MPS,
    site::Integer,
    left::Bool
)
    if left
        site <= x.llim || error("MPS is not left-orthogonal up to site $site (llim=$(x.llim)).")
    else
        site >= x.rlim || error("MPS is not right-orthogonal up to site $site.")
    end
end

"""
Perform `nsweeps` 1-site AMEn sweeps to solve `(a0 + a1⋅A)x=b` for
positive definite `A`. `x` is updated in-place and returned.
# Arguments
- `amen_kargs`: Keyword arguments passed to `amen_update_1site!`. Concerns bond truncation and MPO-MPS multiplication for
the residual computation.
- `localupdate_kwargs`: Keyword arguments passed to `localupdate!`. Concerns the local solver (e.g., GMRES) settings.
"""
function amen!(
    A::MPO,
    x::MPS,
    b::MPS;
    a0=0.0,
    a1=1.0,
    nsweeps::Integer=5,
    _check_normchange=false,
    _do_check_orthogonality=false,
    amen_kwargs=_default_amen_kwargs(),
    howverbose=0,
    localupdate_kwargs...
)
    (a0>=0.0 && a1>0.0) || @warn "AMEn is intended for positive definite operators."
    if a0 != 0.0 || a1 != 1.0
        A = add_identity(A, a0, a1)
    end
    for i in 1:nsweeps
        howverbose>0 && printstyled("AMEn sweep $(i)/$(nsweeps)\n"; bold=true, color=:blue)
        # rightsweep
        halfsweep!(
            A, x, b, 1, false;
            _check_normchange=_check_normchange,
            _do_check_orthogonality=_do_check_orthogonality,
            do_amen_update=true,
            amen_kwargs=amen_kwargs,
            howverbose=howverbose,
            localupdate_kwargs...
        )
        # leftsweep
        halfsweep!(
            A, x, b, 1, true;
            _check_normchange=_check_normchange,
            _do_check_orthogonality=_do_check_orthogonality,
            do_amen_update=true,
            amen_kwargs=amen_kwargs,
            howverbose=howverbose,
            localupdate_kwargs...
        )
    end

    return x
end

function amen(
    A::MPO,
    x::MPS,
    b::MPS;
    a0=0.0,
    a1=1.0,
    nsweeps::Integer=5,
    _check_normchange=false,
    _do_check_orthogonality=false,
    amen_kwargs=(;),
    howverbose=0,
    localupdate_kwargs...
)::MPS
    x_copy = deepcopy(x)
    amen!(
        A, x_copy, b;
        a0=a0,
        a1=a1,
        nsweeps=nsweeps,
        _check_normchange=_check_normchange,
        _do_check_orthogonality=_do_check_orthogonality,
        amen_kwargs=amen_kwargs,
        howverbose=howverbose,
        localupdate_kwargs...
    )
    return x_copy
end

_default_amen_residual_apply_kwargs() = Dict(:alg=>"fit", :cutoff=>1.e-6, :nsweeps=>2, :maxdim=>40 )