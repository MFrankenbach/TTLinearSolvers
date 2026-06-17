function init_environment(x::AbstractMPS)
    return ITensor(one(eltype(x[1])))
end

function update_environment(env::ITensor, x::MPS, A::MPO, site::Integer)
    return env * conj(x[site])' * A[site] * x[site]
end

function update_environment(env::ITensor, x::MPS, b::MPS, site::Integer)
    return env * b[site] * conj(x[site])
end

"""
Make all environments starting from left (leftsweep=false) or right (leftsweep=true) up to `lastsite`.
"""
function make_environments(x::MPS, Ab::AbstractMPS, lastsite::Integer, leftsweep::Bool=false)::Vector{ITensor}
    firstsite = leftsweep ? length(x) : 1
    envs = ITensor[]
    push!(envs, init_environment(x))
    step = leftsweep ? -1 : 1
    for site in firstsite:step:lastsite
        push!(envs, update_environment(envs[end], x, Ab, site))
    end
    return !leftsweep ? envs : reverse(envs)
end

"""
Make left environment (leftsweep=false) or right environment (leftsweep=true) such that the final tensors included in the environment are at `site`.
"""
function make_environment_at(x::MPS, Ab::AbstractMPS, site::Integer, leftsweep::Bool=false)::ITensor
    env = init_environment(x)
    firstsite = leftsweep ? length(x) : 1
    step = leftsweep ? -1 : 1
    for s in firstsite:step:site
        env = update_environment(env, x, Ab, s)
    end
    return env
end

_default_nsite() = 2