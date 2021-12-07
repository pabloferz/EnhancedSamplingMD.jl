# Included into SamplingMethods.jl
#
# using SamplingMethods: CuSVTuple, SamplingMethod, SamplingMethodState, Snapshot


"""    ABF

Adaptive Biasing Force
"""
Base.@kwdef struct ABF{CV, G} <: SamplingMethod
    ξ::CV
    grid::G
    n::Int = 200
end

(self::ABF)(snapshot) = ABFState(self, snapshot)

struct ABFState{S, CV, G, B, H, U, V} <: SamplingMethodState
    snapshot::S
    ξ::CV
    grid::G
    bias::B
    hist::H
    ΣF::U
    F::V
    Wp::V
    Wp₋::V
    n::Int

    function ABFState(sampling_method::ABF, snapshot::S) where {S}
        grid = sampling_method.grid
        N = ndims(grid)
        shape = size(grid)

        ξ = DynamicCV(sampling_method.ξ, snapshot.positions, snapshot.ids)
        bias = fill!(similar(snapshot.forces), 0)
        hist = fill!(similar(snapshot.ids, shape), 0)
        ΣF = fill!(similar(snapshot.forces, (N, shape...)), 0)
        F = fill!(similar(snapshot.forces, N), 0)
        Wp = fill!(similar(snapshot.forces, N), 0)
        Wp₋ = fill!(similar(snapshot.forces, N), 0)

        return new{
            S, typeof(ξ), typeof(grid), typeof(bias), typeof(hist), typeof(ΣF), typeof(F)
        }(
            snapshot, ξ, grid, bias, hist, ΣF, F, Wp, Wp₋, sampling_method.n
        )
    end
end

(sm::ABFState)(sync_backend = nothing) = update!(sm; sync_backend = sync_backend)
#
function (sm::ABFState{<: Snapshot{<: CuSVTuple}})(; sync_backend = nothing)
    # When on GPU, make sure that forces are synchronized after each update.
    return CUDA.@sync update!(sm; sync_backend = sync_backend)
end

function update!(sm::ABFState{<: Snapshot}; sync_backend = nothing)
    ss = sm.snapshot
    rs = ss.positions
    vs = ss.velocities
    ms = ss.masses
    dt = ss.dt

    # Compute the collective variable and its Jacobian
    ξ, Jξ = sm.ξ(rs)
    # Compute momenta
    p = vec(ms' .* vs)
    #Wp₊ = Jξ \ p
    Wp₊ = ldiv(Jξ, p)
    # Second order backward finite difference
    Ẇp = (3 .* Wp₊ .- 4 .* sm.Wp .+ sm.Wp₋) ./ 2dt

    I_ξ = sm.grid[ξ]
    N_ξ = view(sm.hist, I_ξ)
    F_ξ = view(sm.ΣF, :, I_ξ)

    N_ξ .+= 1
    F_ξ .+= Ẇp .+ sm.F
    sm.F .= F_ξ ./ max.(N_ξ, sm.n)
    sm.Wp₋ .= sm.Wp
    sm.Wp .= Wp₊

    # Compute bias
    # TODO: Optimize with `mul!`
    sm.bias[:] .= reshape(Jξ, :, ndims(sm.grid)) * sm.F

    # TODO: Synchronization and biasing should be done outside
    # `update!(::SamplingMethodState)` so that if there are many sampling methods,
    # they can be run independently.

    # Forces may be computed asynchronously on the GPU, so we need to
    # synchronize them before applying the bias.
    if sync_backend !== nothing
        sync_backend()
    end
    # Add bias
    ss.forces .-= sm.bias

    return nothing
end
