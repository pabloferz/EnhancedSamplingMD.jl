#using Pkg; Pkg.activate(first(splitdir(@__DIR__)))

using CUDA
using DLPack
using StaticArrays
using Zygote


#include("butane.jl")


# Advanced Sampling
#include(joinpath("..", "src", "SSAGES.jl"))
#using .SSAGES

# Collective Variables
ξ = SSAGES.DihedralAngle(1, 5, 8, 11)
# Define grid if necessary
grid = SSAGES.Grid(-π, π, 64, true)
# Sampling algorithm
sampling_method = SSAGES.ABF(ξ, grid, 100)

context = hoomd.context.current
sampler = SSAGES.bind(context, sampling_method)


# Run simulation
hoomd.run(100_000)
#hoomd.run(400_000)


# Data analysis
using Plots
using Statistics

θ = let #grid = sampler.grid
    Δ = only(grid.Δl)
    r = range(Δ / 2 + only(grid.r₀); step = Δ, length = only(grid.size))
    collect(r)
end

dA = let sm = sampler.sample_and_bias.update
    F = Array(vec(sm.ΣF) ./ max.(sm.hist, 1))
    dA = (F .- mean(F)) ./ std(F)
end

plot(θ, dA; label = "")
scatter!(θ, dA; label = "")

@inline function withgrad(f::F, ps) where {F}
    ξ, pb = Zygote._pullback(Zygote.Context(), f, ps...)
    ∂ξ = Zygote.tailmemaybe(pb(Zygote.sensitivity(ξ)))
    #ξ, back = Diffractor.∂⃖¹(f, ps)
    #∂ξ = back(ps)
    return ξ, SVector(∂ξ)
end

function test(sm)
    ss = sm.snapshot
    cv = sm.ξ
    rs = ss.positions
    ps = reinterpret(reshape, SVector{3, eltype(rs)}, rs)
    ids = cv.indices
    inds = cv.ξ.indices
    f = SSAGES.CollectiveVariables.cv_function(cv)
    g = ps -> withgrad(f, ps)
    #g = ps -> ForwardDiff.gradient(f, ps)
    #ξ = CUDA.fill(0.0)
    #u = CUDA.fill(zeros(SVector{3}), 4)
    #v = reinterpret(reshape, SVector{4, SVector{3, Float64}}, u)

    #function gputest(ps, ids, inds, u, v, x)
    #    i = threadIdx().x
    #    if i ≤ length(u)
    #        u[i] = ps[ids[inds[i]] + 1]
    #    end
    #    if i == length(u) + 1
    #        sync_threads()
    #        t = g(v[])
    #        x[] = t[1]
    #    end
    #    return nothing
    #end

    #@show g(SVector((SVector((rand() for _ in 1:3)...) for _ in 1:4)...))

    #@cuda threads=(length(u) + 1) gputest(ps, ids, inds, u, v, ξ)
    #CUDA.@allowscalar g(reinterpret(reshape, SVector{4, SVector{3, Float64}}, view(ps, view(ids, inds) .+ 1))[])
    #@device_code_warntype interactive=true gputest(g, ps, ids, inds, u, v, ξ)
    #return x, u
    val_and_grad(g, rs, ids, inds)
end

function val_and_grad(f::F, rs, ids, inds) where {F}
    # size(rs) == (3, 4)
    T = eltype(rs)
    #g = (p₁, p₂, p₃, p₄) -> withgrad(f, p₁, p₂, p₃, p₄)
    ps = reinterpret(reshape, SVector{3, T}, rs)
    x = CUDA.fill(zero(T))
    u = CUDA.fill(zeros(SVector{3}), 4)
    v = reinterpret(reshape, SVector{4, SVector{3, Float64}}, u)

    function kernel(x, u, v, ps, ids, inds)
        i = threadIdx().x
        @inbounds if i ≤ length(u)
            u[i] = ps[ids[inds[i]] + 1]
        end
        @inbounds if i == length(u) + 1
            sync_threads()
            ξ, ∂ξ = f(v[])
            x[] = ξ
            v[] = ∂ξ
        end
        return nothing
    end

    @cuda threads=(length(u) + 1) kernel(x, u, v, ps, ids, inds)

    return x, u
end

#let
#    sm = sampler.sample_and_bias.update
#    ss = sm.snapshot
#    cv = sm.ξ
#    rs = ss.positions
#    cv(rs)
#end

#test(sampler.sample_and_bias.update)
