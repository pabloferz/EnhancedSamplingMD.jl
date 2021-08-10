module CollectiveVariables


# Signal to Revise.jl that this file should be tracked as a package
__revise_mode__ = :eval


# Dependecies
using ChainRulesCore
using CUDA
using Parameters
using StaticArrays
using Zygote


# Imports
using LinearAlgebra: ×, ⋅, norm
using Zygote: _pullback, sensitivity, tailmemaybe


# Exports
export DihedralAngle, DynamicCV, value_and_gradient


# Types
abstract type CVCache <: Function end

struct NoCache <: CVCache
end

struct ValGradCache{V, G, E} <: CVCache
    val::V
    grad::G
    extgrad::E

    function ValGradCache(cv, rs::AbstractArray{T}) where {T}
        val, V = withtype(value_storage(rs))
        grad, G = withtype(grad_storage(cv, rs))
        extgrad, E = withtype(reinterpret(reshape, SVector{3, T}, zero(rs)))
        return new{V, G, E}(val, grad, extgrad)
    end
end

abstract type CollectiveVariable <: Function end

abstract type GroupCV      <: CollectiveVariable end
abstract type FourPointCV  <: CollectiveVariable end
abstract type ThreePointCV <: CollectiveVariable end

struct DynamicCV{CV, T, C <: CVCache} <: CollectiveVariable
    ξ::CV
    indices::T
    cache::C

    function DynamicCV(cv::CV, rs, indices::T) where {CV, T}
        cache, C = withtype(ValGradCache(cv, rs))
        return new{CV, T, C}(cv, indices, cache)
    end

    function DynamicCV(cv::CV, rs, indices::T, cache::NoCache) where {CV, T}
        return new{CV, T, NoCache}(cv, indices, cache)
    end
end

struct DihedralAngle <: FourPointCV
    indices::SVector{4, Int}
end

DihedralAngle(i, j, k, l) = DihedralAngle(SVector(i, j, k, l))

# Methods
cv_function(::DihedralAngle) = dihedral_angle
cv_function(cv::DynamicCV) = cv_function(cv.ξ)

get_cache(cv::DynamicCV) = cv.cache

get_indices(cv::CollectiveVariable) = cv.indices

get_tags(cv::DihedralAngle) = cv.indices
get_tags(cv::DynamicCV) = get_tags(cv.ξ)

"""    dihedral_angle(p₁, p₂, p₃, p₄)

Computes the dihedral (or torsion) angle defined by four points in space (around the line
defined by the two central points).
"""
@inline dihedral_angle(p₁, p₂, p₃, p₄) = dihedral_angle(p₂ - p₁, p₃ - p₂, p₄ - p₃)
#
@inline function dihedral_angle(a, b, c)
    p = a × b
    q = b × c
    return atan((p × q) ⋅ b, (p ⋅ q) * norm(b))
end

# Gradient evaluation
get_value(cache::CVCache) = cv.val
get_gradient(cache::CVCache) = cv.grad
get_extended_gradient(cache::CVCache) = cv.extgrad

value_storage(rs::AbstractArray{T}) where {T} = Ref(zero(T))
value_storage(rs::AnyCuArray{T}) where {T} = CUDA.fill(zero(T))

grad_storage(cv::FourPointCV, rs::AbstractArray{T}) where {T} = nothing
#
function grad_storage(cv::FourPointCV, rs::AnyCuArray{T}) where {T}
    return CUDA.fill(zeros(SVector{3, T}), 4)
end

@inline function withstaticgrad(f::F, ps) where {F}
    ξ, pb = _pullback(Zygote.Context(), f, ps...)
    ∂ξ = tailmemaybe(pb(sensitivity(ξ)))
    return ξ, SVector(∂ξ)
end

function (cv::DynamicCV)(rs::AbstractMatrix{T}) where {T <: Real}
    I = get_indices(cv)
    tags = get_tags(cv)
    ps = reinterpret(reshape, SVector{3, T}, rs)
    f = ps -> withstaticgrad(cv_function(cv), ps)
    ξ, ∂ξ = evaluate!(f, get_cache(cv), ps, I, tags)
    return ξ, ∂ξ
end

function evaluate!(f::F, cache::ValGradCache, ps, I::AbstractVector, tags) where {F}
    ξ = get_value(cache)
    ∂p = get_extended_gradient(cache)
    fill!(∂p, zero(eltype(∂p)))

    inds = SVector(view(I, tags) .+ 1)
    ξ[], ∂ξ = f(view(ps, inds))
    ∂p[inds] .= ∂ξ

    return ξ[], vec(reinterpret(reshape, eltype(ξ), ∂p))
end

@inline function evaluate!(f::F, cache::ValGradCache, ps, I::AnyCuVector, tags) where {F}
    ξ = get_value(cache)
    ∂ξ = get_gradient(cache)
    ∂p = get_extended_gradient(cache)
    ∂x = reinterpret(reshape, SVector{4, eltype(ps)}, ∂ξ)
    fill!(∂p, zero(eltype(∂p)))

    function kernel(ξ, ∂ξ, ∂x, ∂p, ps, I)
        i = threadIdx().x
        @inbounds if i ≤ length(∂ξ)
            ∂ξ[i] = ps[I[tags[i]] + 1]
        end
        sync_threads()
        @inbounds if i == length(∂ξ) + 1
            ξ[], ∂x[] = f(∂x[])
        end
        sync_threads()
        @inbounds if i ≤ length(∂ξ)
            ∂p[I[tags[i]] + 1] = ∂ξ[i]
        end
        return nothing
    end

    @cuda threads=(length(∂ξ) + 1) kernel(ξ, ∂ξ, ∂x, ∂p, ps, I)

    return CUDA.@allowscalar(ξ[]), vec(reinterpret(reshape, eltype(ξ), ∂p))
end

function value_and_gradient(cv::CollectiveVariable, rs::AbstractMatrix{T}) where {T <: Real}
    ids = indices(cv)
    ps = reinterpret(reshape, SVector{3, T}, view(parent(rs), 1:3, ids))
    # TODO: Preallocate the memory for ∂ξ outside the function
    ∂ξ = reinterpret(reshape, SVector{3, T}, zero(rs))
    ξ, pb = _pullback(cv_function(cv), ps...)
    ∂p = tailmemaybe(pb(sensitivity(ξ)))
    if ∂p !== nothing
        ∂ξ[ids] .= ∂p
    end
    return ξ, vec(reinterpret(reshape, T, ∂ξ))
end


# Some (type-pirating ☠) bug fixes to ChainRulesCore

function ChainRulesCore.ProjectTo(x::AbstractArray{T}) where {T <: AbstractFloat}
    return ProjectTo{AbstractArray}(; element=ProjectTo(zero(T)), axes=axes(x))
end

function (project::ProjectTo{AbstractArray})(dx::AbstractArray{S}) where {S <: Number}
    T = ChainRulesCore.project_type(project.element)
    return S <: T ? dx : map(project.element, dx)
end

#function ChainRulesCore.rrule(
#    ::typeof(reinterpret), ::typeof(reshape), ::Type{R}, A::AbstractMatrix{T}
#) where {N, T, R <: SVector{N, T}}
#    y = reinterpret(reshape, R, A)
#    @inline function pullback(V̄)
#        ∂A = reinterpret(reshape, R, zero(A))
#        for (i, v) in pairs(V̄)
#            if v isa AbstractVector
#                ∂A[i] = v
#            end
#        end
#        #∂A = mapreduce(v -> v isa R ? v : zero(R), vcat, Ā; init = similar(A, 0))
#        return (NO_FIELDS, NO_FIELDS, DoesNotExist(), reinterpret(reshape, T, ∂A))
#    end
#    return y, pullback
#end

# Other utils

withtype(x::T) where {T} = (x, T)


end  # module CollectiveVariables
