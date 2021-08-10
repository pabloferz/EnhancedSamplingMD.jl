# Signal to Revise.jl that this file should be tracked as a package
#__revise_mode__ = :eval


# Types
abstract type AbstractGrid end

struct Grid{N, T} <: AbstractGrid
    r₀::SVector{N, T}
    l::SVector{N, T}
    Δl::SVector{N, T}
    size::SVector{N, Int}
    periodicity::SVector{N, Bool}

    function Grid(lower, upper, size, periodicity)
        return Grid(lower, upper, SVector(size), periodicity)
    end

    function Grid(lower, upper, size::SVector{N, <: Integer}, periodicity) where {N}
        l = SVector(upper .- lower)
        Δ = l ./ size
        T = eltype(Δ)
        return new{N, T}(SVector(lower), l, Δ, size, SVector(periodicity))
    end
end

function Base.show(io::IO, g::Grid{N, T}) where {N, T}
    context = IOContext(io, :compact => true)
    join(io, g.size, '×')
    print(io, "-element Grid{", N, ", ", T, "}:\n ")
    itr = Iterators.Stateful(zip(g.r₀, g.l))
    for (lo, l) in itr
        print(context, '[', lo, " .. ", lo + l, ']')
        if Base.peek(itr) !== nothing
            print(io, " × ")
        end
    end
    return nothing
end

function circ_index(r, r₀, l, Δl, m)
    n = floor(Int, mod(r - r₀, l) / Δl) + 1
    return ifelse(n > m, 1, n)
end

function Base.getindex(g::Grid{N}, r::Union{Real, NTuple{N, Real}, SVector{N}}) where {N}
    return CartesianIndex(Tuple(circ_index.(r, g.r₀, g.l, g.Δl, g.size)))
end

Base.size(g::Grid) = Tuple(g.size)
Base.ndims(::Grid{N}) where {N} = N
