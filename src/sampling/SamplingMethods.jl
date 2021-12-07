module SamplingMethods


# Types
abstract type SamplingMethod <: Function end
abstract type SamplingMethodState <: Function end

const CuSVTuple = Tuple{
    AnyCuArray, AnyCuArray, AnyCuArray, AnyCuArray, AnyCuArray, NamedTuple, Real
}

const Snapshot{T} = NamedTuple{
    (:positions, :velocities, :masses, :forces, :ids, :box, :dt), T
}


include("abf.jl")


end  # module SamplingMethods
