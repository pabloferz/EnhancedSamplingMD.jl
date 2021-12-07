module EnhancedSamplingMD.jl


# Dependecies
include("backends/Backends.jl")
include("CollectiveVariables.jl")

using CUDA
using LinearAlgebra
using Reexport
using StaticArrays
@reexport using .CollectiveVariables
#@reexport using .Backends


# Imports
import .Backends: bind, set_backend, supported_backends


# Exports
export bind, set_backend, supported_backends, Grid


# Implementation
include("grids.jl")
include("sampling.jl")
include("utils.jl")


end  # module EnhancedSamplingMD.jl
