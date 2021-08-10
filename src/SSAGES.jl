module SSAGES


# Signal to Revise.jl that this file should be tracked as a package
__revise_mode__ = :eval


# Dependecies
include("Backends.jl")
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


end  # module SSAGES
