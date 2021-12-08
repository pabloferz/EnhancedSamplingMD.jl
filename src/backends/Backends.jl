# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: Pablo Zubieta
# See LICENSE.md at https://github.com/pabloferz/EnhancedSamplingMD.jl

module Backends


using PyCall


# Signal to Revise.jl that this module should be tracked as a package
__revise_mode__ = :eval


# Exports
export ContextWrapper, supported_backends


# Types
"""
Manages access to the backend-dependent simulation context.
"""
struct ContextWrapper
    backend::Module
    context::PyObject
    sampler::PyObject

    function ContextWrapper(context, sampling_method, callback; kwargs...)
        backend = load_backend(context)
        sampler = backend.bind(context, sampling_method, callback; kwargs...)
        return new(backend, context, sampler)
    end
end


# Methods
supported_backends() = (:HOOMD, :OpenMM)


"""    load_backend(context)

Loads the appropriate backend depending on the simulation context.
"""
function load_backend(context)
    module_name = py"type"(context).__module__
    backend_name = if startswith(module_name, "hoomd")
        :HOOMD
    elseif startswith(module_name, r"(simtk\.|)openmm")
        :OpenMM
    else
        :Invalid
    end
    if backend_name === :Invalid
        backends = join(supported_backends(), ", ")
        throw(ArgumentError("Invalid backend, supported options are $backends"))
    end
    path = joinpath(@__DIR__, string(backend, ".jl"))
    return include(path)::Module
end


end  # module Backends
