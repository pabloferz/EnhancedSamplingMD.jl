module Backends


using PyCall


# Signal to Revise.jl that this module should be tracked as a package
__revise_mode__ = :eval


# Exports
export bind, current_backend, set_backend, supported_backends, wrap


# Constants
const CURRENT_BACKEND = Ref{Tuple{Symbol, Module}}()


# Methods
supported_backends() = (:HOOMD,)

"""    set_backend(backend::String)

To see a list of possible backends run `supported_backends()`
"""
set_backend(backend::AbstractString) = set_backend(Symbol(backend))

function set_backend(backend::Symbol)
    if backend ∈ supported_backends()
        if isassigned(CURRENT_BACKEND)
            current, m = CURRENT_BACKEND[]
            if backend === current
                return m
            end
        end
        m = initialize(backend)
        CURRENT_BACKEND[] = (backend, m)
        return m
    end
    throw(ArgumentError("Invalid backend"))
end

function initialize(backend::Symbol)
    path = joinpath(@__DIR__, "backends", string(backend, "Hook.jl"))
    m = include(path)::Module
    return m
end

function current_backend()
    if isassigned(CURRENT_BACKEND)
        return last(CURRENT_BACKEND[])
    end
    @warn "No backend has been set"
end

"""    bind(context, sampling_method; kwargs...)

Couples the sampling method to the simulation context.
"""
function bind(context, sampling_method; kwargs...)
    backend = if startswith(py"type"(context).__module__, "hoomd")
        set_backend(:HOOMD)
    end

    return Base.invokelatest(backend.bind, context, sampling_method; kwargs...)
end

function check_backend_initialization()
    if !isassigned(CURRENT_BACKEND)
        throw(ErrorException("No backend has been set"))
    end
end


end  # module Backends
