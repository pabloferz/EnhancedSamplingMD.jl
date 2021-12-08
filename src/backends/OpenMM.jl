module HOOMD


# Signal Revise.jl that this module should be tracked as a package
__revise_mode__ = :eval


# Dependecies
using CUDA
using DLPack
using PyCall
using StaticArrays

DLExt = pyimport("hoomd.dlext")


# Types
struct ContextWrapper
    context::PyObject
    sysview::PyObject
    synchronize::PyObject

    function ContextWrapper(context::PyObject)
        sysview = @pycall DLExt.SystemView(context."system_definition")::PyObject
        return new(context, sysview, sysview.synchronize)
    end
end

@pydef mutable struct Sampler <: DLExt.HalfStepHook
    function __init__(self, sample_and_bias)
        py"super"(Sampler, self).__init__()
        self.sample_and_bias = sample_and_bias
        return nothing
    end
    function update(self, timestep)
        self.sample_and_bias()
        return nothing
    end
end


# Methods
is_on_gpu(context) = pycall(context.on_gpu, Bool)

location(::Type{Array}) = DLExt.AccessLocation."OnHost"
location(::Type{CuArray}) = DLExt.AccessLocation."OnDevice"

vector(::Type{A}, ::Type{T}, v) where {A, T} = unsafe_wrap(A, DLVector{T}(v))

function matrix(::Type{A}, ::Type{T}, M) where {A, T}
    array = unsafe_wrap(A, DLMatrix{T}(M))
    # HOOMD's particle data is stored in row-major format
    return reshape(array, reverse(size(array)))
end

function take_snapshot(wrapped_context; location = default_location())
    if is_on_gpu(wrapped_context.context) && location == DLExt.AccessLocation."OnDevice"
        return take_snapshot(CuArray, wrapped_context)
    else
        return take_snapshot(Array, wrapped_context)
    end
end

function take_snapshot(A::Union{Type{Array}, Type{CuArray}}, wrapped_context)
    context = wrapped_context.context
    sysview = wrapped_context.sysview
    mode = DLExt.AccessMode."ReadWrite"
    loc = location(A)
    T = Float64

    positions = matrix(A, T, @pycall DLExt.positions_types(sysview, loc, mode)::PyObject)
    momenta = matrix(A, T, @pycall DLExt.velocities_masses(sysview, loc, mode)::PyObject)
    forces = matrix(A, T, @pycall DLExt.net_forces(sysview, loc, mode)::PyObject)
    ids = vector(A, UInt32, @pycall DLExt.rtags(sysview, loc, mode)::PyObject)

    pdata = pycall(sysview."particle_data", PyObject)
    box = pycall(pdata."getGlobalBox", PyObject)

    xy = pycall(box."getTiltFactorXY", T)
    xz = pycall(box."getTiltFactorXZ", T)
    yz = pycall(box."getTiltFactorYZ", T)
    lo = pycall(box."getLo", PyObject)
    L = pycall(box."getL", PyObject)
    Lx, Ly, Lz = L."x", L."y", L."z"

    H = @SMatrix T[
        Lx   Ly * xy   Lz * xz ;
        0    Ly        Lz * yz ;
        0    0         Lz      ;
    ]
    origin = @SVector T[lo."x", lo."y", lo."z"]

    dt = convert(T, context."integrator"."dt")

    return (  # NamedTuple
        positions  = view(positions, 1:3, :),
        velocities = view(momenta, 1:3, :),
        masses     = view(momenta, 4, :),
        forces     = view(forces, 1:3, :),
        ids        = ids,
        box        = (H = H, origin = origin),
        dt         = dt,
    )
end

function bind(context, sampling_method, callback; kwargs...)
    wrapped_context = ContextWrapper(context)
    snapshot = take_snapshot(wrapped_context)
    update = sampling_method(snapshot)
    sample_and_bias = () -> update(; sync_backend = wrapped_context.synchronize)

    sampler = Sampler(sample_and_bias)
    context."integrator"."cpp_integrator"."setHalfStepHook"(sampler)
    return sampler
end


# Module Initialization
function __init__()
    if hasproperty(DLExt.AccessLocation, :OnDevice)
        @eval default_location() = DLExt.AccessLocation."OnDevice"
    else
        @eval default_location() = DLExt.AccessLocation."OnHost"
    end
end


end  # module HOOMD
