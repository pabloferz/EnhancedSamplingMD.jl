using Pkg; Pkg.activate(first(splitdir(@__DIR__)))

using CUDA
using DLPack
using PyCall

@pyimport hoomd
@pyimport hoomd.md as md
@pyimport hoomd.dump as dump
@pyimport hoomd.group as group
@pyimport hoomd.benchmark as benchmark

@pyimport numpy as np

N = 10000
mode_string = "gpu"

if mode_string ∉ ("cpu", "gpu")
    throw("""Execution mode argument must be either "cpu" or "gpu".""")
end

# %%
hoomd.context.initialize("--mode=" * mode_string)
rcut = 3.0
sqrt_N = int(sqrt(N))

system = hoomd.init.create_lattice(
    unitcell = hoomd.lattice.sq(a = 2.0),
    n = [sqrt_N, sqrt_N]
)
nlist = md.nlist.cell(check_period = 1)
# basic LJ forces from Hoomd
lj = md.pair.lj(rcut, nlist)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
md.integrate.mode_standard(dt=0.005)
md.integrate.langevin(group = group.all(), kT=1.0, seed=42)
# equilibrate for 4k steps first
hoomd.run(4000)

# %%
import pysages

pysages.set_backend("hoomd")
simulation = hoomd.context.current
simulation_view = pysages.wrap(simulation)

# %%
ξ1 = pysages.cvs.distance(np.array([0, 1]))
ξ = pysages.collective_variable(ξ1)

grid = pysages.Grid(
    lower = (0,),
    upper = (2.0,),
    shape = (1024,),
    periodicity = (True,)
)

sampling_method = pysages.methods.abf(simulation_view, grid, ξ, N = 100)


# %%
sampler = pysages.bind(simulation, sampling_method)


# %%
hoomd.run(5000)


# %%
benchmark_results = hoomd.benchmark.series(
    warmup = 6000,
    repeat = 5,
    steps = 50000,
    limit_hours = 2
)